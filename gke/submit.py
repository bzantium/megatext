"""Submit xpk workloads to GKE.

Usage:
    python gke/submit.py pretrain --infra gke/infra/v5e.yaml gke/jobs/pretrain_qwen3_swa_8b_stage1.yaml
    python gke/submit.py autotune --infra gke/infra/v5e.yaml --include-sa-block gke/jobs/pretrain_qwen3_swa_8b_stage1.yaml
    python gke/submit.py run --infra gke/infra/v5e.yaml gke/jobs/test/test_pretrain.sh
"""

from __future__ import annotations

import argparse
import os
import subprocess

import yaml

from utils import get_libtpu_init_args

DEFAULT_IMAGE_TEMPLATE = "us-west4-docker.pkg.dev/{project}/megatext/megatext-tpu:{tag}"
DEFAULT_DOCKERFILE = "Dockerfile"

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_job_yaml(path: str) -> dict:
    """Load job YAML with variable substitution.

    The `vars` section defines variables that are substituted in all
    string values under `config` using ${VAR_NAME} syntax.
    """
    import re
    job = load_yaml(path)
    variables = job.pop("vars", {})
    if not variables:
        return job

    def _substitute(val):
        if isinstance(val, str):
            for k, v in variables.items():
                val = val.replace(f"${{{k}}}", str(v))
            return val
        if isinstance(val, dict):
            return {k: _substitute(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_substitute(v) for v in val]
        return val

    job["config"] = _substitute(job.get("config", {}))
    return job


def run_cmd(cmd: list[str], dry_run: bool = False, check: bool = True) -> None:
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=check)


def get_project_number(project: str) -> str:
    return subprocess.run(
        ["gcloud", "projects", "describe", project, "--format=value(projectNumber)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _config_to_args(config: dict) -> list[str]:
    """Convert config dict to key=value CLI args."""
    args = []
    for k, v in config.items():
        sv = str(v)
        if " " in sv:
            args.append(f'"{k}={sv}"')
        else:
            args.append(f"{k}={sv}")
    return args


def _docker_build(image: str, dockerfile: str, dry_run: bool) -> None:
    print("==> Building Docker image (local)")
    run_cmd(["docker", "build", "-t", image, "-f", dockerfile, "."], dry_run=dry_run)
    print("==> Pushing Docker image")
    run_cmd(["docker", "push", image], dry_run=dry_run)


def _xpk_submit(*, project, zone, cluster, workload_name, image, tpu_type,
                num_slices, priority, project_number, command, dry_run) -> None:
    xpk_cmd = [
        "xpk", "workload", "create",
        "--project", project, "--zone", zone, "--cluster", cluster,
        "--workload", workload_name,
        "--base-docker-image", image,
        "--tpu-type", tpu_type,
        "--num-slices", str(num_slices),
        "--priority", priority,
        "--command", command,
    ]
    if project_number:
        xpk_cmd.extend(["--project-number", project_number])
    print("==> Submitting xpk workload")
    run_cmd(xpk_cmd, dry_run=dry_run)
    print(f"\n==> Monitor with:")
    print(f"    xpk workload list --cluster {cluster} --project {project} --zone {zone}")


def _force_delete(*, project, zone, cluster, workload_name, dry_run) -> None:
    run_cmd([
        "xpk", "workload", "delete",
        "--project", project, "--zone", zone, "--cluster", cluster,
        "--workload", workload_name, "--force",
    ], dry_run=dry_run, check=False)


def _resolve_infra(args):
    """Resolve infrastructure config from infra YAML + CLI overrides."""
    infra = load_yaml(args.infra)
    return {
        "project": infra["project_name"],
        "zone": infra["zone"],
        "cluster": infra["cluster_name"],
        "tpu_type": args.tpu_type or infra["tpu_type"],
        "priority": args.priority or infra.get("priority", "medium"),
        "project_number": infra.get("project_number") or get_project_number(infra["project_name"]),
        "image": args.image or DEFAULT_IMAGE_TEMPLATE.format(project=infra["project_name"], tag="latest"),
    }


def _wrap_with_libtpu(tpu_type: str, inner_cmd: str) -> str:
    libtpu_args = get_libtpu_init_args(tpu_type)
    if libtpu_args:
        return f'export LIBTPU_INIT_ARGS="{libtpu_args}" && {inner_cmd}'
    return inner_cmd


# ── Subcommand: pretrain ─────────────────────────────────────────────────────

def cmd_pretrain(args) -> None:
    """Submit a training job from a YAML job definition."""
    infra = _resolve_infra(args)
    job = load_job_yaml(args.job)

    workload_name = args.workload_name or job["workload_name"]

    if args.build:
        _docker_build(infra["image"], args.dockerfile, args.dry_run)
    if args.force:
        _force_delete(project=infra["project"], zone=infra["zone"],
                      cluster=infra["cluster"], workload_name=workload_name, dry_run=args.dry_run)

    # Build bash command
    parts = ["set -euo pipefail", "bash gke/setup/preflight.sh"]
    if job.get("bucket"):
        parts.append(f"bash gke/setup/setup_gcsfuse.sh BUCKET={job['bucket']} MOUNT_PATH={job['mount_path']}")
    parts.append("python -m megatext.trainers.pretrain " + " ".join(_config_to_args(job["config"])))
    command = _wrap_with_libtpu(infra["tpu_type"], " && ".join(parts))

    _xpk_submit(
        project=infra["project"], zone=infra["zone"], cluster=infra["cluster"],
        workload_name=workload_name, image=infra["image"], tpu_type=infra["tpu_type"],
        num_slices=args.num_slices, priority=infra["priority"],
        project_number=infra["project_number"], command=command, dry_run=args.dry_run,
    )


# ── Subcommand: autotune ────────────────────────────────────────────────────

def cmd_autotune(args) -> None:
    """Submit an autotune job from a YAML job definition."""
    infra = _resolve_infra(args)
    job = load_job_yaml(args.job)

    workload_name = args.workload_name or f"autotune-{job['workload_name']}"

    if args.build:
        _docker_build(infra["image"], args.dockerfile, args.dry_run)
    if args.force:
        _force_delete(project=infra["project"], zone=infra["zone"],
                      cluster=infra["cluster"], workload_name=workload_name, dry_run=args.dry_run)

    # Build bash command (no gcsfuse — autotune uses synthetic data)
    parts = ["set -euo pipefail", "bash gke/setup/preflight.sh"]

    autotune_flags = [
        "--scope", args.scope,
        "--max-batch-size", str(args.max_batch_size),
        "--num-profile-steps", str(args.num_profile_steps),
        "--warmup-steps", str(args.warmup_steps),
    ]
    if args.include_sa_block:
        autotune_flags.append("--include-sa-block")

    parts.append(
        "python -m megatext.autotune.search "
        + " ".join(_config_to_args(job["config"]))
        + " " + " ".join(autotune_flags)
    )
    command = _wrap_with_libtpu(infra["tpu_type"], " && ".join(parts))

    _xpk_submit(
        project=infra["project"], zone=infra["zone"], cluster=infra["cluster"],
        workload_name=workload_name, image=infra["image"], tpu_type=infra["tpu_type"],
        num_slices=args.num_slices, priority=infra["priority"],
        project_number=infra["project_number"], command=command, dry_run=args.dry_run,
    )


# ── Subcommand: run (legacy bash scripts) ───────────────────────────────────

def cmd_run(args) -> None:
    """Submit a raw bash script (backward compat with old .sh jobs)."""
    infra = _resolve_infra(args)

    workload_name = args.workload_name or args.job.split("/")[-1].replace(".sh", "").replace("_", "-")

    if args.build:
        _docker_build(infra["image"], args.dockerfile, args.dry_run)
    if args.force:
        _force_delete(project=infra["project"], zone=infra["zone"],
                      cluster=infra["cluster"], workload_name=workload_name, dry_run=args.dry_run)

    command = _wrap_with_libtpu(infra["tpu_type"], f"bash {args.job}")

    _xpk_submit(
        project=infra["project"], zone=infra["zone"], cluster=infra["cluster"],
        workload_name=workload_name, image=infra["image"], tpu_type=infra["tpu_type"],
        num_slices=args.num_slices, priority=infra["priority"],
        project_number=infra["project_number"], command=command, dry_run=args.dry_run,
    )


# ── Subcommand: stop (suspend jobset) ────────────────────────────────────────

def cmd_stop(args) -> None:
    """Suspend a running workload (pods terminated, logs preserved)."""
    run_cmd([
        "kubectl", "patch", "jobset", args.workload_name,
        "--type=merge", "-p", '{"spec":{"suspend":true}}',
    ], dry_run=args.dry_run)
    print(f"Workload '{args.workload_name}' suspended. Logs are still accessible.")


# ── Subcommand: delete ──────────────────────────────────────────────────────

def cmd_delete(args) -> None:
    """Delete a workload."""
    infra = load_yaml(args.infra)
    run_cmd([
        "xpk", "workload", "delete",
        "--project", infra["project_name"],
        "--zone", infra["zone"],
        "--cluster", infra["cluster_name"],
        "--workload", args.workload_name,
        "--force",
    ], dry_run=args.dry_run)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit xpk workloads to GKE")
    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--infra", required=True, help="Path to infra YAML")
    common.add_argument("--build", action="store_true", help="Build & push Docker image")
    common.add_argument("--force", action="store_true", help="Delete existing workload first")
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--workload-name", default=None)
    common.add_argument("--num-slices", type=int, default=1)
    common.add_argument("--tpu-type", default=None,
                        help="Override TPU type from infra YAML")
    common.add_argument("--priority", default=None, choices=["low", "medium", "high"])
    common.add_argument("--image", default=None)
    common.add_argument("--dockerfile", default=DEFAULT_DOCKERFILE)

    sub = parser.add_subparsers(dest="command", required=True)

    # pretrain
    p_pretrain = sub.add_parser("pretrain", parents=[common], help="Submit training job from YAML")
    p_pretrain.add_argument("job", help="Path to job YAML")
    p_pretrain.set_defaults(func=cmd_pretrain)

    # autotune
    p_autotune = sub.add_parser("autotune", parents=[common], help="Submit autotune job from YAML")
    p_autotune.add_argument("job", help="Path to job YAML")
    p_autotune.add_argument("--scope", default="all", choices=["all", "batch_remat", "parallelism"])
    p_autotune.add_argument("--include-sa-block", action="store_true")
    p_autotune.add_argument("--max-batch-size", type=int, default=8)
    p_autotune.add_argument("--num-profile-steps", type=int, default=3)
    p_autotune.add_argument("--warmup-steps", type=int, default=3)
    p_autotune.set_defaults(func=cmd_autotune)

    # run (legacy bash)
    p_run = sub.add_parser("run", parents=[common], help="Submit raw bash script")
    p_run.add_argument("job", help="Path to bash script")
    p_run.set_defaults(func=cmd_run)

    # stop
    p_stop = sub.add_parser("stop", help="Suspend a running workload (logs preserved)")
    p_stop.add_argument("workload_name", help="Workload name to suspend")
    p_stop.add_argument("--dry-run", action="store_true")
    p_stop.set_defaults(func=cmd_stop)

    # delete
    p_delete = sub.add_parser("delete", help="Delete a workload")
    p_delete.add_argument("workload_name", help="Workload name to delete")
    p_delete.add_argument("--infra", required=True, help="Path to infra YAML")
    p_delete.add_argument("--dry-run", action="store_true")
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
