"""Submit xpk workloads to GKE.

Usage:
    python gke/submit.py pretrain --infra gke/infra/v5e.yaml gke/jobs/pretrain_qwen3_swa_8b_stage1.yaml
    python gke/submit.py profile --infra gke/infra/v5e.yaml gke/jobs/train/pretrain_qwen3_8b_stage1.yaml
    python gke/submit.py autotune --infra gke/infra/v5e.yaml --include-sa-block gke/jobs/pretrain_qwen3_swa_8b_stage1.yaml
    python gke/submit.py run --infra gke/infra/v5e.yaml gke/jobs/test/test_pretrain.sh
"""

from __future__ import annotations

import argparse
import re
import subprocess

import yaml

try:
    from utils import get_libtpu_init_args
except ImportError:
    from gke.utils import get_libtpu_init_args

DEFAULT_IMAGE_TEMPLATE = "us-west4-docker.pkg.dev/{project}/megatext/megatext-tpu:{tag}"
DEFAULT_DOCKERFILE = "Dockerfile"
MAX_XPK_WORKLOAD_NAME_LEN = 39
PROFILE_DEFAULT_STEPS = 8
PROFILE_DEFAULT_SKIP_STEPS = 3
PROFILE_DEFAULT_PROFILE_STEPS = 3
SMOKE_RUN_STEPS = 5
SMOKE_RUN_WARMUP_STEPS = 0

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
                num_slices, priority, project_number, command, dry_run,
                show_monitor_hint: bool = True) -> None:
    workload_name = _sanitize_workload_name(workload_name)
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
    if show_monitor_hint:
        print(f"\n==> Monitor with:")
        print(f"    xpk workload list --cluster {cluster} --project {project} --zone {zone}")


def _force_delete(*, project, zone, cluster, workload_name, dry_run) -> None:
    workload_name = _sanitize_workload_name(workload_name)
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


def _sanitize_workload_name(name: str, max_len: int = MAX_XPK_WORKLOAD_NAME_LEN) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = re.sub(r"^[^a-z]+", "", sanitized)
    if not sanitized:
        sanitized = "xpk-workload"
    sanitized = sanitized[:max_len].rstrip("-")
    if not sanitized:
        sanitized = "xpk"
    if not sanitized[0].isalpha():
        sanitized = f"x{sanitized}"
    sanitized = sanitized[:max_len].rstrip("-")
    if not sanitized[-1].isalnum():
        sanitized = sanitized.rstrip("-")
    if not sanitized:
        sanitized = "xpk"
    return sanitized


def _prepend_prefix(value: str, prefix: str) -> str:
    return value if value.startswith(prefix) else f"{prefix}{value}"


def _prefix_job_names(job: dict, prefix: str) -> dict:
    prefixed_job = dict(job)
    config = dict(job.get("config", {}))
    if "run_name" in config:
        config["run_name"] = _prepend_prefix(str(config["run_name"]), prefix)
    prefixed_job["config"] = config
    if "workload_name" in prefixed_job:
        prefixed_job["workload_name"] = _prepend_prefix(str(prefixed_job["workload_name"]), prefix)
    return prefixed_job


def _build_profile_job(job: dict, *, steps: int, profiler_steps: int, skip_first_steps: int, dataset_type: str) -> dict:
    profile_job = dict(job)
    config = dict(job.get("config", {}))
    config.update({
        "dataset_type": dataset_type,
        "steps": steps,
        "profiler": "xplane",
        "skip_first_n_steps_for_profiler": skip_first_steps,
        "profiler_steps": profiler_steps,
        "profile_cleanly": True,
        "enable_checkpointing": False,
        "save_checkpoint_on_completion": False,
        "eval_interval": -1,
        "log_period": 1,
        # Force an uncached compile path for cleaner compile-time comparisons.
        "jax_cache_dir": "",
    })
    if dataset_type == "synthetic":
        # Synthetic profiling should not carry train-data mounts or dataset paths.
        config.pop("dataset_path", None)
        config.pop("data_cache_dir", None)
        profile_job.pop("bucket", None)
        profile_job.pop("mount_path", None)
    profile_job["config"] = config
    return _prefix_job_names(profile_job, "profile-")


def _build_smoke_pretrain_job(job: dict, *, steps: int, warmup_steps: int) -> dict:
    smoke_job = dict(job)
    config = dict(job.get("config", {}))
    config.update({
        "steps": steps,
        "warmup_steps": warmup_steps,
        "global_batch_size": None,
        "gradient_accumulation_steps": 1,
        "enable_checkpointing": False,
        "save_checkpoint_on_completion": False,
        "eval_interval": -1,
        "enable_tensorboard": False,
        "use_vertex_tensorboard": False,
        "gcs_metrics": False,
        "metrics_file": "",
        "save_config_to_gcs": False,
    })
    smoke_job["config"] = config
    return _prefix_job_names(smoke_job, "smoke-")


def _build_autotune_job(job: dict) -> dict:
    return _prefix_job_names(job, "autotune-")


def _apply_config_overrides(job: dict, overrides: list[str] | None) -> dict:
    if not overrides:
        return job

    overridden_job = dict(job)
    config = dict(job.get("config", {}))
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid config override {override!r}; expected KEY=VALUE.")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid config override {override!r}; key cannot be empty.")
        config[key] = yaml.safe_load(raw_value)
    overridden_job["config"] = config
    return overridden_job




def _submit_autotune_workload(args, infra: dict, job: dict, workload_name: str) -> None:
    if args.force:
        _force_delete(project=infra["project"], zone=infra["zone"],
                      cluster=infra["cluster"], workload_name=workload_name, dry_run=args.dry_run)

    parts = ["set -euo pipefail", "bash gke/setup/preflight.sh"]

    autotune_flags = [
        "--scope", args.scope,
        "--max-batch-size", str(args.max_batch_size),
        "--num-profile-steps", str(args.num_profile_steps),
        "--warmup-steps", str(args.warmup_steps),
    ]
    if args.include_sa_block:
        autotune_flags.append("--include-sa-block")
    if args.refine_sa_backward:
        autotune_flags.append("--refine-sa-backward")

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

# ── Subcommand: pretrain ─────────────────────────────────────────────────────

def cmd_pretrain(args) -> None:
    """Submit a training job from a YAML job definition."""
    infra = _resolve_infra(args)
    job = load_job_yaml(args.job)
    if args.smoke_run:
        job = _build_smoke_pretrain_job(
            job,
            steps=SMOKE_RUN_STEPS,
            warmup_steps=SMOKE_RUN_WARMUP_STEPS,
        )
    job = _apply_config_overrides(job, args.set)

    workload_name = args.workload_name or job["workload_name"]
    if args.smoke_run:
        workload_name = _prepend_prefix(str(workload_name), "smoke-")

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


def cmd_profile(args) -> None:
    """Submit a short profiling job using a training YAML plus profile overrides."""
    infra = _resolve_infra(args)
    base_job = load_job_yaml(args.job)
    job = _build_profile_job(
        base_job,
        steps=args.steps,
        profiler_steps=args.profiler_steps,
        skip_first_steps=args.skip_first_steps,
        dataset_type=args.dataset_type,
    )
    job = _apply_config_overrides(job, args.set)

    workload_name = args.workload_name or job["workload_name"]

    if args.build:
        _docker_build(infra["image"], args.dockerfile, args.dry_run)
    if args.force:
        _force_delete(project=infra["project"], zone=infra["zone"],
                      cluster=infra["cluster"], workload_name=workload_name, dry_run=args.dry_run)

    parts = ["set -euo pipefail", "bash gke/setup/preflight.sh"]
    if job.get("bucket"):
        parts.append(f"bash gke/setup/setup_gcsfuse.sh BUCKET={job['bucket']} MOUNT_PATH={job['mount_path']}")
    parts.append("python -m megatext.trainers.profile " + " ".join(_config_to_args(job["config"])))
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
    job = _build_autotune_job(load_job_yaml(args.job))
    job = _apply_config_overrides(job, args.set)

    workload_name = _prepend_prefix(str(args.workload_name or job["workload_name"]), "autotune-")

    if args.build:
        _docker_build(infra["image"], args.dockerfile, args.dry_run)
    _submit_autotune_workload(args, infra, job, workload_name)


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


# ── Subcommand: delete ──────────────────────────────────────────────────────

def cmd_delete(args) -> None:
    """Delete a workload."""
    infra = load_yaml(args.infra)
    workload_name = _sanitize_workload_name(args.workload_name)
    run_cmd([
        "xpk", "workload", "delete",
        "--project", infra["project_name"],
        "--zone", infra["zone"],
        "--cluster", infra["cluster_name"],
        "--workload", workload_name,
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
    common.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a top-level config key in the job YAML. Can be passed multiple times.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # pretrain
    p_pretrain = sub.add_parser("pretrain", parents=[common], help="Submit training job from YAML")
    p_pretrain.add_argument("job", help="Path to job YAML")
    p_pretrain.add_argument(
        "--smoke-run",
        action="store_true",
        help=f"Override train config with steps={SMOKE_RUN_STEPS} and warmup_steps={SMOKE_RUN_WARMUP_STEPS}",
    )
    p_pretrain.set_defaults(func=cmd_pretrain)

    # profile
    p_profile = sub.add_parser("profile", parents=[common], help="Submit profiling job from YAML")
    p_profile.add_argument("job", help="Path to job YAML")
    p_profile.add_argument("--steps", type=int, default=PROFILE_DEFAULT_STEPS)
    p_profile.add_argument("--profiler-steps", type=int, default=PROFILE_DEFAULT_PROFILE_STEPS)
    p_profile.add_argument("--skip-first-steps", type=int, default=PROFILE_DEFAULT_SKIP_STEPS)
    p_profile.add_argument("--dataset-type", default="synthetic")
    p_profile.set_defaults(func=cmd_profile)

    # autotune
    p_autotune = sub.add_parser("autotune", parents=[common], help="Submit autotune job from YAML")
    p_autotune.add_argument("job", help="Path to job YAML")
    p_autotune.add_argument("--scope", default="batch_remat", choices=["batch_remat", "parallelism", "all"])
    p_autotune.add_argument("--include-sa-block", action="store_true")
    p_autotune.add_argument("--refine-sa-backward", action="store_true")
    p_autotune.add_argument("--max-batch-size", type=int, default=8)
    p_autotune.add_argument("--num-profile-steps", type=int, default=3)
    p_autotune.add_argument("--warmup-steps", type=int, default=3)
    p_autotune.set_defaults(func=cmd_autotune)

    # run (legacy bash)
    p_run = sub.add_parser("run", parents=[common], help="Submit raw bash script")
    p_run.add_argument("job", help="Path to bash script")
    p_run.set_defaults(func=cmd_run)

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
