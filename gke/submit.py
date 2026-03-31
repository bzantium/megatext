"""Submit an xpk workload to GKE.

Usage:
    python gke/submit.py --infra gke/infra/v5e.yaml gke/jobs/train/pretrain_qwen3_8b.sh
    python gke/submit.py --infra gke/infra/v5e.yaml --build --force gke/jobs/train/autotune_qwen3_8b.sh
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile

import yaml

from utils import get_libtpu_init_args

# --- Docker defaults ---
DEFAULT_IMAGE_TEMPLATE = "us-west4-docker.pkg.dev/{project}/megatext/megatext-tpu:{tag}"
DEFAULT_DOCKERFILE = "Dockerfile"


def load_infra(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_project_number(project: str) -> str:
    """Get project number via gcloud (uses user credentials, not ADC)."""
    result = subprocess.run(
        ["gcloud", "projects", "describe", project, "--format=value(projectNumber)"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def run_cmd(cmd: list[str], dry_run: bool = False, check: bool = True) -> None:
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=check)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit xpk workload to GKE")
    parser.add_argument("job", help="Path to job bash script (e.g. gke/jobs/qwen3_8b.sh)")
    parser.add_argument("--infra", required=True, help="Path to infra YAML (e.g. gke/infra/v5e.yaml)")
    parser.add_argument("--build", action="store_true", help="Build & push docker image before submitting (local Docker)")
    parser.add_argument("--cloud-build", action="store_true", help="Build & push via Cloud Build (no local Docker needed)")
    parser.add_argument("--kaniko", action="store_true", help="Use kaniko with layer caching for --cloud-build (run setup_kaniko.sh first)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--workload-name", default=None, help="Override workload name")
    parser.add_argument("--num-slices", type=int, default=1, help="Number of TPU slices (default: 1)")
    parser.add_argument("--tpu-type", default=None, help="Override tpu_type")
    parser.add_argument("--priority", default=None, help="Override priority")
    parser.add_argument("--image", default=None, help="Docker image path (default: derived from project)")
    parser.add_argument("--dockerfile", default=DEFAULT_DOCKERFILE, help="Dockerfile path")
    parser.add_argument("--force", action="store_true", help="Delete existing workload before submitting")
    parser.add_argument("--libtpu-init-args", default=None,
                        help="Override LIBTPU_INIT_ARGS (default: auto from TPU type)")
    args = parser.parse_args()

    import os
    if not os.path.isfile(args.job):
        parser.error(f"Job script not found: {args.job}")

    infra = load_infra(args.infra)

    project = infra["project_name"]
    zone = infra["zone"]
    cluster_name = infra["cluster_name"]
    tpu_type = args.tpu_type or infra["tpu_type"]
    num_slices = args.num_slices
    priority = args.priority or infra.get("priority", "medium")
    project_number = infra.get("project_number") or get_project_number(project)
    image = args.image or DEFAULT_IMAGE_TEMPLATE.format(project=project, tag="latest")
    dockerfile = args.dockerfile

    workload_name = args.workload_name or args.job.split("/")[-1].replace(".sh", "").replace("_", "-")

    # --- Docker build & push ---
    region = zone.rsplit("-", 1)[0]  # us-west4-a -> us-west4
    if args.cloud_build and args.kaniko:
        print("==> Building Docker image via Cloud Build (kaniko cached)")
        kaniko_cache_repo = f"{region}-docker.pkg.dev/{project}/kaniko-cache"
        cloudbuild_config = {
            "steps": [{
                "name": "gcr.io/kaniko-project/executor:latest",
                "args": [
                    f"--destination={image}",
                    f"--cache=true",
                    f"--cache-repo={kaniko_cache_repo}",
                    f"--dockerfile={dockerfile}",
                    "--context=dir:///workspace",
                    "--snapshot-mode=redo",
                    "--compressed-caching=false",
                ],
            }],
            "timeout": "1800s",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="cloudbuild-", delete=False,
        ) as f:
            yaml.dump(cloudbuild_config, f)
            cloudbuild_path = f.name
        run_cmd([
            "gcloud", "builds", "submit",
            "--project", project,
            "--region", region,
            "--config", cloudbuild_path,
            ".",
        ], dry_run=args.dry_run)
    elif args.cloud_build:
        print("==> Building Docker image via Cloud Build")
        cloudbuild_config = {
            "steps": [{
                "name": "gcr.io/cloud-builders/docker",
                "args": [
                    "build",
                    "-t", image,
                    "-f", dockerfile,
                    ".",
                ],
            }],
            "images": [image],
            "timeout": "1800s",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="cloudbuild-", delete=False,
        ) as f:
            yaml.dump(cloudbuild_config, f)
            cloudbuild_path = f.name
        run_cmd([
            "gcloud", "builds", "submit",
            "--project", project,
            "--region", region,
            "--config", cloudbuild_path,
            ".",
        ], dry_run=args.dry_run)
    elif args.build:
        print("==> Building Docker image (local)")
        build_cmd = ["docker", "build", "-t", image, "-f", dockerfile, "."]
        run_cmd(build_cmd, dry_run=args.dry_run)
        print("==> Pushing Docker image")
        run_cmd(["docker", "push", image], dry_run=args.dry_run)

    # --- Delete existing workload if --force ---
    if args.force:
        print("==> Deleting existing workload (if any)")
        run_cmd([
            "xpk", "workload", "delete",
            "--project", project,
            "--zone", zone,
            "--cluster", cluster_name,
            "--workload", workload_name,
            "--force",
        ], dry_run=args.dry_run, check=False)

    # --- Resolve LIBTPU_INIT_ARGS for this TPU type ---
    libtpu_args = args.libtpu_init_args or get_libtpu_init_args(tpu_type)

    # Build command with optional LIBTPU_INIT_ARGS export
    if libtpu_args:
        command = f'export LIBTPU_INIT_ARGS="{libtpu_args}" && bash {args.job}'
    else:
        command = f"bash {args.job}"

    # --- xpk workload create ---
    xpk_cmd = [
        "xpk", "workload", "create",
        "--project", project,
        "--zone", zone,
        "--cluster", cluster_name,
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
    run_cmd(xpk_cmd, dry_run=args.dry_run)

    print(f"\n==> Monitor with:")
    print(f"    xpk workload list --cluster {cluster_name} --project {project} --zone {zone}")


if __name__ == "__main__":
    main()
