"""Submit a checkpoint conversion job to GKE (single-host TPU).

Generates a Kubernetes Job manifest from a YAML job definition and applies it.
Runs on a single TPU host without xpk's multi-host coordination.

Usage:
    python gke/submit_convert.py --infra gke/infra/v5e.yaml gke/jobs/convert/example.yaml
    python gke/submit_convert.py --infra gke/infra/v5e.yaml --force gke/jobs/convert/example.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile

import yaml

DEFAULT_IMAGE_TEMPLATE = "us-west4-docker.pkg.dev/{project}/megatext/megatext-tpu:{tag}"
DEFAULT_DOCKERFILE = "Dockerfile"

# Chips per host bounds for each TPU generation (single host).
_CHIPS_PER_HOST = {
    "v4": (2, 2, 1),     # 4 chips/host
    "v5e": (2, 2, 1),    # 4 chips/host
    "v5litepod": (2, 2, 1),
    "v5p": (2, 2, 1),    # 4 chips/host
    "v6e": (2, 2, 1),    # 4 chips/host
    "tpu7x": (2, 2, 1),  # 4 chips/host (ironwood)
}


def _parse_tpu_topology(tpu_type: str) -> tuple[tuple[int, int, int], int]:
    """Parse tpu_type (e.g. 'v5litepod-256') into (chips_per_host_bounds, num_chips_per_host)."""
    # Strip the chip count suffix: "v5litepod-256" -> "v5litepod"
    parts = tpu_type.rsplit("-", 1)
    generation = parts[0]

    bounds = _CHIPS_PER_HOST.get(generation)
    if bounds is None:
        raise ValueError(
            f"Unknown TPU generation '{generation}' from tpu_type='{tpu_type}'. "
            f"Known: {list(_CHIPS_PER_HOST.keys())}"
        )
    num_chips = bounds[0] * bounds[1] * bounds[2]
    return bounds, num_chips


def load_infra(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd: list[str], dry_run: bool = False, check: bool = True) -> None:
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=check)


def build_job_manifest(
    name: str,
    image: str,
    command: str,
    tpu_type: str,
    nodepool: str | None = None,
    service_account: str = "lmt-ksa",
    gcs_bucket: str = "lmt-tpu-datasets",
) -> dict:
    """Build a Kubernetes Job manifest for a single-host TPU job."""
    script_content = f"set -euo pipefail && bash gke/setup/preflight.sh && {command}"

    chip_bounds, num_chips = _parse_tpu_topology(tpu_type)
    chip_bounds_str = ",".join(str(d) for d in chip_bounds)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "annotations": {
                        "gke-gcsfuse/volumes": "true",
                        "gke-gcsfuse/cpu-limit": "0",
                        "gke-gcsfuse/memory-limit": "0",
                        "gke-gcsfuse/ephemeral-storage-limit": "0",
                    },
                },
                "spec": {
                    "serviceAccountName": service_account,
                    "restartPolicy": "Never",
                    **({"nodeSelector": {"cloud.google.com/gke-nodepool": nodepool}} if nodepool else {}),
                    "containers": [{
                        "name": "worker",
                        "image": image,
                        "resources": {"limits": {"google.com/tpu": num_chips}},
                        "env": [
                            {"name": "JAX_PLATFORMS", "value": "tpu"},
                            {"name": "TPU_CHIPS_PER_HOST_BOUNDS", "value": chip_bounds_str},
                            {"name": "TPU_HOST_BOUNDS", "value": "1,1,1"},
                            {"name": "TPU_WORKER_ID", "value": "0"},
                            {"name": "TPU_WORKER_HOSTNAMES", "value": "localhost"},
                        ],
                        "command": ["/bin/bash", "-c"],
                        "args": [script_content],
                        "volumeMounts": [
                            {"name": "gcs-fuse", "mountPath": "/data"},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                    }],
                    "tolerations": [{
                        "key": "google.com/tpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }],
                    "volumes": [
                        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
                        {
                            "name": "gcs-fuse",
                            "csi": {
                                "driver": "gcsfuse.csi.storage.gke.io",
                                "volumeAttributes": {
                                    "bucketName": gcs_bucket,
                                    "fileCacheCapacity": "100Gi",
                                    "fileCacheForRangeRead": "true",
                                    "mountOptions": "implicit-dirs,uid=0,gid=0",
                                },
                            },
                        },
                    ],
                },
            },
        },
    }


def get_project_number(project: str) -> str:
    result = subprocess.run(
        ["gcloud", "projects", "describe", project, "--format=value(projectNumber)"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a single-host job to GKE")
    parser.add_argument("job", help="Path to job YAML (e.g. gke/jobs/convert/example.yaml)")
    parser.add_argument("--infra", required=True, help="Path to infra YAML (e.g. gke/infra/v5e.yaml)")
    parser.add_argument("--build", action="store_true", help="Build & push Docker image before submitting")
    parser.add_argument("--force", action="store_true", help="Delete existing job before submitting")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest without applying")
    parser.add_argument("--workload-name", default=None, help="Override job name")
    parser.add_argument("--tpu-type", default=None, help="Override tpu_type from infra YAML")
    parser.add_argument("--nodepool", default=None, help="Target nodepool")
    parser.add_argument("--image", default=None, help="Docker image (default: derived from project)")
    parser.add_argument("--dockerfile", default=DEFAULT_DOCKERFILE, help="Dockerfile path")
    args = parser.parse_args()

    infra = load_infra(args.infra)
    job = load_infra(args.job)

    project = infra["project_name"]
    tpu_type = args.tpu_type or infra["tpu_type"]
    image = args.image or DEFAULT_IMAGE_TEMPLATE.format(project=project, tag="latest")
    nodepool = args.nodepool or infra.get("nodepool")
    workload_name = args.workload_name or job["workload_name"]

    # Build & push
    if args.build:
        print("==> Building Docker image (local)")
        run_cmd(["docker", "build", "-t", image, "-f", args.dockerfile, "."], dry_run=args.dry_run)
        print("==> Pushing Docker image")
        run_cmd(["docker", "push", image], dry_run=args.dry_run)

    # Build command from args
    job_args = job.get("args", {})
    direction = job_args.pop("direction", "hf-to-megatext")
    cli_parts = [f"python -m megatext.conversion.convert {direction}"]
    for k, v in job_args.items():
        if isinstance(v, bool):
            if v:
                cli_parts.append(f"--{k}")
        else:
            cli_parts.append(f"--{k} {v}")
    command = " ".join(cli_parts)

    # Generate manifest
    manifest = build_job_manifest(
        name=workload_name,
        image=image,
        command=command,
        tpu_type=tpu_type,
        nodepool=nodepool,
        gcs_bucket=job.get("bucket", "lmt-tpu-datasets"),
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"convert-{workload_name}-", delete=False,
    ) as f:
        yaml.dump(manifest, f, default_flow_style=False)
        manifest_path = f.name

    if args.dry_run:
        print(f"==> Generated manifest: {manifest_path}")
        with open(manifest_path) as f:
            print(f.read())
        return

    # Delete existing job if --force
    if args.force:
        print(f"==> Deleting existing job '{workload_name}' (if any)")
        run_cmd(["kubectl", "delete", "job", workload_name, "--ignore-not-found"], dry_run=args.dry_run)

    # Apply
    print(f"==> Applying job manifest")
    run_cmd(["kubectl", "apply", "-f", manifest_path], dry_run=args.dry_run)

    print(f"\n==> Monitor with:")
    print(f"    kubectl logs -f job/{workload_name}")


if __name__ == "__main__":
    main()
