"""Delete an xpk workload from GKE.

Usage:
    python gke/delete_workload.py --infra gke/infra/v5e.yaml --workload-name pretrain-qwen3-swa-8b
"""

from __future__ import annotations

import argparse
import subprocess

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete xpk workload from GKE")
    parser.add_argument("--infra", required=True, help="Path to infra YAML (e.g. gke/infra/v5e.yaml)")
    parser.add_argument("--workload-name", required=True, help="Name of workload to delete")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    with open(args.infra) as f:
        infra = yaml.safe_load(f)

    cmd = [
        "xpk", "workload", "delete",
        "--workload", args.workload_name,
        "--project", infra["project_name"],
        "--zone", infra["zone"],
        "--cluster", infra["cluster_name"],
    ]

    print(f"  $ {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
