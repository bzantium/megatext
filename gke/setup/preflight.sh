#!/bin/bash
# Vendored from: AI-Hypercomputer/maxtext (preflight.sh)
# Pre-flight setup: network optimization via rto_setup.sh.
#
# Usage:
#   bash gke/setup/preflight.sh
echo "Running preflight.sh"

# Stop execution if any command exits with error
set -e

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

SCRIPT_DIR="$(dirname "$0")"

# Check if sudo is available
if command -v sudo >/dev/null 2>&1; then
    echo "running rto_setup.sh with sudo"
    sudo bash "${SCRIPT_DIR}/rto_setup.sh"
else
    echo "running rto_setup.sh without sudo"
    bash "${SCRIPT_DIR}/rto_setup.sh"
fi
