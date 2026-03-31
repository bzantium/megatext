#!/bin/bash
# Mount a GCS bucket via gcsfuse with ML-optimized cache settings.
#
# Usage:
#   bash gke/setup/setup_gcsfuse.sh BUCKET=my-dataset-bucket MOUNT_PATH=/tmp/gcsfuse
#   bash gke/setup/setup_gcsfuse.sh BUCKET=my-dataset-bucket MOUNT_PATH=/tmp/gcsfuse FILE_PATH=/tmp/gcsfuse/my_dataset
set -e

# Parse KEY=VALUE arguments
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ -z "${BUCKET}" || -z "${MOUNT_PATH}" ]]; then
    echo "Error: BUCKET and MOUNT_PATH are required."
    echo "Usage: bash setup_gcsfuse.sh BUCKET=<bucket-name> MOUNT_PATH=<path> [FILE_PATH=<path>]"
    exit 1
fi

# Install gcsfuse if not present
if ! command -v gcsfuse >/dev/null 2>&1; then
    echo "Installing gcsfuse..."
    SUDO=""
    if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
    GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
    $SUDO install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | $SUDO gpg --dearmor -o /etc/apt/keyrings/gcsfuse.gpg
    echo "deb [signed-by=/etc/apt/keyrings/gcsfuse.gpg] https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" \
        | $SUDO tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq gcsfuse
    echo "gcsfuse installed."
else
    echo "gcsfuse already installed."
fi

# Strip gs:// prefix if present
if [[ "$BUCKET" =~ gs:\/\/ ]]; then
    BUCKET="${BUCKET/gs:\/\//}"
    echo "Removed gs:// from bucket name, bucket is $BUCKET"
fi

# Clean up existing mount
if [[ -d "${MOUNT_PATH}" ]]; then
    echo "$MOUNT_PATH exists, removing..."
    fusermount -u "$MOUNT_PATH" || rm -rf "$MOUNT_PATH"
fi

mkdir -p "${MOUNT_PATH}"

# Build app-name for gcsfuse metrics
MEGATEXT_VERSION=$(pip list 2>/dev/null | grep '^megatext ' | awk '{print $2}')
MEGATEXT_VERSION="${MEGATEXT_VERSION:-unknown}"
GRAIN_VERSION=$(pip list 2>/dev/null | grep '^grain ' | awk '{print $2}')
GRAIN_VERSION="${GRAIN_VERSION:-unknown}"
APP_NAME="megatext-gcsfuse/megatext-$MEGATEXT_VERSION/grain-$GRAIN_VERSION"

# Mount with aggressive metadata caching (read-only, optimized for ML reads)
# See https://cloud.google.com/storage/docs/gcsfuse-cli
TIMESTAMP=$(date +%Y%m%d-%H%M)
gcsfuse --implicit-dirs --log-severity=debug \
    --type-cache-max-size-mb=-1 \
    --stat-cache-max-size-mb=-1 \
    --kernel-list-cache-ttl-secs=-1 \
    --metadata-cache-ttl-secs=-1 \
    --log-file="$HOME/gcsfuse_$TIMESTAMP.json" \
    --app-name="$APP_NAME" \
    "$BUCKET" "$MOUNT_PATH"

echo "Mounted gs://${BUCKET} at ${MOUNT_PATH}"

# Prefill metadata cache: https://cloud.google.com/storage/docs/cloud-storage-fuse/performance#improve-first-time-reads
if [[ -n "${FILE_PATH}" ]]; then
    FILE_COUNT=$(ls -R "$FILE_PATH" | wc -l)
    echo "$FILE_COUNT files found in $FILE_PATH"
fi
