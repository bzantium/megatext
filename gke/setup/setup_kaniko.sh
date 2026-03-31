#!/bin/bash
# Create an Artifact Registry repo for kaniko layer caching in Cloud Build.
#
# Run this once before using: python gke/submit.py --cloud-build --kaniko ...
#
# Usage:
#   bash gke/setup/setup_kaniko.sh PROJECT=my-project ZONE=us-west4-a
set -euo pipefail

# --- Parse arguments ---
PROJECT="${PROJECT:-}"
ZONE="${ZONE:-us-west4-a}"

for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [ -z "$PROJECT" ]; then
    echo "Error: PROJECT is required."
    echo "Usage: bash gke/setup/setup_kaniko.sh PROJECT=my-project [ZONE=us-west4-a]"
    exit 1
fi

REGION="${ZONE%-*}"  # us-west4-a -> us-west4
CACHE_REPO="kaniko-cache"

echo "==> Setting up kaniko cache for project=$PROJECT region=$REGION"

if gcloud artifacts repositories describe "$CACHE_REPO" \
    --project="$PROJECT" --location="$REGION" >/dev/null 2>&1; then
    echo "Repo ${REGION}-docker.pkg.dev/${PROJECT}/${CACHE_REPO} already exists."
else
    echo "Creating Artifact Registry repo for kaniko cache..."
    gcloud artifacts repositories create "$CACHE_REPO" \
        --project="$PROJECT" \
        --location="$REGION" \
        --repository-format=docker \
        --description="Kaniko layer cache for Cloud Build"
    # Set 14-day cleanup policy
    gcloud artifacts repositories set-cleanup-policies "$CACHE_REPO" \
        --project="$PROJECT" \
        --location="$REGION" \
        --policy=<(cat <<'POLICY'
[
  {
    "name": "delete-old-cache",
    "action": {"type": "Delete"},
    "condition": {"olderThan": "1209600s"}
  }
]
POLICY
)
    echo "Repo created with 14-day cleanup policy."
fi

echo ""
echo "==> Done. Use with:"
echo "    python gke/submit.py --cloud-build --kaniko --infra gke/infra/v5e.yaml gke/jobs/my_job.sh"
