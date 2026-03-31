#!/bin/bash
# Setup gcloud credentials and tools for xpk workload submission.
#
# Run this once on your local machine or TPU VM before using:
#   python gke/submit.py --infra gke/infra/v5e.yaml gke/jobs/my_job.sh
#
# Usage:
#   bash gke/setup/setup_xpk.sh PROJECT=my-project ZONE=us-west4-a CLUSTER=tpu
#
# What this script does:
#   1. Authenticates gcloud (user account)
#   2. Configures Docker credential helpers for Artifact Registry and GCR
#   3. Installs kubectl and gke-gcloud-auth-plugin
#   4. Fetches GKE cluster credentials
#   5. Verifies everything works
set -euo pipefail

# --- Parse arguments ---
PROJECT="${PROJECT:-}"
ZONE="${ZONE:-us-west4-a}"
CLUSTER="${CLUSTER:-tpu}"

for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [ -z "$PROJECT" ]; then
    echo "Error: PROJECT is required."
    echo "Usage: bash gke/setup/setup_xpk.sh PROJECT=my-project [ZONE=us-west4-a] [CLUSTER=tpu]"
    exit 1
fi

REGION="${ZONE%-*}"  # us-west4-a -> us-west4

echo "==> Setting up xpk for project=$PROJECT zone=$ZONE cluster=$CLUSTER"

# --- 1. Authenticate gcloud ---
echo ""
echo "==> Step 1: Authenticate gcloud"
if gcloud auth list --filter="status:ACTIVE" --format="value(account)" | grep -q "@"; then
    ACCOUNT=$(gcloud auth list --filter="status:ACTIVE" --format="value(account)")
    echo "Already authenticated as: $ACCOUNT"
else
    echo "Running gcloud auth login..."
    gcloud auth login
fi

# --- 2. Configure Docker credential helpers ---
echo ""
echo "==> Step 2: Configure Docker credentials for Artifact Registry and GCR"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
gcloud auth configure-docker gcr.io --quiet
echo "Docker credential helpers configured."

# --- 3. Install kubectl and gke-gcloud-auth-plugin ---
echo ""
echo "==> Step 3: Install kubectl and gke-gcloud-auth-plugin"

if command -v kubectl >/dev/null 2>&1; then
    echo "kubectl already installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client 2>&1 | head -1)"
else
    echo "Installing kubectl..."
    sudo apt-get update -qq && sudo apt-get install -y -qq kubectl
fi

if command -v gke-gcloud-auth-plugin >/dev/null 2>&1; then
    echo "gke-gcloud-auth-plugin already installed."
else
    echo "Installing gke-gcloud-auth-plugin..."
    sudo apt-get update -qq && sudo apt-get install -y -qq google-cloud-cli-gke-gcloud-auth-plugin
fi

# --- 4. Get GKE cluster credentials ---
echo ""
echo "==> Step 4: Fetch GKE cluster credentials"
gcloud container clusters get-credentials "$CLUSTER" \
    --location="$REGION" \
    --project="$PROJECT"

# --- 5. Verify ---
echo ""
echo "==> Step 5: Verify setup"

echo -n "  gcloud project: "
gcloud projects describe "$PROJECT" --format="value(projectId)" 2>/dev/null && echo " OK" || echo " FAILED (check IAM permissions)"

echo -n "  kubectl: "
kubectl get pods --no-headers 2>/dev/null | wc -l | xargs -I{} echo "{} pods found - OK" || echo "FAILED"

echo -n "  xpk: "
if command -v xpk >/dev/null 2>&1; then
    echo "installed"
else
    echo "NOT FOUND - install with: pip install xpk"
fi

echo ""
echo "==> Setup complete. You can now run:"
echo "    python gke/submit.py --infra gke/infra/v5e.yaml gke/jobs/my_job.sh"
