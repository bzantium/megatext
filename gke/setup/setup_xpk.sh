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

# --- 2. Configure Docker credentials and Artifact Registry ---
echo ""
echo "==> Step 2: Configure Docker credentials for Artifact Registry and GCR"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
gcloud auth configure-docker gcr.io --quiet
echo "Docker credential helpers configured."

echo ""
echo "==> Step 2b: Ensure Artifact Registry repository exists"
if gcloud artifacts repositories describe megatext --location="$REGION" --project="$PROJECT" >/dev/null 2>&1; then
    echo "Artifact Registry repo 'megatext' already exists."
else
    echo "Creating Artifact Registry repo 'megatext'..."
    gcloud artifacts repositories create megatext \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT" \
        --description="Megatext Docker images"
    echo "Created."
fi

# --- 3. Install kubectl and gke-gcloud-auth-plugin ---
echo ""
echo "==> Step 3: Install kubectl and gke-gcloud-auth-plugin"

# Ensure apt keyrings directory exists
sudo mkdir -p /etc/apt/keyrings

# Add Kubernetes apt repo if kubectl is not installable
if command -v kubectl >/dev/null 2>&1; then
    echo "kubectl already installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client 2>&1 | head -1)"
else
    if ! apt-cache show kubectl >/dev/null 2>&1; then
        echo "Adding Kubernetes apt repository..."
        sudo apt-get install -y -qq apt-transport-https ca-certificates curl
        curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.32/deb/Release.key | \
            sudo gpg --dearmor --yes -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
        echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.32/deb/ /' | \
            sudo tee /etc/apt/sources.list.d/kubernetes.list > /dev/null
        sudo apt-get update -qq
    fi
    echo "Installing kubectl..."
    sudo apt-get install -y -qq kubectl
fi

# Add Google Cloud SDK apt repo if gke-gcloud-auth-plugin is not installable
if command -v gke-gcloud-auth-plugin >/dev/null 2>&1; then
    echo "gke-gcloud-auth-plugin already installed."
else
    if ! apt-cache show google-cloud-cli-gke-gcloud-auth-plugin >/dev/null 2>&1; then
        echo "Adding Google Cloud SDK apt repository..."
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
            sudo gpg --dearmor --yes -o /etc/apt/keyrings/cloud.google.gpg
        echo 'deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | \
            sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list > /dev/null
        sudo apt-get update -qq
    fi
    echo "Installing gke-gcloud-auth-plugin..."
    sudo apt-get install -y -qq google-cloud-cli-gke-gcloud-auth-plugin
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
