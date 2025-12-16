#!/bin/bash
# Deployment script for OR-Tools Recommender to Google Cloud Run
# Usage: ./deploy.sh

set -e  # Exit on error

# Add gcloud to PATH if it exists in home directory
if [ -d "$HOME/google-cloud-sdk/bin" ]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# Configuration
PROJECT_ID="resonant-tube-437220-t3"  # AllMart - Maps and Tookan
SERVICE_NAME="or-tools-recommender"
REGION="us-central1"
REPOSITORY="cloud-run-source-deploy"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}"

# Dashboard URL for fleet optimization callbacks
# Set this to your dashboard's public URL (the OR-Tools service calls this to fetch agent/task data)
# Leave empty to require dashboard_url in each request
DASHBOARD_URL="https://allmart-dashboard-95621826490.us-central1.run.app"  # Example: "https://your-dashboard.vercel.app" or "https://your-dashboard.com"

echo "=================================="
echo "OR-Tools Recommender Deployment"
echo "=================================="
echo ""
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"
if [ -n "$DASHBOARD_URL" ]; then
    echo "Dashboard URL: ${DASHBOARD_URL}"
else
    echo "Dashboard URL: (will use request data or localhost default)"
fi
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed"
    echo ""
    echo "Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    echo ""
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not logged in to gcloud"
    echo "Running: gcloud auth login"
    gcloud auth login
fi

# Set project
echo "Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Build Docker image for Cloud Run (linux/amd64)
echo ""
echo "📦 Building Docker image..."
cd OR_Tools_prototype
docker build --platform linux/amd64 -t ${IMAGE_NAME}:latest .

# Configure Docker for Artifact Registry
echo ""
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Push to Artifact Registry
echo ""
echo "⬆️  Pushing image to Artifact Registry..."
docker push ${IMAGE_NAME}:latest

# Build environment variables string
ENV_VARS=""
if [ -n "$DASHBOARD_URL" ]; then
    ENV_VARS="--set-env-vars DASHBOARD_URL=${DASHBOARD_URL}"
fi

# Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --timeout 300 \
  ${ENV_VARS}

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Service URL:"
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)'
echo ""
echo "Test the deployment:"
echo "  curl \$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)')/health"
echo ""
echo "WebSocket endpoint:"
echo "  wss://\$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)' | sed 's/https:\/\///')/socket.io/"
echo ""
