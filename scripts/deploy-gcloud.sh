#!/bin/bash
#===============================================================================
# VitalFlow-Radar: Google Cloud Run Deployment Script
#===============================================================================
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Authenticate: gcloud auth login
#   3. Set project: gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   ./scripts/deploy-gcloud.sh [PROJECT_ID] [REGION]
#
# Example:
#   ./scripts/deploy-gcloud.sh my-vitalflow-project us-central1
#===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
SERVICE_NAME="vitalflow-radar"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          VitalFlow-Radar - Google Cloud Deployment         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Validate prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No project ID specified${NC}"
    echo "Usage: $0 <PROJECT_ID> [REGION]"
    echo "Or set default: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✓ gcloud CLI installed${NC}"
echo -e "${GREEN}✓ Project: ${PROJECT_ID}${NC}"
echo -e "${GREEN}✓ Region: ${REGION}${NC}"
echo ""

# Set project
echo -e "${YELLOW}[2/6] Setting up project...${NC}"
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo -e "${YELLOW}[3/6] Enabling required APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    --quiet

echo -e "${GREEN}✓ APIs enabled${NC}"
echo ""

# Build the image using Cloud Build
echo -e "${YELLOW}[4/6] Building Docker image with Cloud Build...${NC}"

# Generate a short hash for tagging
SHORT_SHA=$(date +%Y%m%d%H%M%S)

gcloud builds submit \
    --config cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_SHORT_SHA=${SHORT_SHA}" \
    --quiet

echo -e "${GREEN}✓ Image built and pushed${NC}"
echo ""

# Deploy to Cloud Run
echo -e "${YELLOW}[5/6] Deploying to Cloud Run...${NC}"

# Build environment variables string from .env file if it exists
# Default to simulation mode for cloud (no radar hardware available)
ENV_VARS="VITALFLOW_ENV=production,VITALFLOW_MODE=simulation"
if [ -f ".env" ]; then
    # Read key environment variables from .env
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        # Only include specific prefixes
        if [[ "$key" =~ ^(CONFLUENT_|GOOGLE_|VERTEX_|VITALFLOW_) ]]; then
            # Remove quotes from value if present
            value="${value%\"}"
            value="${value#\"}"
            ENV_VARS="${ENV_VARS},${key}=${value}"
        fi
    done < .env
    echo -e "${GREEN}✓ Loaded environment variables from .env${NC}"
fi

gcloud run deploy "$SERVICE_NAME" \
    --image "${IMAGE_NAME}:latest" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "$ENV_VARS" \
    --quiet

echo -e "${GREEN}✓ Deployed to Cloud Run${NC}"
echo ""

# Get the service URL
echo -e "${YELLOW}[6/6] Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format 'value(status.url)')

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Deployment Successful!                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Service URL:${NC} ${SERVICE_URL}"
echo ""
echo -e "${YELLOW}Quick links:${NC}"
echo -e "  Dashboard:  ${SERVICE_URL}"
echo -e "  API Health: ${SERVICE_URL}/api/health"
echo -e "  API Docs:   ${SERVICE_URL}/docs"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo -e "  View logs:     gcloud run logs read --service ${SERVICE_NAME} --region ${REGION}"
echo -e "  Describe:      gcloud run services describe ${SERVICE_NAME} --region ${REGION}"
echo -e "  Delete:        gcloud run services delete ${SERVICE_NAME} --region ${REGION}"
echo ""
