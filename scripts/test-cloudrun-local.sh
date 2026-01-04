#!/bin/bash
#===============================================================================
# VitalFlow-Radar: Local Testing Script for Cloud Run Setup
#===============================================================================
#
# This script builds and runs the Cloud Run Docker image locally for testing.
#
# Usage:
#   ./scripts/test-cloudrun-local.sh
#
# Then open: http://localhost:8080
#===============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     VitalFlow-Radar - Local Cloud Run Testing              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Build the Docker image
echo -e "${YELLOW}[1/2] Building Docker image...${NC}"
docker build -t vitalflow-radar:local -f Dockerfile.cloudrun .

echo -e "${GREEN}✓ Image built successfully${NC}"
echo ""

# Run the container
echo -e "${YELLOW}[2/2] Starting container...${NC}"
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  VitalFlow-Radar is starting on http://localhost:8080      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

docker run --rm -it \
    -p 8080:8080 \
    -e VITALFLOW_ENV=development \
    -e VITALFLOW_SECRET_KEY=dev-secret-key-change-in-production \
    --name vitalflow-local \
    vitalflow-radar:local
