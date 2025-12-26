#!/bin/bash
#===============================================================================
# VitalFlow-Radar: Development Server Startup
# Runs FastAPI backend and React frontend in development mode
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="${PROJECT_DIR}/backend"
FRONTEND_DIR="${PROJECT_DIR}/frontend"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║         VitalFlow-Radar: Development Server                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check dependencies
check_deps() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 not found${NC}"
        exit 1
    fi
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js not found${NC}"
        exit 1
    fi
}

# Setup Python virtual environment
setup_backend() {
    echo -e "${BLUE}Setting up backend...${NC}"
    cd "$BACKEND_DIR"
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Backend ready${NC}"
}

# Setup frontend
setup_frontend() {
    echo -e "${BLUE}Setting up frontend...${NC}"
    cd "$FRONTEND_DIR"
    
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    echo -e "${GREEN}✓ Frontend ready${NC}"
}

# Start backend
start_backend() {
    echo -e "${BLUE}Starting FastAPI backend on :8000...${NC}"
    cd "$BACKEND_DIR"
    source venv/bin/activate
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    sleep 2
    
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${GREEN}✓ Backend running (PID: $BACKEND_PID)${NC}"
    else
        echo -e "${RED}✗ Backend failed to start${NC}"
        exit 1
    fi
}

# Start frontend
start_frontend() {
    echo -e "${BLUE}Starting React frontend on :5173...${NC}"
    cd "$FRONTEND_DIR"
    npm run dev &
    FRONTEND_PID=$!
    sleep 3
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "${GREEN}✓ Frontend running (PID: $FRONTEND_PID)${NC}"
    else
        echo -e "${RED}✗ Frontend failed to start${NC}"
        exit 1
    fi
}

# Cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✓ Stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main
check_deps
setup_backend
setup_frontend
start_backend
start_frontend

# Get local IP
IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    Development Server Ready                        ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "Frontend:  ${BLUE}http://localhost:5173${NC}"
echo -e "API:       ${BLUE}http://localhost:8000${NC}"
echo -e "API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "Network:   ${BLUE}http://${IP_ADDR}:5173${NC}"
echo ""
echo -e "Login:     ${YELLOW}admin / admin123${NC}"
echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop"
echo ""

# Wait
wait
