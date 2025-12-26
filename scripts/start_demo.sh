#!/bin/bash
# VitalFlow-Radar: Quick Start Demo Script
# =========================================
# 
# This script starts the full VitalFlow system for hackathon judges to test.
# No Kafka or Google Cloud credentials required for basic demo mode.
#
# Usage:
#   ./scripts/start_demo.sh           # Start full system with demo traffic
#   ./scripts/start_demo.sh --no-traffic  # Start without traffic generator
#
# Requirements:
#   - Python 3.9+
#   - Node.js 16+
#   - npm or pnpm

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██╗   ██╗██╗████████╗ █████╗ ██╗     ███████╗██╗      ██████╗ ██╗   ║
║   ██║   ██║██║╚══██╔══╝██╔══██╗██║     ██╔════╝██║     ██╔═══██╗██║   ║
║   ██║   ██║██║   ██║   ███████║██║     █████╗  ██║     ██║   ██║██║   ║
║   ╚██╗ ██╔╝██║   ██║   ██╔══██║██║     ██╔══╝  ██║     ██║   ██║╚═╝   ║
║    ╚████╔╝ ██║   ██║   ██║  ██║███████╗██║     ███████╗╚██████╔╝██╗   ║
║     ╚═══╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝ ╚═════╝ ╚═╝   ║
║                                                                       ║
║              77GHz Contactless Vital Signs Monitoring                 ║
║                  Confluent Challenge Submission                       ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Parse arguments
WITH_TRAFFIC=true
for arg in "$@"; do
    case $arg in
        --no-traffic)
            WITH_TRAFFIC=false
            shift
            ;;
    esac
done

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 is not installed${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 is available${NC}"
        return 0
    fi
}

# Function to check if a port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 1
    else
        return 0
    fi
}

# Function to kill process on port
kill_port() {
    if ! check_port $1; then
        echo -e "${YELLOW}Port $1 is in use, attempting to free it...${NC}"
        fuser -k $1/tcp 2>/dev/null || true
        sleep 1
    fi
}

echo -e "${BLUE}Checking prerequisites...${NC}"
echo ""

# Check requirements
check_command python3 || { echo -e "${RED}Please install Python 3.9+${NC}"; exit 1; }
check_command node || { echo -e "${RED}Please install Node.js 16+${NC}"; exit 1; }
check_command npm || check_command pnpm || { echo -e "${RED}Please install npm or pnpm${NC}"; exit 1; }

echo ""
echo -e "${BLUE}Setting up environment...${NC}"
echo ""

cd "$PROJECT_ROOT"

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo -e "${YELLOW}Installing backend dependencies...${NC}"
pip install -q -r backend/requirements.txt

# Install frontend dependencies
echo -e "${YELLOW}Installing frontend dependencies...${NC}"
cd frontend
if command -v pnpm &> /dev/null; then
    pnpm install --silent 2>/dev/null || npm install --silent
else
    npm install --silent
fi
cd ..

# Ensure ports are available
echo ""
echo -e "${BLUE}Checking ports...${NC}"
kill_port 8000  # Backend
kill_port 5173  # Frontend (Vite)

# Create log directory
mkdir -p logs

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down VitalFlow...${NC}"
    
    # Kill background processes
    [ ! -z "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ ! -z "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    [ ! -z "$TRAFFIC_PID" ] && kill $TRAFFIC_PID 2>/dev/null
    
    # Kill by port as fallback
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k 5173/tcp 2>/dev/null || true
    
    echo -e "${GREEN}VitalFlow stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo ""
echo -e "${GREEN}Starting VitalFlow Demo...${NC}"
echo ""

# Start Backend (FastAPI)
echo -e "${CYAN}[1/3] Starting Backend API on http://localhost:8000${NC}"
cd "$PROJECT_ROOT/backend"
VITALFLOW_MODE=simulation uvicorn main:app --host 0.0.0.0 --port 8000 --log-level warning > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for backend to start
echo -n "     Waiting for backend"
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 0.5
done

# Verify backend is running
if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo -e " ${RED}✗ Failed to start backend${NC}"
    echo "Check logs/backend.log for details"
    exit 1
fi

# Start Frontend (Vite)
echo -e "${CYAN}[2/3] Starting Frontend on http://localhost:5173${NC}"
cd "$PROJECT_ROOT/frontend"
npm run dev -- --host 0.0.0.0 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for frontend to start
echo -n "     Waiting for frontend"
for i in {1..30}; do
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 0.5
done

# Optionally start traffic generator
if [ "$WITH_TRAFFIC" = true ]; then
    echo -e "${CYAN}[3/3] Starting Demo Traffic Generator${NC}"
    sleep 2  # Let services stabilize
    
    cd "$PROJECT_ROOT"
    python scripts/demo_traffic_http.py --continuous --duration 600 > logs/traffic.log 2>&1 &
    TRAFFIC_PID=$!
    echo -e "     Traffic generator ${GREEN}started${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                     VitalFlow Demo Ready!                          ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}📊 Dashboard:${NC}     http://localhost:5173"
echo -e "  ${CYAN}🔌 API Docs:${NC}      http://localhost:8000/docs"
echo -e "  ${CYAN}❤️  Health Check:${NC}  http://localhost:8000/api/health"
echo ""
echo -e "  ${YELLOW}Demo Credentials:${NC}"
echo -e "     Username: admin"
echo -e "     Password: admin123"
echo ""

if [ "$WITH_TRAFFIC" = true ]; then
    echo -e "  ${GREEN}✨ Demo traffic is flowing - watch the dashboard!${NC}"
    echo ""
fi

echo -e "  ${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"

# Keep running and show combined logs
echo ""
echo -e "${BLUE}Streaming logs (Ctrl+C to stop):${NC}"
echo ""

# Tail logs
tail -f logs/backend.log logs/frontend.log logs/traffic.log 2>/dev/null &
TAIL_PID=$!

# Wait for any background process to exit
wait $BACKEND_PID $FRONTEND_PID
