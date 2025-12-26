#!/bin/bash
#===============================================================================
# VitalFlow-Radar: Raspberry Pi Setup Script (FastAPI + React)
#===============================================================================
# Sets up Raspberry Pi for VitalFlow-Radar production deployment.
#
# Usage:
#   chmod +x setup_raspberry_pi.sh
#   sudo ./setup_raspberry_pi.sh
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VITALFLOW_USER="vitalflow"
VITALFLOW_HOME="/home/${VITALFLOW_USER}"
VITALFLOW_DIR="${VITALFLOW_HOME}/VitalFlow-Radar"
VITALFLOW_DATA="/var/lib/vitalflow"
VITALFLOW_LOG="/var/log/vitalflow"
API_PORT=8000
FRONTEND_PORT=3000
NODE_VERSION="20"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║         VitalFlow-Radar: Raspberry Pi Setup                       ║"
echo "║              FastAPI + React Production                           ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: Run as root (sudo)${NC}"
   exit 1
fi

# Detect hardware
echo -e "${BLUE}[1/12] Detecting hardware...${NC}"
if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(cat /proc/device-tree/model)
    echo -e "${GREEN}✓ Detected: ${PI_MODEL}${NC}"
fi

# Update system
echo -e "${BLUE}[2/12] Updating system...${NC}"
apt-get update -qq
apt-get upgrade -y -qq

# Install system dependencies
echo -e "${BLUE}[3/12] Installing dependencies...${NC}"
apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    nginx \
    sqlite3 \
    libsqlite3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    curl \
    ufw \
    fail2ban \
    certbot \
    python3-certbot-nginx

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Install Node.js
echo -e "${BLUE}[4/12] Installing Node.js ${NODE_VERSION}...${NC}"
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
    apt-get install -y nodejs
fi
echo -e "${GREEN}✓ Node.js $(node --version) installed${NC}"

# Configure serial ports
echo -e "${BLUE}[5/12] Configuring serial ports...${NC}"
cat > /etc/udev/rules.d/99-ti-radar.rules << 'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="0451", MODE="0666", GROUP="dialout"
KERNEL=="ttyUSB*", MODE="0666", GROUP="dialout"
KERNEL=="ttyACM*", MODE="0666", GROUP="dialout"
EOF
udevadm control --reload-rules
echo -e "${GREEN}✓ Serial ports configured${NC}"

# Create user
echo -e "${BLUE}[6/12] Creating user...${NC}"
if ! id "${VITALFLOW_USER}" &>/dev/null; then
    useradd -m -s /bin/bash "${VITALFLOW_USER}"
    usermod -aG dialout,gpio "${VITALFLOW_USER}"
fi
echo -e "${GREEN}✓ User configured${NC}"

# Create directories
echo -e "${BLUE}[7/12] Creating directories...${NC}"
mkdir -p "${VITALFLOW_DATA}" "${VITALFLOW_LOG}" "${VITALFLOW_DIR}"
chown -R ${VITALFLOW_USER}:${VITALFLOW_USER} "${VITALFLOW_DATA}" "${VITALFLOW_LOG}" "${VITALFLOW_DIR}"
echo -e "${GREEN}✓ Directories created${NC}"

# Copy application files
echo -e "${BLUE}[8/12] Setting up application...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "${SCRIPT_DIR}/../backend" ]; then
    cp -r "${SCRIPT_DIR}/../"* "${VITALFLOW_DIR}/"
fi
chown -R ${VITALFLOW_USER}:${VITALFLOW_USER} "${VITALFLOW_DIR}"
echo -e "${GREEN}✓ Application files copied${NC}"

# Setup Python backend
echo -e "${BLUE}[9/12] Setting up Python backend...${NC}"
sudo -u ${VITALFLOW_USER} python3.11 -m venv "${VITALFLOW_DIR}/backend/venv"
sudo -u ${VITALFLOW_USER} "${VITALFLOW_DIR}/backend/venv/bin/pip" install --upgrade pip
sudo -u ${VITALFLOW_USER} "${VITALFLOW_DIR}/backend/venv/bin/pip" install -r "${VITALFLOW_DIR}/backend/requirements.txt"
echo -e "${GREEN}✓ Python backend configured${NC}"

# Build React frontend
echo -e "${BLUE}[10/12] Building React frontend...${NC}"
cd "${VITALFLOW_DIR}/frontend"
sudo -u ${VITALFLOW_USER} npm install
sudo -u ${VITALFLOW_USER} npm run build
echo -e "${GREEN}✓ Frontend built${NC}"

# Create systemd services
echo -e "${BLUE}[11/12] Creating services...${NC}"

# Backend service
cat > /etc/systemd/system/vitalflow-api.service << EOF
[Unit]
Description=VitalFlow API Server
After=network.target

[Service]
Type=simple
User=${VITALFLOW_USER}
WorkingDirectory=${VITALFLOW_DIR}/backend
Environment=PATH=${VITALFLOW_DIR}/backend/venv/bin:/usr/bin
Environment=VITALFLOW_DB=${VITALFLOW_DATA}/vitalflow.db
ExecStart=${VITALFLOW_DIR}/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port ${API_PORT}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vitalflow-api.service
echo -e "${GREEN}✓ Services created${NC}"

# Configure Nginx
echo -e "${BLUE}[12/12] Configuring Nginx...${NC}"
cat > /etc/nginx/sites-available/vitalflow << EOF
server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # React frontend
    location / {
        root ${VITALFLOW_DIR}/frontend/dist;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }
}
EOF

ln -sf /etc/nginx/sites-available/vitalflow /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
echo -e "${GREEN}✓ Nginx configured${NC}"

# Configure firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Create environment file
cat > "${VITALFLOW_DIR}/.env" << EOF
VITALFLOW_SECRET_KEY=$(openssl rand -hex 32)
VITALFLOW_ADMIN_PASSWORD=admin123
VITALFLOW_DB=${VITALFLOW_DATA}/vitalflow.db
EOF
chown ${VITALFLOW_USER}:${VITALFLOW_USER} "${VITALFLOW_DIR}/.env"
chmod 600 "${VITALFLOW_DIR}/.env"

# Start services
systemctl start vitalflow-api.service
sleep 3

# Get IP
IP_ADDR=$(hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    Installation Complete!                         ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "Dashboard: ${BLUE}http://${IP_ADDR}${NC}"
echo -e "API Docs:  ${BLUE}http://${IP_ADDR}/api/docs${NC}"
echo ""
echo -e "Login: ${YELLOW}admin / admin123${NC}"
echo ""
echo -e "Commands:"
echo -e "  Status:  ${BLUE}sudo systemctl status vitalflow-api${NC}"
echo -e "  Logs:    ${BLUE}sudo journalctl -u vitalflow-api -f${NC}"
echo -e "  Restart: ${BLUE}sudo systemctl restart vitalflow-api${NC}"
echo ""
