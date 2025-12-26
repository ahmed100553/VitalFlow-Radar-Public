#!/bin/bash
#===============================================================================
# VitalFlow-Radar: SSL Certificate Setup
# Configures Let's Encrypt certificates for HTTPS
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║         VitalFlow-Radar: SSL Certificate Setup                    ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: Run as root (sudo)${NC}"
   exit 1
fi

# Get domain
read -p "Enter your domain (e.g., vitalflow.example.com): " DOMAIN
read -p "Enter your email for Let's Encrypt: " EMAIL

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    echo -e "${RED}Error: Domain and email are required${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Obtaining SSL certificate for ${DOMAIN}...${NC}"

# Obtain certificate
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL"

# Update Nginx for HTTPS
cat > /etc/nginx/sites-available/vitalflow << EOF
server {
    listen 80;
    server_name ${DOMAIN};
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ${DOMAIN};

    ssl_certificate /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Gzip
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    # React frontend
    location / {
        root /home/vitalflow/VitalFlow-Radar/frontend/dist;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }
}
EOF

# Test and reload
nginx -t && systemctl reload nginx

# Setup auto-renewal
systemctl enable certbot.timer
systemctl start certbot.timer

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    SSL Setup Complete!                            ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "Dashboard: ${BLUE}https://${DOMAIN}${NC}"
echo ""
echo -e "Certificate auto-renewal is enabled."
echo ""
