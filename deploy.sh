#!/bin/bash

# SoniqueDNA Deployment Script
# Run this script on your EC2 server to deploy the application

echo "🚀 Starting SoniqueDNA Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root"
    exit 1
fi

# Set application directory
APP_DIR="/var/www/soniquedna"
BACKEND_DIR="$APP_DIR/backend"
FRONTEND_DIR="$APP_DIR/frontend"

print_status "Setting up application directory..."

# Create application directory if it doesn't exist
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Copy application files (assuming you're running this from the project root)
if [ -d "backend" ] && [ -d "frontend" ]; then
    print_status "Copying application files..."
    cp -r backend/* $BACKEND_DIR/
    cp -r frontend/* $FRONTEND_DIR/
else
    print_error "Backend and frontend directories not found. Please run this script from the project root."
    exit 1
fi

# Backend Setup
print_status "Setting up backend..."

cd $BACKEND_DIR

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
print_status "Installing backend dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Set up environment file
if [ ! -f ".env" ]; then
    print_status "Creating backend environment file..."
    cp env.production .env
    print_warning "Please edit $BACKEND_DIR/.env with your actual API keys"
fi

# Frontend Setup
print_status "Setting up frontend..."

cd $FRONTEND_DIR

# Install dependencies
print_status "Installing frontend dependencies..."
npm install

# Set up environment file
if [ ! -f ".env" ]; then
    print_status "Creating frontend environment file..."
    cp env.production .env
fi

# Build frontend
print_status "Building frontend for production..."
npm run build

# Create systemd service file
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/soniquedna-backend.service > /dev/null <<EOF
[Unit]
Description=SoniqueDNA Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BACKEND_DIR
Environment=PATH=$BACKEND_DIR/venv/bin
ExecStart=$BACKEND_DIR/venv/bin/python app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create Caddy configuration
print_status "Creating Caddy configuration..."
sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
soniquedna.deepsantoshwar.xyz {
    # Serve frontend static files
    root * $FRONTEND_DIR/dist
    try_files {path} /index.html
    file_server

    # API routes - proxy to backend
    handle /api/* {
        reverse_proxy localhost:8080
        uri strip_prefix /api
    }

    # Spotify callback route
    handle /callback {
        reverse_proxy localhost:8080
    }

    # Enable compression
    encode gzip

    # Security headers
    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        X-XSS-Protection "1; mode=block"
        Referrer-Policy "strict-origin-when-cross-origin"
    }
}
EOF

# Create log directory
sudo mkdir -p /var/log/soniquedna
sudo chown $USER:$USER /var/log/soniquedna

# Start services
print_status "Starting services..."

# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable soniquedna-backend
sudo systemctl enable caddy

# Start services
sudo systemctl start soniquedna-backend
sudo systemctl start caddy

# Check service status
print_status "Checking service status..."
if sudo systemctl is-active --quiet soniquedna-backend; then
    print_status "Backend service is running"
else
    print_error "Backend service failed to start"
    sudo systemctl status soniquedna-backend
fi

if sudo systemctl is-active --quiet caddy; then
    print_status "Caddy service is running"
else
    print_error "Caddy service failed to start"
    sudo systemctl status caddy
fi

# Configure firewall
print_status "Configuring firewall..."
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 22
sudo ufw --force enable

print_status "Deployment completed!"
print_status "Your application should be available at: https://soniquedna.deepsantoshwar.xyz"
print_warning "Don't forget to:"
print_warning "1. Update your Spotify app settings with the new callback URL"
print_warning "2. Edit $BACKEND_DIR/.env with your actual API keys"
print_warning "3. Check the logs if there are any issues:"
print_warning "   - Backend logs: sudo journalctl -u soniquedna-backend -f"
print_warning "   - Caddy logs: sudo journalctl -u caddy -f" 