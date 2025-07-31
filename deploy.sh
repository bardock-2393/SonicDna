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

# Check if required commands exist
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed. Please run server-setup.sh first."
        exit 1
    fi
}

print_status "Checking required commands..."
check_command python3
check_command pip
check_command node
check_command npm
check_command caddy

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
    # Create directories if they don't exist
    mkdir -p $BACKEND_DIR
    mkdir -p $FRONTEND_DIR
    
    # Copy files with proper error handling
    if cp -r backend/* $BACKEND_DIR/ 2>/dev/null; then
        print_status "Backend files copied successfully"
    else
        print_error "Failed to copy backend files"
        exit 1
    fi
    
    if cp -r frontend/* $FRONTEND_DIR/ 2>/dev/null; then
        print_status "Frontend files copied successfully"
    else
        print_error "Failed to copy frontend files"
        exit 1
    fi
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
if pip install -r requirements.txt; then
    print_status "Backend dependencies installed successfully"
else
    print_error "Failed to install backend dependencies"
    exit 1
fi

# Set up environment file
if [ ! -f ".env" ]; then
    print_status "Creating backend environment file..."
    if [ -f "env.production" ]; then
        cp env.production .env
        print_warning "Please edit $BACKEND_DIR/.env with your actual API keys"
    else
        print_warning "env.production not found, creating basic .env file"
        cat > .env << 'EOF'
# Production Environment Variables for SoniqueDNA Backend

# Spotify API Configuration
SPOTIFY_CLIENT_ID=5b5e4ceb834347e6a6c3b998cfaf0088
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIFY_REDIRECT_URI=https://soniquedna.deepsantoshwar.xyz/callback

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Qloo API Configuration
QLOO_API_KEY=your_qloo_api_key_here

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
FLASK_SECRET_KEY=your_secret_key_here

# Rate Limiting Configuration
SPOTIFY_RATE_LIMIT_DELAY=0.03
QLOO_RATE_LIMIT_DELAY=0.05
GEMINI_RATE_LIMIT_DELAY=0.1

# Database Configuration (if using)
DATABASE_URL=sqlite:///music_recommendations.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/var/log/soniquedna/backend.log
EOF
        print_warning "Please edit $BACKEND_DIR/.env with your actual API keys"
    fi
fi

# Frontend Setup
print_status "Setting up frontend..."

cd $FRONTEND_DIR

# Install dependencies
print_status "Installing frontend dependencies..."
if npm install; then
    print_status "Frontend dependencies installed successfully"
else
    print_error "Failed to install frontend dependencies"
    exit 1
fi

# Set up environment file
if [ ! -f ".env" ]; then
    print_status "Creating frontend environment file..."
    if [ -f "env.production" ]; then
        cp env.production .env
    else
        print_warning "env.production not found, creating basic .env file"
        cat > .env << 'EOF'
# Production Environment Variables for SoniqueDNA
VITE_SPOTIFY_CLIENT_ID=5b5e4ceb834347e6a6c3b998cfaf0088
VITE_SPOTIFY_REDIRECT_URI=https://soniquedna.deepsantoshwar.xyz/callback
VITE_BACKEND_URL=https://soniquedna.deepsantoshwar.xyz/api
VITE_APP_ENV=production
EOF
    fi
fi

# Build frontend
print_status "Building frontend for production..."
if npm run build; then
    print_status "Frontend built successfully"
else
    print_error "Frontend build failed"
    exit 1
fi

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
Environment=FLASK_ENV=production
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

# Stop existing services if running
print_status "Stopping existing services..."
sudo systemctl stop soniquedna-backend 2>/dev/null || true
sudo systemctl stop caddy 2>/dev/null || true

# Start services
print_status "Starting backend service..."
if sudo systemctl start soniquedna-backend; then
    print_status "Backend service started successfully"
else
    print_error "Failed to start backend service"
    sudo systemctl status soniquedna-backend
    exit 1
fi

print_status "Starting Caddy service..."
if sudo systemctl start caddy; then
    print_status "Caddy service started successfully"
else
    print_error "Failed to start Caddy service"
    sudo systemctl status caddy
    exit 1
fi

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
sudo ufw allow 80 2>/dev/null || true
sudo ufw allow 443 2>/dev/null || true
sudo ufw allow 22 2>/dev/null || true
sudo ufw --force enable 2>/dev/null || true

print_status "Deployment completed!"
print_status "Your application should be available at: https://soniquedna.deepsantoshwar.xyz"

# Test the application
print_status "Testing application..."
sleep 5

# Test backend health
if curl -s https://soniquedna.deepsantoshwar.xyz/api/health > /dev/null; then
    print_status "✅ Backend health check passed"
else
    print_warning "⚠️  Backend health check failed - this might be normal if DNS is not yet propagated"
fi

# Test frontend
if curl -s https://soniquedna.deepsantoshwar.xyz > /dev/null; then
    print_status "✅ Frontend is accessible"
else
    print_warning "⚠️  Frontend check failed - this might be normal if DNS is not yet propagated"
fi

print_warning "Don't forget to:"
print_warning "1. Update your Spotify app settings with the new callback URL"
print_warning "2. Edit $BACKEND_DIR/.env with your actual API keys"
print_warning "3. Check the logs if there are any issues:"
print_warning "   - Backend logs: sudo journalctl -u soniquedna-backend -f"
print_warning "   - Caddy logs: sudo journalctl -u caddy -f"
print_warning "4. Wait for DNS propagation (can take up to 24 hours)" 