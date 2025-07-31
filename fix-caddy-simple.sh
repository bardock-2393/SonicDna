#!/bin/bash

# Simple Caddy Fix for SoniqueDNA
# This script creates a simple Caddy configuration that works

echo "🔧 Creating Simple Caddy Configuration..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Set paths
FRONTEND_DIR="/var/www/soniquedna/frontend"

print_status "Creating simple Caddy configuration..."

# Create a simple Caddy configuration that doesn't strip /api
sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
soniquedna.deepsantoshwar.xyz {
    # API routes - proxy to backend without stripping /api
    handle /api/* {
        reverse_proxy localhost:8080
    }

    # Spotify callback route
    handle /callback {
        reverse_proxy localhost:8080
    }

    # Serve frontend static files (for all other routes)
    root * $FRONTEND_DIR/dist
    try_files {path} /index.html
    file_server

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

print_status "Validating Caddy configuration..."
if sudo caddy validate --config /etc/caddy/Caddyfile; then
    print_status "✅ Caddy configuration is valid"
else
    print_error "❌ Caddy configuration is invalid"
    exit 1
fi

print_status "Restarting Caddy..."
sudo systemctl restart caddy

print_status "Waiting for Caddy to start..."
sleep 5

print_status "Checking Caddy status..."
if sudo systemctl is-active --quiet caddy; then
    print_status "✅ Caddy service is running"
else
    print_error "❌ Caddy service failed to start"
    sudo systemctl status caddy
    exit 1
fi

print_status "Testing API endpoints..."

# Test API endpoints
print_status "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "https://soniquedna.deepsantoshwar.xyz/api/health")
echo "Health response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_status "✅ Health endpoint working"
else
    print_error "❌ Health endpoint not working"
    print_error "Response: $HEALTH_RESPONSE"
fi

print_status "Testing Spotify auth endpoint..."
AUTH_RESPONSE=$(curl -s "https://soniquedna.deepsantoshwar.xyz/api/spotify-auth-url?redirect_uri=https://soniquedna.deepsantoshwar.xyz/callback&force_reauth=true")
echo "Auth response: $AUTH_RESPONSE"

if echo "$AUTH_RESPONSE" | grep -q "auth_url"; then
    print_status "✅ Spotify auth endpoint working"
else
    print_error "❌ Spotify auth endpoint not working"
    print_error "Response: $AUTH_RESPONSE"
fi

print_status "Simple Caddy fix completed!"
print_warning "Please:"
print_warning "1. Clear your browser cache (Ctrl+Shift+Delete)"
print_warning "2. Hard refresh the page (Ctrl+F5)"
print_warning "3. Test the Spotify connection again" 