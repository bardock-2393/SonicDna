#!/bin/bash

# Fix Caddy Routing for SoniqueDNA API
# This script ensures API requests are properly routed to the backend

echo "🔧 Fixing Caddy Routing for API Requests..."

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

print_status "Creating corrected Caddy configuration..."

# Create a more explicit Caddy configuration
sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
soniquedna.deepsantoshwar.xyz {
    # API routes - proxy to backend (MUST be first)
    handle /api/* {
        reverse_proxy localhost:8080 {
            header_up Host {host}
            header_up X-Real-IP {remote}
            header_up X-Forwarded-For {remote}
            header_up X-Forwarded-Proto {scheme}
        }
        uri strip_prefix /api
    }

    # Spotify callback route
    handle /callback {
        reverse_proxy localhost:8080 {
            header_up Host {host}
            header_up X-Real-IP {remote}
            header_up X-Forwarded-For {remote}
            header_up X-Forwarded-Proto {scheme}
        }
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

# Test API endpoints with verbose output
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

print_status "Testing direct backend (should work)..."
DIRECT_RESPONSE=$(curl -s "http://localhost:8080/health")
echo "Direct backend response: $DIRECT_RESPONSE"

print_status "Caddy routing fix completed!"
print_warning "If API endpoints still don't work:"
print_warning "1. Check if backend is running: sudo systemctl status soniquedna-backend"
print_warning "2. Check Caddy logs: sudo journalctl -u caddy -f"
print_warning "3. Clear browser cache and hard refresh" 