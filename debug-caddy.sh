#!/bin/bash

# Debug Caddy Configuration for SoniqueDNA
# This script will help us understand what's happening with Caddy

echo "🔍 Debugging Caddy Configuration..."

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

print_status "Checking current Caddy configuration..."
sudo cat /etc/caddy/Caddyfile

print_status "Checking Caddy logs..."
sudo journalctl -u caddy -n 10

print_status "Testing different approaches..."

# Approach 1: Test with explicit port
print_status "Approach 1: Testing with explicit port..."
curl -v "http://localhost:8080/api/health"

# Approach 2: Test Caddy with different config
print_status "Approach 2: Creating minimal Caddy config..."
sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
soniquedna.deepsantoshwar.xyz {
    reverse_proxy localhost:8080
}
EOF

sudo systemctl restart caddy
sleep 3

print_status "Testing minimal config..."
curl -s "https://soniquedna.deepsantoshwar.xyz/health"

# Approach 3: Test with different route structure
print_status "Approach 3: Testing with different route structure..."
sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
soniquedna.deepsantoshwar.xyz {
    handle /api/* {
        reverse_proxy localhost:8080
    }
    handle /* {
        reverse_proxy localhost:8080
    }
}
EOF

sudo systemctl restart caddy
sleep 3

print_status "Testing different route structure..."
curl -s "https://soniquedna.deepsantoshwar.xyz/api/health"

# Approach 4: Check if there are multiple Caddy processes
print_status "Approach 4: Checking for multiple Caddy processes..."
ps aux | grep caddy

# Approach 5: Check if there are other web servers running
print_status "Approach 5: Checking for other web servers..."
sudo netstat -tlnp | grep :80
sudo netstat -tlnp | grep :443
sudo netstat -tlnp | grep :8080

print_status "Debug completed!"
print_warning "Check the output above to understand what's happening." 