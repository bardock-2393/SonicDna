#!/bin/bash

# Fix Hardcoded URLs Script for SoniqueDNA
# This script replaces all hardcoded localhost URLs with production URLs

echo "🔧 Fixing hardcoded URLs in frontend code..."

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
PRODUCTION_URL="https://soniquedna.deepsantoshwar.xyz/api"
PRODUCTION_CALLBACK="https://soniquedna.deepsantoshwar.xyz/callback"

print_status "Replacing hardcoded URLs in frontend code..."

cd $FRONTEND_DIR

# Create a backup
print_status "Creating backup of frontend code..."
cp -r src src.backup.$(date +%Y%m%d_%H%M%S)

# Replace localhost:5500 with production URL
print_status "Replacing localhost:5500 URLs..."
find src -name "*.tsx" -o -name "*.ts" -o -name "*.js" -o -name "*.jsx" | xargs sed -i 's|http://localhost:5500|'$PRODUCTION_URL'|g'

# Replace 127.0.0.1:8080 with production callback
print_status "Replacing 127.0.0.1:8080 URLs..."
find src -name "*.tsx" -o -name "*.ts" -o -name "*.js" -o -name "*.jsx" | xargs sed -i 's|http://127.0.0.1:8080/callback|'$PRODUCTION_CALLBACK'|g'

# Replace any remaining localhost references
print_status "Replacing remaining localhost references..."
find src -name "*.tsx" -o -name "*.ts" -o -name "*.js" -o -name "*.jsx" | xargs sed -i 's|http://localhost|https://soniquedna.deepsantoshwar.xyz|g'

print_status "Rebuilding frontend with fixed URLs..."
npm run build

print_status "Restarting services..."
sudo systemctl restart soniquedna-backend
sudo systemctl restart caddy

print_status "Checking service status..."
if sudo systemctl is-active --quiet soniquedna-backend; then
    print_status "✅ Backend service is running"
else
    print_error "❌ Backend service failed to start"
    sudo systemctl status soniquedna-backend
fi

if sudo systemctl is-active --quiet caddy; then
    print_status "✅ Caddy service is running"
else
    print_error "❌ Caddy service failed to start"
    sudo systemctl status caddy
fi

print_status "Hardcoded URL fixes completed!"
print_warning "Please:"
print_warning "1. Clear your browser cache (Ctrl+Shift+Delete)"
print_warning "2. Hard refresh the page (Ctrl+F5)"
print_warning "3. Test the Spotify connection again" 