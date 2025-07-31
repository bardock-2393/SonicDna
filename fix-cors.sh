#!/bin/bash

# Fix CORS and Configuration Issues for SoniqueDNA
# Run this script on your EC2 server

echo "🔧 Fixing CORS and Configuration Issues..."

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
BACKEND_DIR="/var/www/soniquedna/backend"

print_status "Fixing frontend environment configuration..."

# Create correct frontend environment file
cat > $FRONTEND_DIR/.env << 'EOF'
# Production Environment Variables for SoniqueDNA
VITE_SPOTIFY_CLIENT_ID=5b5e4ceb834347e6a6c3b998cfaf0088
VITE_SPOTIFY_REDIRECT_URI=https://soniquedna.deepsantoshwar.xyz/callback
VITE_BACKEND_URL=https://soniquedna.deepsantoshwar.xyz/api
VITE_APP_ENV=production
EOF

print_status "Frontend environment file updated"

# Rebuild frontend with correct configuration
print_status "Rebuilding frontend with correct configuration..."
cd $FRONTEND_DIR
npm run build

print_status "Fixing backend CORS configuration..."

# Update backend CORS settings
cat > $BACKEND_DIR/cors_fix.py << 'EOF'
import os

# Read the current app.py
with open('app.py', 'r') as f:
    content = f.read()

# Update CORS origins
new_cors_origins = """CORS(app, origins=[
    'http://localhost:5173', 
    'http://127.0.0.1:5173', 
    'http://localhost:3000', 
    'http://127.0.0.1:3000', 
    'http://localhost:8080', 
    'http://127.0.0.1:8080',
    'https://soniquedna.deepsantoshwar.xyz',
    'http://soniquedna.deepsantoshwar.xyz',
    'https://www.soniquedna.deepsantoshwar.xyz',
    'http://www.soniquedna.deepsantoshwar.xyz'
], supports_credentials=True)"""

# Replace the CORS line
import re
pattern = r'CORS\(app, origins=\[.*?\], supports_credentials=True\)'
content = re.sub(pattern, new_cors_origins, content, flags=re.DOTALL)

# Write back to app.py
with open('app.py', 'w') as f:
    f.write(content)

print("CORS configuration updated successfully!")
EOF

# Run the CORS fix
cd $BACKEND_DIR
python3 cors_fix.py

# Clean up
rm cors_fix.py

print_status "Restarting services..."

# Restart backend service
sudo systemctl restart soniquedna-backend

# Restart Caddy
sudo systemctl restart caddy

print_status "Checking service status..."

# Check if services are running
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

print_status "Testing application..."

# Wait a moment for services to start
sleep 5

# Test the application
if curl -s https://soniquedna.deepsantoshwar.xyz > /dev/null; then
    print_status "✅ Frontend is accessible"
else
    print_warning "⚠️  Frontend not accessible yet"
fi

if curl -s https://soniquedna.deepsantoshwar.xyz/api/health > /dev/null; then
    print_status "✅ Backend API is accessible"
else
    print_warning "⚠️  Backend API not accessible yet"
fi

print_status "CORS and configuration fixes completed!"
print_warning "If you still see CORS errors:"
print_warning "1. Clear your browser cache"
print_warning "2. Hard refresh the page (Ctrl+F5)"
print_warning "3. Check browser developer tools for any remaining errors" 