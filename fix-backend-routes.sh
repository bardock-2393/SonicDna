#!/bin/bash

# Fix Backend Routes for SoniqueDNA
# This script adds /api prefix to backend routes to match Caddy configuration

echo "🔧 Fixing Backend Routes to Match API Structure..."

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
BACKEND_DIR="/var/www/soniquedna/backend"

print_status "Creating backup of backend code..."
cd $BACKEND_DIR
cp app.py app.py.backup.$(date +%Y%m%d_%H%M%S)

print_status "Adding /api prefix to backend routes..."

# Create a Python script to fix the routes
cat > fix_routes.py << 'EOF'
import re

# Read the current app.py
with open('app.py', 'r') as f:
    content = f.read()

# Define routes that need /api prefix
routes_to_fix = [
    r"@app\.route\('/spotify-auth-url'",
    r"@app\.route\('/exchange-token'",
    r"@app\.route\('/refresh-token'",
    r"@app\.route\('/check-token'",
    r"@app\.route\('/spotify-profile'",
    r"@app\.route\('/logout'",
    r"@app\.route\('/spotify-session-clear'",
    r"@app\.route\('/musicrecommandation'",
    r"@app\.route\('/musicrecommendation'",
    r"@app\.route\('/music-recommendation'",
    r"@app\.route\('/crossdomain-recommendations'",
    r"@app\.route\('/crossdomain-progress'",
    r"@app\.route\('/create-playlist'",
    r"@app\.route\('/test-database'",
    r"@app\.route\('/user-analytics'",
    r"@app\.route\('/user-history'",
    r"@app\.route\('/clear-cache'",
    r"@app\.route\('/'",
    r"@app\.route\('/health'"
]

# Fix each route
for route_pattern in routes_to_fix:
    # Replace the route pattern
    if '/spotify-auth-url' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/spotify-auth-url'", content)
    elif '/exchange-token' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/exchange-token'", content)
    elif '/refresh-token' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/refresh-token'", content)
    elif '/check-token' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/check-token'", content)
    elif '/spotify-profile' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/spotify-profile'", content)
    elif '/logout' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/logout'", content)
    elif '/spotify-session-clear' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/spotify-session-clear'", content)
    elif '/musicrecommandation' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/musicrecommandation'", content)
    elif '/musicrecommendation' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/musicrecommendation'", content)
    elif '/music-recommendation' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/music-recommendation'", content)
    elif '/crossdomain-recommendations' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/crossdomain-recommendations'", content)
    elif '/crossdomain-progress' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/crossdomain-progress'", content)
    elif '/create-playlist' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/create-playlist'", content)
    elif '/test-database' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/test-database'", content)
    elif '/user-analytics' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/user-analytics'", content)
    elif '/user-history' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/user-history'", content)
    elif '/clear-cache' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/clear-cache'", content)
    elif '/health' in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/health'", content)
    elif "/'" in route_pattern:
        content = re.sub(route_pattern, "@app.route('/api/'", content)

# Write the modified content back
with open('app.py', 'w') as f:
    f.write(content)

print("Backend routes updated successfully!")
EOF

# Run the route fix script
python3 fix_routes.py

# Clean up
rm fix_routes.py

print_status "Restarting backend service..."
sudo systemctl restart soniquedna-backend

print_status "Waiting for backend to start..."
sleep 5

print_status "Checking backend status..."
if sudo systemctl is-active --quiet soniquedna-backend; then
    print_status "✅ Backend service is running"
else
    print_error "❌ Backend service failed to start"
    sudo systemctl status soniquedna-backend
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

print_status "Backend route fix completed!"
print_warning "Please:"
print_warning "1. Clear your browser cache (Ctrl+Shift+Delete)"
print_warning "2. Hard refresh the page (Ctrl+F5)"
print_warning "3. Test the Spotify connection again" 