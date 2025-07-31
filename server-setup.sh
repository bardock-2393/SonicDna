#!/bin/bash

# Server Setup Script for SoniqueDNA
# Run this script on your EC2 server to install all necessary software

echo "🔧 Setting up EC2 server for SoniqueDNA..."

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

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
print_status "Installing essential packages..."
sudo apt install -y curl wget git unzip software-properties-common apt-transport-https ca-certificates gnupg lsb-release

# Install Python and pip
print_status "Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install Node.js and npm
print_status "Installing Node.js and npm..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install Caddy
print_status "Installing Caddy web server..."
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy

# Configure firewall
print_status "Configuring firewall..."
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# Check if UFW is active
if sudo ufw status | grep -q "Status: active"; then
    print_status "Firewall is active and configured"
else
    print_warning "Firewall may not be active - check manually with: sudo ufw status"
fi

# Create application directory
print_status "Creating application directory..."
sudo mkdir -p /var/www/soniquedna
sudo chown $USER:$USER /var/www/soniquedna

# Create log directory
print_status "Creating log directory..."
sudo mkdir -p /var/log/soniquedna
sudo chown $USER:$USER /var/log/soniquedna

# Verify installations
print_status "Verifying installations..."

echo "Python version:"
python3 --version

echo "Node.js version:"
node --version

echo "npm version:"
npm --version

echo "Caddy version:"
caddy version

# Test if ports are accessible
print_status "Testing port accessibility..."
if nc -z localhost 22; then
    print_status "✅ Port 22 (SSH) is accessible"
else
    print_warning "⚠️  Port 22 (SSH) may not be accessible"
fi

if nc -z localhost 80; then
    print_status "✅ Port 80 (HTTP) is accessible"
else
    print_warning "⚠️  Port 80 (HTTP) may not be accessible"
fi

if nc -z localhost 443; then
    print_status "✅ Port 443 (HTTPS) is accessible"
else
    print_warning "⚠️  Port 443 (HTTPS) may not be accessible"
fi

print_status "Server setup completed!"
print_status "You can now run the deployment script to deploy your application."
print_warning "Next steps:"
print_warning "1. Upload your application files to the server"
print_warning "2. Run the deploy.sh script"
print_warning "3. Configure your environment variables" 