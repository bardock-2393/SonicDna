#!/bin/bash

# SonicDNA Deployment Script for Ubuntu Server
# This script sets up the entire deployment environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="soniquedna.deepsantoshwar.xyz"
BACKEND_PORT=5500
FRONTEND_PORT=8080
PROJECT_NAME="sonicdna"
GITHUB_REPO="https://github.com/bardock-2393/SonicDna.git"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root. Please run as a regular user with sudo privileges."
        exit 1
    fi
}

# Function to update system
update_system() {
    print_status "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    print_success "System updated successfully"
}

# Function to install required packages
install_packages() {
    print_status "Installing required packages..."
    
    # Install essential packages
    sudo apt install -y curl wget git unzip software-properties-common apt-transport-https ca-certificates gnupg lsb-release
    
    # Install Node.js 18.x
    print_status "Installing Node.js 18.x..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt install -y nodejs
    
    # Install Python 3.11 and pip
    print_status "Installing Python 3.11..."
    sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
    
    # Install Caddy
    print_status "Installing Caddy..."
    sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
    sudo apt update
    sudo apt install -y caddy
    
    print_success "All packages installed successfully"
}

# Function to create project directory
setup_project() {
    print_status "Setting up project directory..."
    
    # Create project directory
    sudo mkdir -p /opt/$PROJECT_NAME
    sudo chown $USER:$USER /opt/$PROJECT_NAME
    
    print_success "Project directory created at /opt/$PROJECT_NAME"
    print_warning "Please copy your project files to /opt/$PROJECT_NAME manually"
}

# Function to setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd /opt/$PROJECT_NAME/backend2
    
    # Create virtual environment
    python3.11 -m venv venv
    source venv/bin/activate
    
    # Install Python dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Create systemd service for backend
    sudo tee /etc/systemd/system/${PROJECT_NAME}-backend.service > /dev/null <<EOF
[Unit]
Description=SonicDNA Backend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/$PROJECT_NAME/backend2
Environment=PATH=/opt/$PROJECT_NAME/backend2/venv/bin
ExecStart=/opt/$PROJECT_NAME/backend2/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start backend service
    sudo systemctl daemon-reload
    sudo systemctl enable ${PROJECT_NAME}-backend
    sudo systemctl start ${PROJECT_NAME}-backend
    
    print_success "Backend setup completed"
}

# Function to setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd /opt/$PROJECT_NAME/frontend
    
    # Install Node.js dependencies
    npm install
    
    # Build frontend for production
    npm run build
    
    # Install serve globally for production serving
    sudo npm install -g serve
    
    # Create systemd service for frontend
    sudo tee /etc/systemd/system/${PROJECT_NAME}-frontend.service > /dev/null <<EOF
[Unit]
Description=SonicDNA Frontend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/$PROJECT_NAME/frontend
ExecStart=/usr/bin/serve -s dist -l $FRONTEND_PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start frontend service
    sudo systemctl daemon-reload
    sudo systemctl enable ${PROJECT_NAME}-frontend
    sudo systemctl start ${PROJECT_NAME}-frontend
    
    print_success "Frontend setup completed"
}

# Function to configure Caddy
configure_caddy() {
    print_status "Configuring Caddy..."
    
    # Create Caddyfile
    sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
$DOMAIN {
    # Frontend (port 8080)
    handle /* {
        reverse_proxy localhost:$FRONTEND_PORT
    }
    
    # Backend API routes (port 5500)
    handle /auth/* {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /recommendations/* {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /playlists/* {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /spotify-auth-url {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /exchange-token {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /refresh-token {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /check-token {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /spotify-profile {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /logout {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /spotify-session-clear {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /musicrecommandation {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /musicrecommendation {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /music-recommendation {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /crossdomain-recommendations {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /crossdomain-progress/* {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /create-playlist {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /test-database {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /user-analytics/* {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /user-history/* {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /clear-cache {
        reverse_proxy localhost:$BACKEND_PORT
    }
    
    handle /health {
        reverse_proxy localhost:$BACKEND_PORT
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
    
    # Test Caddy configuration
    sudo caddy validate --config /etc/caddy/Caddyfile
    
    # Restart Caddy
    sudo systemctl restart caddy
    sudo systemctl enable caddy
    
    print_success "Caddy configuration completed"
}

# Function to setup firewall
setup_firewall() {
    print_status "Setting up firewall..."
    
    # Allow SSH, HTTP, and HTTPS
    sudo ufw allow ssh
    sudo ufw allow 80
    sudo ufw allow 443
    
    # Enable firewall
    sudo ufw --force enable
    
    print_success "Firewall configured"
}

# Function to create environment files
create_env_files() {
    print_status "Creating environment files..."
    
    # Backend environment file
    if [ ! -f "/opt/$PROJECT_NAME/backend2/.env" ]; then
        print_warning "Please create /opt/$PROJECT_NAME/backend2/.env file with your environment variables"
        print_warning "Required variables: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, QLOO_API_KEY, GEMINI_API_KEY"
    fi
    
    # Frontend environment file
    if [ ! -f "/opt/$PROJECT_NAME/frontend/.env" ]; then
        print_warning "Please create /opt/$PROJECT_NAME/frontend/.env file with your environment variables"
    fi
    
    print_success "Environment files checked"
}

# Function to create update script
create_update_script() {
    print_status "Creating update script..."
    
    sudo tee /opt/$PROJECT_NAME/update.sh > /dev/null <<'EOF'
#!/bin/bash

# SonicDNA Update Script
set -e

PROJECT_NAME="sonicdna"
PROJECT_DIR="/opt/$PROJECT_NAME"

echo "Updating SonicDNA..."

# Navigate to project directory
cd $PROJECT_DIR

# Pull latest changes
git pull origin main

# Update backend
echo "Updating backend..."
cd backend2
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart ${PROJECT_NAME}-backend

# Update frontend
echo "Updating frontend..."
cd ../frontend
npm install
npm run build
sudo systemctl restart ${PROJECT_NAME}-frontend

echo "Update completed successfully!"
EOF
    
    chmod +x /opt/$PROJECT_NAME/update.sh
    print_success "Update script created at /opt/$PROJECT_NAME/update.sh"
}

# Function to create status check script
create_status_script() {
    print_status "Creating status check script..."
    
    sudo tee /opt/$PROJECT_NAME/status.sh > /dev/null <<'EOF'
#!/bin/bash

# SonicDNA Status Check Script

PROJECT_NAME="sonicdna"

echo "=== SonicDNA Service Status ==="
echo

echo "Backend Service:"
sudo systemctl status ${PROJECT_NAME}-backend --no-pager -l

echo
echo "Frontend Service:"
sudo systemctl status ${PROJECT_NAME}-frontend --no-pager -l

echo
echo "Caddy Service:"
sudo systemctl status caddy --no-pager -l

echo
echo "Port Status:"
echo "Backend (5500): $(netstat -tlnp | grep :5500 || echo 'Not listening')"
echo "Frontend (8080): $(netstat -tlnp | grep :8080 || echo 'Not listening')"

echo
echo "Caddy Configuration:"
sudo caddy validate --config /etc/caddy/Caddyfile
EOF
    
    chmod +x /opt/$PROJECT_NAME/status.sh
    print_success "Status script created at /opt/$PROJECT_NAME/status.sh"
}

# Function to display final instructions
display_instructions() {
    print_success "Deployment completed successfully!"
    echo
    echo "=== Next Steps ==="
    echo "1. Configure your environment variables:"
    echo "   - Backend: /opt/$PROJECT_NAME/backend2/.env"
    echo "   - Frontend: /opt/$PROJECT_NAME/frontend/.env"
    echo
    echo "2. Update your Spotify app settings:"
    echo "   - Redirect URI: https://$DOMAIN/callback"
    echo
    echo "3. Check service status:"
    echo "   sudo /opt/$PROJECT_NAME/status.sh"
    echo
    echo "4. Update the application:"
    echo "   sudo /opt/$PROJECT_NAME/update.sh"
    echo
    echo "5. View logs:"
    echo "   sudo journalctl -u ${PROJECT_NAME}-backend -f"
    echo "   sudo journalctl -u ${PROJECT_NAME}-frontend -f"
    echo "   sudo journalctl -u caddy -f"
    echo
    echo "Your application should be available at: https://$DOMAIN"
    echo
    print_warning "Don't forget to configure your DNS to point $DOMAIN to this server's IP address!"
}

# Main execution
main() {
    echo "=== SonicDNA Ubuntu Deployment Script ==="
    echo
    
    check_root
    update_system
    install_packages
    setup_project
    setup_backend
    setup_frontend
    configure_caddy
    setup_firewall
    create_env_files
    create_update_script
    create_status_script
    display_instructions
}

# Run main function
main "$@" 