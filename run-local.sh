#!/bin/bash

# SonicDNA Local Development Script
# This script runs the project locally with the same configuration as production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=5500
FRONTEND_PORT=8080

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

# Function to check if Python is installed
check_python() {
    if ! command -v python3.11 &> /dev/null; then
        print_error "Python 3.11 is not installed. Please install it first."
        exit 1
    fi
}

# Function to check if Node.js is installed
check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install it first."
        exit 1
    fi
}

# Function to setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd backend2
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3.11 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Backend setup completed"
}

# Function to setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Frontend setup completed"
}

# Function to start backend
start_backend() {
    print_status "Starting backend on port $BACKEND_PORT..."
    
    cd backend2
    source venv/bin/activate
    
    # Start backend in background
    python app.py &
    BACKEND_PID=$!
    
    print_success "Backend started with PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    print_status "Starting frontend on port $FRONTEND_PORT..."
    
    cd frontend
    
    # Start frontend in background
    npm run dev &
    FRONTEND_PID=$!
    
    print_success "Frontend started with PID: $FRONTEND_PID"
}

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down services..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        print_status "Backend stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_status "Frontend stopped"
    fi
    
    print_success "All services stopped"
    exit 0
}

# Function to display status
show_status() {
    echo
    echo "=== SonicDNA Local Development Status ==="
    echo "Backend: http://localhost:$BACKEND_PORT"
    echo "Frontend: http://localhost:$FRONTEND_PORT"
    echo
    echo "Press Ctrl+C to stop all services"
    echo
}

# Function to check environment files
check_env_files() {
    print_status "Checking environment files..."
    
    if [ ! -f "backend2/.env" ]; then
        print_warning "Backend .env file not found. Please create backend2/.env with your environment variables:"
        print_warning "Required variables: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, QLOO_API_KEY, GEMINI_API_KEY"
    else
        print_success "Backend .env file found"
    fi
    
    if [ ! -f "frontend/.env" ]; then
        print_warning "Frontend .env file not found. Please create frontend/.env if needed"
    else
        print_success "Frontend .env file found"
    fi
}

# Main execution
main() {
    echo "=== SonicDNA Local Development Script ==="
    echo
    
    # Check prerequisites
    check_python
    check_node
    
    # Check environment files
    check_env_files
    
    # Setup services
    setup_backend
    setup_frontend
    
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM
    
    # Start services
    start_backend
    sleep 2  # Give backend time to start
    start_frontend
    
    # Show status
    show_status
    
    # Wait for user to stop
    wait
}

# Run main function
main "$@" 