#!/bin/bash

# Quick fix script to install Python 3.11 on Ubuntu server

set -e

echo "=== Python 3.11 Installation Fix ==="
echo

# Check Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu version: $UBUNTU_VERSION"

# Install required packages
echo "Installing required packages..."
sudo apt update
sudo apt install -y software-properties-common

# Add deadsnakes PPA
echo "Adding deadsnakes PPA..."
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install Python 3.11
echo "Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Verify installation
echo "Verifying Python 3.11 installation..."
python3.11 --version

echo
echo "Python 3.11 installation completed successfully!"
echo "You can now run the deployment script again." 