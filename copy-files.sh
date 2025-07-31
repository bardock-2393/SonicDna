#!/bin/bash

# Script to copy project files to deployment directory

set -e

PROJECT_NAME="sonicdna"
DEPLOY_DIR="/opt/$PROJECT_NAME"

echo "=== Copying Project Files ==="
echo

# Check if we're in the project directory
if [ ! -d "backend2" ] || [ ! -d "frontend" ]; then
    echo "Error: Please run this script from the project root directory (where backend2 and frontend folders are located)"
    exit 1
fi

echo "Copying project files to $DEPLOY_DIR..."

# Create deployment directory if it doesn't exist
sudo mkdir -p $DEPLOY_DIR

# Copy all project files
sudo cp -r backend2 $DEPLOY_DIR/
sudo cp -r frontend $DEPLOY_DIR/
sudo cp -r *.sh $DEPLOY_DIR/ 2>/dev/null || true
sudo cp -r *.md $DEPLOY_DIR/ 2>/dev/null || true

# Set proper ownership
sudo chown -R $USER:$USER $DEPLOY_DIR

echo "Files copied successfully!"
echo "Project is now available at: $DEPLOY_DIR"
echo
echo "You can now continue with the deployment by running:"
echo "cd $DEPLOY_DIR && ./deploy.sh" 