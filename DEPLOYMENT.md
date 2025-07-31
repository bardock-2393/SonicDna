# SonicDNA Deployment Guide

This guide will help you deploy the SonicDNA project on Ubuntu using Caddy as a reverse proxy.

## Prerequisites

- Ubuntu 20.04 or later
- A domain name (e.g., `soniquedna.deepsantoshwar.xyz`)
- SSH access to your server
- Sudo privileges

## Quick Deployment

### 1. Server Setup

SSH into your Ubuntu server and run the deployment script:

```bash
# Download the deployment script
wget https://raw.githubusercontent.com/bardock-2393/SonicDna/main/deploy.sh

# Make it executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

The script will automatically:
- Update the system
- Install all required packages (Node.js, Python 3.11, Caddy)
- Clone the repository
- Set up backend and frontend services
- Configure Caddy reverse proxy
- Set up firewall
- Create management scripts

### 2. Environment Configuration

After deployment, you need to configure your environment variables:

#### Backend Environment (`/opt/sonicdna/backend2/.env`)

```bash
# Create the backend environment file
sudo nano /opt/sonicdna/backend2/.env
```

Add your environment variables:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
QLOO_API_KEY=your_qloo_api_key
GEMINI_API_KEY=your_gemini_api_key
```

#### Frontend Environment (`/opt/sonicdna/frontend/.env`)

```bash
# Create the frontend environment file
sudo nano /opt/sonicdna/frontend/.env
```

Add any frontend-specific environment variables if needed.

### 3. Spotify App Configuration

Update your Spotify app settings in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard):

- **Redirect URI**: `https://soniquedna.deepsantoshwar.xyz/callback`

### 4. DNS Configuration

Configure your DNS provider to point your domain to your server's IP address:

- **A Record**: `soniquedna.deepsantoshwar.xyz` → `YOUR_SERVER_IP`

### 5. Restart Services

After configuring environment variables:

```bash
# Restart all services
sudo systemctl restart sonicdna-backend
sudo systemctl restart sonicdna-frontend
sudo systemctl restart caddy
```

## Local Development

To run the project locally with the same configuration:

```bash
# Make the local script executable
chmod +x run-local.sh

# Run the local development script
./run-local.sh
```

This will start:
- Backend on `http://localhost:5500`
- Frontend on `http://localhost:8080`

## Management Commands

### Check Service Status

```bash
sudo /opt/sonicdna/status.sh
```

### Update Application

```bash
sudo /opt/sonicdna/update.sh
```

### View Logs

```bash
# Backend logs
sudo journalctl -u sonicdna-backend -f

# Frontend logs
sudo journalctl -u sonicdna-frontend -f

# Caddy logs
sudo journalctl -u caddy -f
```

### Manual Service Control

```bash
# Start services
sudo systemctl start sonicdna-backend
sudo systemctl start sonicdna-frontend
sudo systemctl start caddy

# Stop services
sudo systemctl stop sonicdna-backend
sudo systemctl stop sonicdna-frontend
sudo systemctl stop caddy

# Restart services
sudo systemctl restart sonicdna-backend
sudo systemctl restart sonicdna-frontend
sudo systemctl restart caddy
```

## Architecture Overview

```
Internet → Caddy (Port 80/443) → Frontend (Port 8080) + Backend (Port 5500)
```

- **Caddy**: Reverse proxy with automatic HTTPS
- **Frontend**: React/Vite application served by `serve`
- **Backend**: Flask application with Python virtual environment

## Port Configuration

- **Frontend**: Port 8080 (internal)
- **Backend**: Port 5500 (internal)
- **External**: Port 80 (HTTP) and 443 (HTTPS)

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check service status
   sudo systemctl status sonicdna-backend
   sudo systemctl status sonicdna-frontend
   sudo systemctl status caddy
   
   # Check logs
   sudo journalctl -u sonicdna-backend -n 50
   ```

2. **Port conflicts**
   ```bash
   # Check what's using the ports
   sudo netstat -tlnp | grep :5500
   sudo netstat -tlnp | grep :8080
   ```

3. **Caddy configuration issues**
   ```bash
   # Validate Caddy configuration
   sudo caddy validate --config /etc/caddy/Caddyfile
   
   # Test Caddy configuration
   sudo caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
   ```

4. **Environment variables not loaded**
   ```bash
   # Check if .env files exist
   ls -la /opt/sonicdna/backend2/.env
   ls -la /opt/sonicdna/frontend/.env
   ```

### SSL Certificate Issues

If you encounter SSL certificate issues:

```bash
# Check Caddy logs
sudo journalctl -u caddy -f

# Manually obtain certificate
sudo caddy cert --config /etc/caddy/Caddyfile
```

### Database Issues

If you encounter database-related issues:

```bash
# Check database file permissions
ls -la /opt/sonicdna/backend2/music_recommendations.db

# Check database integrity
cd /opt/sonicdna/backend2
source venv/bin/activate
python check_db.py
```

## Security Considerations

1. **Firewall**: Only ports 22 (SSH), 80 (HTTP), and 443 (HTTPS) are open
2. **Environment Variables**: Sensitive data is stored in `.env` files
3. **Service Isolation**: Each service runs under the same user account
4. **Automatic Updates**: Caddy automatically manages SSL certificates

## Backup and Recovery

### Backup

```bash
# Backup the entire project
sudo tar -czf sonicdna-backup-$(date +%Y%m%d).tar.gz /opt/sonicdna

# Backup environment files
sudo cp /opt/sonicdna/backend2/.env /opt/sonicdna/backend2/.env.backup
sudo cp /opt/sonicdna/frontend/.env /opt/sonicdna/frontend/.env.backup
```

### Recovery

```bash
# Restore from backup
sudo tar -xzf sonicdna-backup-YYYYMMDD.tar.gz -C /

# Restart services
sudo systemctl restart sonicdna-backend
sudo systemctl restart sonicdna-frontend
sudo systemctl restart caddy
```

## Performance Optimization

1. **Enable Gzip Compression**: Already configured in Caddy
2. **Static File Caching**: Configured in Caddy
3. **Database Optimization**: Consider using SQLite with proper indexing
4. **Memory Management**: Monitor service memory usage

## Monitoring

### System Resources

```bash
# Check system resources
htop
df -h
free -h
```

### Service Monitoring

```bash
# Monitor all services
watch -n 5 'systemctl status sonicdna-backend sonicdna-frontend caddy'
```

## Support

If you encounter issues:

1. Check the logs using the commands above
2. Verify your environment variables are correctly set
3. Ensure your DNS is properly configured
4. Check that your Spotify app settings are correct

For additional help, refer to the project's GitHub repository or create an issue. 