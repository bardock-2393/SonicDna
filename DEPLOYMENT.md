# 🚀 SoniqueDNA Deployment Guide

This guide will help you deploy your SoniqueDNA application on your EC2 server with Caddy web server.

## 📋 Prerequisites

- EC2 server running Ubuntu (IP: 13.203.35.141)
- Domain: soniquedna.deepsantoshwar.xyz
- SSH access to your EC2 server
- Your API keys (Spotify, Gemini, Qloo)

## 🔧 Step-by-Step Deployment

### 1. **Connect to Your EC2 Server**

```bash
ssh -i your-key.pem ubuntu@13.203.35.141
```

### 2. **Run Server Setup Script**

```bash
# Upload the server-setup.sh script to your server
chmod +x server-setup.sh
./server-setup.sh
```

This will install:
- Python 3 and pip
- Node.js and npm
- Caddy web server
- Configure firewall
- Create necessary directories

### 3. **Upload Your Application**

You have several options:

**Option A: Using Git (Recommended)**
```bash
cd /var/www/soniquedna
git clone <your-repository-url> .
```

**Option B: Using SCP**
```bash
# From your local machine
scp -r -i your-key.pem backend/ ubuntu@13.203.35.141:/var/www/soniquedna/
scp -r -i your-key.pem frontend/ ubuntu@13.203.35.141:/var/www/soniquedna/
```

**Option C: Using rsync**
```bash
rsync -avz -e "ssh -i your-key.pem" backend/ ubuntu@13.203.35.141:/var/www/soniquedna/backend/
rsync -avz -e "ssh -i your-key.pem" frontend/ ubuntu@13.203.35.141:/var/www/soniquedna/frontend/
```

### 4. **Run Deployment Script**

```bash
cd /var/www/soniquedna
chmod +x deploy.sh
./deploy.sh
```

### 5. **Configure Environment Variables**

Edit the backend environment file:
```bash
nano /var/www/soniquedna/backend/.env
```

Replace the placeholder values with your actual API keys:
```bash
# Spotify API Configuration
SPOTIFY_CLIENT_ID=5b5e4ceb834347e6a6c3b998cfaf0088
SPOTIFY_CLIENT_SECRET=your_actual_spotify_client_secret
SPOTIFY_REDIRECT_URI=https://soniquedna.deepsantoshwar.xyz/callback

# Gemini API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key

# Qloo API Configuration
QLOO_API_KEY=your_actual_qloo_api_key

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
FLASK_SECRET_KEY=your_random_secret_key_here
```

### 6. **Update Spotify App Settings**

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Select your app
3. Add `https://soniquedna.deepsantoshwar.xyz/callback` to Redirect URIs
4. Update your app's website URL to `https://soniquedna.deepsantoshwar.xyz`

### 7. **Verify Deployment**

Check if services are running:
```bash
sudo systemctl status soniquedna-backend
sudo systemctl status caddy
```

Test your application:
- Frontend: https://soniquedna.deepsantoshwar.xyz
- Backend health: https://soniquedna.deepsantoshwar.xyz/api/health

## 🔍 Troubleshooting

### Check Logs

**Backend logs:**
```bash
sudo journalctl -u soniquedna-backend -f
```

**Caddy logs:**
```bash
sudo journalctl -u caddy -f
```

**Application logs:**
```bash
tail -f /var/log/soniquedna/backend.log
```

### Common Issues

**1. Backend not starting:**
```bash
# Check if port 8080 is available
sudo netstat -tlnp | grep :8080

# Restart backend service
sudo systemctl restart soniquedna-backend
```

**2. Caddy not starting:**
```bash
# Check Caddy configuration
sudo caddy validate --config /etc/caddy/Caddyfile

# Restart Caddy
sudo systemctl restart caddy
```

**3. SSL certificate issues:**
```bash
# Check Caddy logs for SSL errors
sudo journalctl -u caddy | grep -i ssl
```

**4. Permission issues:**
```bash
# Fix permissions
sudo chown -R ubuntu:ubuntu /var/www/soniquedna
sudo chown -R ubuntu:ubuntu /var/log/soniquedna
```

## 🔄 Updating Your Application

To update your application:

```bash
cd /var/www/soniquedna

# Pull latest changes (if using git)
git pull origin main

# Or upload new files manually

# Run deployment script again
./deploy.sh
```

## 📊 Monitoring

### Health Check Endpoints

- `GET /api/health` - Backend health status
- `GET /api/` - Basic health check

### Performance Monitoring

Monitor your application performance:
```bash
# Check memory usage
free -h

# Check disk usage
df -h

# Check CPU usage
htop
```

## 🔒 Security Considerations

1. **Firewall**: Only ports 22, 80, and 443 are open
2. **SSL**: Caddy automatically manages SSL certificates
3. **Environment Variables**: API keys are stored securely in `.env` files
4. **Service Isolation**: Backend runs as a systemd service

## 📞 Support

If you encounter issues:

1. Check the logs first
2. Verify all environment variables are set correctly
3. Ensure your domain DNS is pointing to the correct IP
4. Verify Spotify app settings are updated

## 🎉 Success!

Once deployed, your SoniqueDNA application will be available at:
- **Frontend**: https://soniquedna.deepsantoshwar.xyz
- **API**: https://soniquedna.deepsantoshwar.xyz/api/*

Your application is now live and ready to use! 🚀 