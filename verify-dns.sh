#!/bin/bash

# DNS Verification Script for SoniqueDNA
# Run this script to verify your domain configuration

echo "🔍 Verifying DNS Configuration for SoniqueDNA..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

DOMAIN="soniquedna.deepsantoshwar.xyz"
EXPECTED_IP="13.203.35.141"

print_info "Checking DNS configuration for: $DOMAIN"
print_info "Expected IP: $EXPECTED_IP"

# Check if dig is available
if ! command -v dig &> /dev/null; then
    print_warning "dig command not found. Installing..."
    sudo apt update
    sudo apt install -y dnsutils
fi

# Check if nslookup is available
if ! command -v nslookup &> /dev/null; then
    print_warning "nslookup command not found. Installing..."
    sudo apt update
    sudo apt install -y dnsutils
fi

echo ""
print_status "=== DNS Resolution Tests ==="

# Test A record
print_info "Testing A record..."
A_RECORD=$(dig +short A $DOMAIN)
if [ -n "$A_RECORD" ]; then
    if [ "$A_RECORD" = "$EXPECTED_IP" ]; then
        print_status "✅ A record is correct: $A_RECORD"
    else
        print_error "❌ A record mismatch. Expected: $EXPECTED_IP, Got: $A_RECORD"
    fi
else
    print_error "❌ No A record found for $DOMAIN"
fi

# Test CNAME record
print_info "Testing CNAME record..."
CNAME_RECORD=$(dig +short CNAME $DOMAIN)
if [ -n "$CNAME_RECORD" ]; then
    print_warning "⚠️  CNAME record found: $CNAME_RECORD (this might conflict with A record)"
else
    print_status "✅ No CNAME record (good for A record)"
fi

# Test nameservers
print_info "Testing nameservers..."
NS_RECORDS=$(dig +short NS $DOMAIN)
if [ -n "$NS_RECORDS" ]; then
    print_status "✅ Nameservers found:"
    echo "$NS_RECORDS" | while read ns; do
        echo "   - $ns"
    done
else
    print_error "❌ No nameservers found"
fi

# Test child nameserver
CHILD_NS="soniquedna.deepsantoshwar.xyz"
print_info "Testing child nameserver: $CHILD_NS"
CHILD_NS_IP=$(dig +short A $CHILD_NS)
if [ -n "$CHILD_NS_IP" ]; then
    if [ "$CHILD_NS_IP" = "$EXPECTED_IP" ]; then
        print_status "✅ Child nameserver is correct: $CHILD_NS_IP"
    else
        print_error "❌ Child nameserver mismatch. Expected: $EXPECTED_IP, Got: $CHILD_NS_IP"
    fi
else
    print_error "❌ Child nameserver not found"
fi

echo ""
print_status "=== Connectivity Tests ==="

# Test HTTP connectivity
print_info "Testing HTTP connectivity..."
if curl -s --connect-timeout 10 http://$DOMAIN > /dev/null; then
    print_status "✅ HTTP is accessible"
else
    print_warning "⚠️  HTTP not accessible (might be normal if HTTPS redirect is enabled)"
fi

# Test HTTPS connectivity
print_info "Testing HTTPS connectivity..."
if curl -s --connect-timeout 10 https://$DOMAIN > /dev/null; then
    print_status "✅ HTTPS is accessible"
else
    print_error "❌ HTTPS not accessible"
fi

# Test backend API
print_info "Testing backend API..."
if curl -s --connect-timeout 10 https://$DOMAIN/api/health > /dev/null; then
    print_status "✅ Backend API is accessible"
else
    print_warning "⚠️  Backend API not accessible (might be normal if not deployed yet)"
fi

echo ""
print_status "=== SSL Certificate Test ==="

# Test SSL certificate
print_info "Testing SSL certificate..."
SSL_INFO=$(echo | openssl s_client -servername $DOMAIN -connect $DOMAIN:443 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
if [ -n "$SSL_INFO" ]; then
    print_status "✅ SSL certificate is valid"
    echo "$SSL_INFO"
else
    print_warning "⚠️  SSL certificate not found or invalid"
fi

echo ""
print_status "=== Propagation Check ==="

# Check multiple DNS servers
print_info "Checking DNS propagation..."
DNS_SERVERS=("8.8.8.8" "1.1.1.1" "208.67.222.222")

for dns_server in "${DNS_SERVERS[@]}"; do
    print_info "Testing with DNS server: $dns_server"
    RESOLVED_IP=$(dig +short @$dns_server A $DOMAIN)
    if [ -n "$RESOLVED_IP" ]; then
        if [ "$RESOLVED_IP" = "$EXPECTED_IP" ]; then
            print_status "✅ $dns_server: $RESOLVED_IP"
        else
            print_warning "⚠️  $dns_server: $RESOLVED_IP (expected: $EXPECTED_IP)"
        fi
    else
        print_error "❌ $dns_server: No response"
    fi
done

echo ""
print_status "=== Summary ==="

if [ "$A_RECORD" = "$EXPECTED_IP" ] && [ "$CHILD_NS_IP" = "$EXPECTED_IP" ]; then
    print_status "🎉 DNS configuration looks good!"
    print_info "Your domain should be working correctly."
else
    print_error "❌ DNS configuration has issues."
    print_warning "Please check your Hostinger DNS settings."
fi

print_info "If you see propagation issues, wait up to 24 hours for DNS changes to spread globally." 