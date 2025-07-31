from flask import Blueprint, request, jsonify
from services.spotify import SpotifyService
from utils.helpers import validate_input_data, sanitize_string
import os

auth_routes = Blueprint('auth', __name__)
spotify_service = SpotifyService()

@auth_routes.route('/spotify-auth-url', methods=['GET', 'POST'])
def spotify_auth_url():
    """Generate Spotify OAuth URL with state parameter"""
    try:
        # Handle both GET (query params) and POST (JSON body)
        if request.method == 'GET':
            redirect_uri = request.args.get('redirect_uri')
            force_reauth = request.args.get('force_reauth', 'false').lower() == 'true'
            session_id = request.args.get('session_id')
        else:
            data = request.get_json()
            redirect_uri = data.get("redirect_uri") if data else None
            force_reauth = data.get("force_reauth", False) if data else False
            session_id = data.get("session_id") if data else None
        
        if not redirect_uri:
            return jsonify({"error": "Missing redirect_uri parameter"}), 400
        
        redirect_uri = sanitize_string(redirect_uri)
        
        # Generate auth URL with re-authentication support
        auth_data = spotify_service.generate_auth_url(redirect_uri, force_reauth=force_reauth, session_id=session_id)
        
        return jsonify({
            "auth_url": auth_data["auth_url"],
            "state": auth_data["state"]
        })
        
    except Exception as e:
        print(f"Auth URL generation error: {e}")
        return jsonify({"error": "Failed to generate auth URL"}), 500

@auth_routes.route('/exchange-token', methods=['POST'])
def exchange_token():
    """Exchange authorization code for Spotify access token"""
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["code"]):
            return jsonify({"error": "Missing code parameter"}), 400
        
        code = sanitize_string(data["code"])
        redirect_uri = sanitize_string(data.get("redirect_uri", "http://127.0.0.1:8080/callback"))
        
        # Exchange code for token
        token_data = spotify_service.exchange_token(code, redirect_uri)
        
        if not token_data:
            return jsonify({"error": "Failed to exchange token"}), 400
        
        return jsonify({
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "expires_in": token_data.get("expires_in")
        })
        
    except Exception as e:
        print(f"Token exchange error: {e}")
        return jsonify({"error": "Failed to exchange token"}), 500

@auth_routes.route('/spotify-profile', methods=['POST'])
def spotify_profile():
    """Get user profile and validate token"""
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        
        # Get user profile
        profile = spotify_service.get_user_profile(spotify_token)
        
        if not profile:
            return jsonify({"error": "Failed to get user profile"}), 400
        
        return jsonify({
            "id": profile["user_id"],
            "display_name": profile["name"],
            "images": [{"url": profile["avatar"]}] if profile["avatar"] else [],
            "country": profile["country"]
        })
        
    except Exception as e:
        print(f"Profile fetch error: {e}")
        return jsonify({"error": "Failed to get user profile"}), 500

@auth_routes.route('/logout', methods=['POST'])
def logout():
    """Clear user session - stateless operation"""
    try:
        data = request.get_json()
        
        # Accept access_token, client_id, client_secret (for token revocation)
        access_token = data.get("access_token")
        client_id = data.get("client_id")
        client_secret = data.get("client_secret")
        
        # Optional: Revoke token if credentials provided
        if access_token and client_id and client_secret:
            try:
                # In a real implementation, you would call Spotify's token revocation endpoint
                print(f"Token revocation requested for client: {client_id}")
            except Exception as e:
                print(f"Token revocation failed: {e}")
        
        # Generate re-authentication URL
        import secrets
        import time
        unique_state = f"{secrets.token_urlsafe(32)}_{int(time.time())}"
        session_id = f"session_{secrets.token_urlsafe(16)}_{int(time.time())}"
        
        # Stateless logout - return success with re-auth URL
        return jsonify({
            "success": True,
            "message": "Logged out successfully",
            "force_reauth": True,
            "unique_state": unique_state,
            "session_id": session_id,
            "reauth_url": f"http://localhost:5500/auth/spotify-auth-url?redirect_uri=http://127.0.0.1:8080/callback&force_reauth=true&session_id={session_id}"
        })
        
    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({"error": "Failed to logout"}), 500

@auth_routes.route('/spotify-session-clear', methods=['POST'])
def spotify_session_clear():
    """Clear Spotify session and force re-authentication"""
    try:
        import secrets
        import time
        session_id = f"session_{secrets.token_urlsafe(16)}_{int(time.time())}"
        
        return jsonify({
            'success': True,
            'message': 'Spotify session cleared',
            'session_id': session_id,
            'reauth_url': f"http://localhost:5500/auth/spotify-auth-url?redirect_uri=http://127.0.0.1:8080/callback&force_reauth=true&session_id={session_id}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 