from flask import Blueprint, request, jsonify
from services.spotify import SpotifyService
from utils.helpers import validate_input_data, sanitize_string, extract_playlist_id_from_url
import os

playlist_routes = Blueprint('playlists', __name__)
spotify_service = SpotifyService()

@playlist_routes.route('/create-playlist', methods=['POST'])
def create_playlist():
    """Create Spotify playlist with recommended tracks"""
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token", "user_id", "name"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        user_id = sanitize_string(data["user_id"])
        name = sanitize_string(data["name"])
        tracks = data.get("tracks", [])
        
        # Create playlist
        playlist = spotify_service.create_playlist(spotify_token, user_id, name)
        
        if not playlist:
            return jsonify({"error": "Failed to create playlist"}), 500
        
        # Add tracks if provided
        if tracks and playlist.get("playlist_id"):
            track_uris = [f"spotify:track:{track}" for track in tracks if track]
            if track_uris:
                success = spotify_service.add_tracks_to_playlist(spotify_token, playlist["playlist_id"], track_uris)
                if not success:
                    print("Warning: Failed to add tracks to playlist")
        
        return jsonify({
            "playlist_id": playlist["playlist_id"],
            "playlist_url": playlist["playlist_url"]
        })
        
    except Exception as e:
        print(f"Playlist creation error: {e}")
        return jsonify({"error": "Failed to create playlist"}), 500

@playlist_routes.route('/search-playlists', methods=['POST'])
def search_playlists():
    """Search user's Spotify playlists"""
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token", "query"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        query = sanitize_string(data["query"])
        
        # Search playlists
        playlists = spotify_service.search_playlists(spotify_token, query)
        
        return jsonify({
            "playlists": playlists
        })
        
    except Exception as e:
        print(f"Playlist search error: {e}")
        return jsonify({"error": "Failed to search playlists"}), 500

@playlist_routes.route('/get-playlist-by-id', methods=['POST'])
def get_playlist_by_id():
    """Get detailed playlist information"""
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token", "playlist_id"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        playlist_id = sanitize_string(data["playlist_id"])
        
        # Get playlist details
        playlist = spotify_service.get_playlist_by_id(spotify_token, playlist_id)
        
        if not playlist:
            return jsonify({"error": "Playlist not found"}), 404
        
        return jsonify({
            "playlist": {
                "id": playlist["id"],
                "name": playlist["name"],
                "description": playlist["description"]
            },
            "tracks": playlist["tracks"]
        })
        
    except Exception as e:
        print(f"Playlist fetch error: {e}")
        return jsonify({"error": "Failed to get playlist"}), 500

@playlist_routes.route('/get-playlist-by-url', methods=['POST'])
def get_playlist_by_url():
    """Extract playlist ID from URL and get details"""
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token", "playlist_url"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        playlist_url = sanitize_string(data["playlist_url"])
        
        # Extract playlist ID from URL
        playlist_id = extract_playlist_id_from_url(playlist_url)
        
        if not playlist_id:
            return jsonify({"error": "Invalid playlist URL"}), 400
        
        # Get playlist details
        playlist = spotify_service.get_playlist_by_id(spotify_token, playlist_id)
        
        if not playlist:
            return jsonify({"error": "Playlist not found"}), 404
        
        return jsonify({
            "playlist": {
                "id": playlist["id"],
                "name": playlist["name"],
                "description": playlist["description"]
            },
            "tracks": playlist["tracks"]
        })
        
    except Exception as e:
        print(f"Playlist URL fetch error: {e}")
        return jsonify({"error": "Failed to get playlist"}), 500 