import requests
import base64
import time
from utils import retry_on_failure

def exchange_code_for_token(code, client_id, client_secret, redirect_uri):
    """Exchange authorization code for access token"""
    url = "https://accounts.spotify.com/api/token"
    
    # Encode client credentials
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error exchanging code for token: {e}")
        return None

def get_spotify_user_id(access_token):
    """Get Spotify user ID"""
    url = "https://api.spotify.com/v1/me"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json().get("id")
    except Exception as e:
        print(f"Error getting user ID: {e}")
        return None

def get_spotify_user_profile(access_token):
    """Get Spotify user profile"""
    url = "https://api.spotify.com/v1/me"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return None

def get_spotify_top_artists(access_token, limit=10, time_range="medium_term"):
    """Get user's top artists from Spotify"""
    url = f"https://api.spotify.com/v1/me/top/artists"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "limit": limit,
        "time_range": time_range
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        artists = []
        for item in response.json().get("items", []):
            artists.append(item.get("name"))
        
        return artists
    except Exception as e:
        print(f"Error getting top artists: {e}")
        return []

def get_spotify_top_artists_with_images(access_token, limit=10, time_range="medium_term"):
    """Get user's top artists with images from Spotify"""
    url = f"https://api.spotify.com/v1/me/top/artists"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "limit": limit,
        "time_range": time_range
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        artists = []
        for item in response.json().get("items", []):
            artist_info = {
                "name": item.get("name"),
                "id": item.get("id"),
                "image": item.get("images", [{}])[0].get("url") if item.get("images") else None,
                "genres": item.get("genres", [])
            }
            artists.append(artist_info)
        
        return artists
    except Exception as e:
        print(f"Error getting top artists with images: {e}")
        return []

def get_spotify_top_tracks(access_token, limit=10, time_range="medium_term"):
    """Get user's top tracks from Spotify"""
    url = f"https://api.spotify.com/v1/me/top/tracks"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "limit": limit,
        "time_range": time_range
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        tracks = []
        for item in response.json().get("items", []):
            tracks.append(item.get("name"))
        
        return tracks
    except Exception as e:
        print(f"Error getting top tracks: {e}")
        return []

def get_spotify_recently_played(access_token, limit=10):
    """Get user's recently played tracks from Spotify"""
    url = "https://api.spotify.com/v1/me/player/recently-played"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": limit}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        artists = []
        for item in response.json().get("items", []):
            track = item.get("track", {})
            if track:
                artists.append(track.get("artists", [{}])[0].get("name"))
        
        return artists
    except Exception as e:
        print(f"Error getting recently played: {e}")
        return []

def get_spotify_artist_id(artist_name, token):
    """Get Spotify artist ID for a given artist name"""
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": artist_name,
        "type": "artist",
        "limit": 1
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        artists = response.json().get("artists", {}).get("items", [])
        if artists:
            return artists[0].get("id")
        return None
    except Exception as e:
        print(f"Error getting artist ID for {artist_name}: {e}")
        return None

def get_spotify_artist_genres(artist_name, token):
    """Get Spotify artist genres for a given artist name"""
    artist_id = get_spotify_artist_id(artist_name, token)
    if not artist_id:
        return []
    
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        return response.json().get("genres", [])
    except Exception as e:
        print(f"Error getting artist genres for {artist_name}: {e}")
        return []

def get_audio_features(track_ids, access_token):
    """Get audio features for multiple tracks"""
    url = "https://api.spotify.com/v1/audio-features"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"ids": ",".join(track_ids)}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        return response.json().get("audio_features", [])
    except Exception as e:
        print(f"Error getting audio features: {e}")
        return []

def get_artist_tracks_smart(artist_id, access_token, limit=15):
    """Get tracks for a given artist using smart selection"""
    tracks = []
    
    # First try to get top tracks
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"market": "US"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        top_tracks = response.json().get("tracks", [])
        tracks.extend(top_tracks[:limit//2])  # Add half from top tracks
        
        # If we need more tracks, get from albums
        if len(tracks) < limit:
            remaining_limit = limit - len(tracks)
            
            # Get artist albums
            albums_url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
            albums_params = {
                "include_groups": "album,single",
                "limit": 5,
                "market": "US"
            }
            
            albums_response = requests.get(albums_url, headers=headers, params=albums_params, timeout=5)
            albums_response.raise_for_status()
            
            albums = albums_response.json().get("items", [])
            
            for album in albums[:3]:  # Check first 3 albums
                album_id = album.get("id")
                album_tracks_url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
                album_tracks_params = {"limit": remaining_limit//3, "market": "US"}
                
                album_tracks_response = requests.get(album_tracks_url, headers=headers, params=album_tracks_params, timeout=5)
                album_tracks_response.raise_for_status()
                
                album_tracks = album_tracks_response.json().get("items", [])
                tracks.extend(album_tracks)
                
                if len(tracks) >= limit:
                    break
        
        return tracks[:limit]
        
    except Exception as e:
        print(f"Error getting artist tracks for {artist_id}: {e}")
        return []

def get_trending_tracks_for_context(context_type, access_token, limit=10):
    """Get trending tracks for a given context"""
    # This is a simplified implementation - in a real app, you'd use more sophisticated logic
    url = "https://api.spotify.com/v1/browse/new-releases"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": limit, "country": "US"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        
        tracks = []
        for album in response.json().get("albums", {}).get("items", []):
            album_id = album.get("id")
            
            # Get tracks from this album
            album_tracks_url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
            album_tracks_params = {"limit": 3, "market": "US"}
            
            album_tracks_response = requests.get(album_tracks_url, headers=headers, params=album_tracks_params, timeout=5)
            album_tracks_response.raise_for_status()
            
            album_tracks = album_tracks_response.json().get("items", [])
            tracks.extend(album_tracks)
            
            if len(tracks) >= limit:
                break
        
        return tracks[:limit]
        
    except Exception as e:
        print(f"Error getting trending tracks for context {context_type}: {e}")
        return []



def create_spotify_playlist(user_id, name, description, access_token):
    """Create a new Spotify playlist"""
    url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "name": name,
        "description": description,
        "public": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error creating playlist: {e}")
        return None

def add_tracks_to_playlist(playlist_id, track_uris, access_token):
    """Add tracks to a Spotify playlist"""
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    data = {"uris": track_uris}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error adding tracks to playlist: {e}")
        return None

def get_spotify_track_uri(track_url, access_token):
    """Get Spotify track URI from a track URL"""
    # Extract track ID from URL
    if "track/" in track_url:
        track_id = track_url.split("track/")[1].split("?")[0]
        return f"spotify:track:{track_id}"
    return None

def get_user_country(access_token):
    """Get user's country from Spotify profile"""
    profile = get_spotify_user_profile(access_token)
    if profile:
        return profile.get("country")
    return None

def analyze_track_emotional_context(audio_features, track_name, artist_name):
    """Analyze track emotional context from audio features"""
    if not audio_features:
        return {"mood": "unknown", "energy": "medium", "confidence": 0.0}
    
    features = audio_features[0] if isinstance(audio_features, list) else audio_features
    
    # Extract key features
    valence = features.get("valence", 0.5)  # Happiness (0-1)
    energy = features.get("energy", 0.5)    # Energy level (0-1)
    danceability = features.get("danceability", 0.5)  # Danceability (0-1)
    tempo = features.get("tempo", 120)      # BPM
    
    # Determine mood based on valence
    if valence > 0.7:
        mood = "happy"
    elif valence > 0.4:
        mood = "neutral"
    else:
        mood = "sad"
    
    # Determine energy level
    if energy > 0.7:
        energy_level = "high"
    elif energy > 0.4:
        energy_level = "medium"
    else:
        energy_level = "low"
    
    # Calculate confidence based on feature strength
    confidence = min(abs(valence - 0.5) * 2 + abs(energy - 0.5) * 2, 1.0)
    
    return {
        "mood": mood,
        "energy": energy_level,
        "danceability": danceability,
        "tempo": tempo,
        "confidence": confidence,
        "track_name": track_name,
        "artist_name": artist_name
    }

def filter_tracks_by_context(tracks, access_token, context_type, user_artists=None):
    """Filter tracks based on context type"""
    if not tracks:
        return []
    
    # Get audio features for tracks
    track_ids = [track.get("id") for track in tracks if track.get("id")]
    if not track_ids:
        return tracks
    
    audio_features = get_audio_features(track_ids, access_token)
    if not audio_features:
        return tracks
    
    # Create track-feature mapping
    track_features = {}
    for i, track in enumerate(tracks):
        if track.get("id") and i < len(audio_features):
            track_features[track["id"]] = audio_features[i]
    
    # Filter based on context
    filtered_tracks = []
    for track in tracks:
        track_id = track.get("id")
        if not track_id or track_id not in track_features:
            continue
        
        features = track_features[track_id]
        is_suitable = False
        
        if context_type == "dance":
            # High energy and danceability
            is_suitable = (features.get("energy", 0) > 0.6 and 
                          features.get("danceability", 0) > 0.6)
        
        elif context_type == "workout":
            # High energy and tempo
            is_suitable = (features.get("energy", 0) > 0.7 and 
                          features.get("tempo", 120) > 120)
        
        elif context_type == "study":
            # Lower energy, higher valence
            is_suitable = (features.get("energy", 0) < 0.6 and 
                          features.get("valence", 0.5) > 0.4)
        
        elif context_type == "relaxing":
            # Low energy, higher valence
            is_suitable = (features.get("energy", 0) < 0.4 and 
                          features.get("valence", 0.5) > 0.5)
        
        else:
            # General context - include all
            is_suitable = True
        
        if is_suitable:
            filtered_tracks.append(track)
    
    return filtered_tracks


def get_spotify_top_tracks_detailed(access_token, limit=10, time_range="medium_term"):
    """Get detailed top tracks with artist information"""
    url = "https://api.spotify.com/v1/me/top/tracks"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": limit, "time_range": time_range}
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        tracks_with_artists = []
        for track in data['items']:
            track_info = {
                'name': track['name'],
                'artists': [artist['name'] for artist in track['artists']],
                'album': track['album']['name'],
                'id': track['id'],
                'popularity': track['popularity']
            }
            tracks_with_artists.append(track_info)
        
        return tracks_with_artists
    except Exception as e:
        print(f"Error fetching detailed top tracks: {e}")
        return []

def get_spotify_playlist_by_id(access_token, playlist_id, market=None):
    """Get a specific Spotify playlist by ID"""
    try:
        url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        params = {}
        if market:
            params['market'] = market
            
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        playlist = response.json()
        
        # Extract playlist details
        playlist_info = {
            'id': playlist['id'],
            'name': playlist['name'],
            'description': playlist.get('description', ''),
            'owner': playlist['owner']['display_name'],
            'tracks_count': playlist['tracks']['total'],
            'public': playlist['public'],
            'images': playlist.get('images', []),
            'external_url': playlist['external_urls']['spotify'],
            'followers': playlist.get('followers', {}).get('total', 0),
            'tracks': []
        }
        
        # Extract track information
        for item in playlist['tracks']['items']:
            track = item['track']
            if track:  # Some tracks might be None if unavailable
                track_info = {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'external_url': track['external_urls']['spotify']
                }
                playlist_info['tracks'].append(track_info)
        
        return playlist_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting Spotify playlist by ID: {e}")
        return None 
