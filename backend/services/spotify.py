import requests
import base64
import time
import random
from typing import Dict, List, Optional
import os

class SpotifyService:
    """Optimized Spotify API service with minimal overhead"""
    
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID', "5b5e4ceb834347e6a6c3b998cfaf0088")
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET', "9c9aadd2b18e49859df887e5e9cc6ede")
        self.base_url = "https://api.spotify.com/v1"
        self.auth_url = "https://accounts.spotify.com/api/token"
    
    def generate_auth_url(self, redirect_uri: str, force_reauth: bool = False, session_id: str = None) -> Dict[str, str]:
        """Generate Spotify OAuth URL with state parameter"""
        import secrets
        import time
        
        # Generate unique state with timestamp for re-authentication
        if force_reauth:
            state = f"{secrets.token_urlsafe(32)}_{int(time.time())}"
        else:
            state = self._generate_state()
            
        # Updated scope to include audio features access
        scope = "user-read-private user-read-email user-top-read user-read-recently-played playlist-modify-public playlist-modify-private user-read-playback-state user-read-currently-playing"
        
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state
        }
        
        # Force re-authentication if requested
        if force_reauth:
            params["show_dialog"] = "true"  # Force Spotify to show the authorization dialog
        
        auth_url = f"https://accounts.spotify.com/authorize?{self._build_query_string(params)}"
        return {"auth_url": auth_url, "state": state}
    
    def exchange_token(self, code: str, redirect_uri: str) -> Optional[Dict]:
        """Exchange authorization code for access token"""
        credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        
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
            response = requests.post(self.auth_url, headers=headers, data=data, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Token exchange error: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict]:
        """Refresh access token using refresh token"""
        credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        
        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
        
        try:
            response = requests.post(self.auth_url, headers=headers, data=data, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Token refresh error: {e}")
            return None
    
    def is_token_expired(self, access_token: str) -> bool:
        """Check if token is expired by making a test API call"""
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(f"{self.base_url}/me", headers=headers, timeout=5)
            return response.status_code == 401
        except Exception:
            return True
    
    def get_user_profile(self, access_token: str) -> Optional[Dict]:
        """Get user profile - single API call"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = requests.get(f"{self.base_url}/me", headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                "user_id": data.get("id"),
                "name": data.get("display_name"),
                "avatar": data.get("images", [{}])[0].get("url") if data.get("images") else None,
                "country": data.get("country")
            }
        except Exception as e:
            print(f"Profile fetch error: {e}")
            return None
    
    def get_top_artists_with_images(self, access_token: str, limit: int = 10, time_range: str = "medium_term") -> List[Dict]:
        """Get user's top artists with images from Spotify"""
        url = f"{self.base_url}/me/top/artists"
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
                    "genres": item.get("genres", []),
                    "popularity": item.get("popularity", 0),
                    "followers": item.get("followers", {}).get("total", 0) if item.get("followers") else 0
                }
                artists.append(artist_info)
            
            return artists
        except Exception as e:
            print(f"Error getting top artists with images: {e}")
            return []

    def get_top_artists_with_genres(self, access_token: str, limit: int = 20, time_range: str = "long_term") -> List[Dict]:
        """Get user's top artists with genres from Spotify"""
        url = f"{self.base_url}/me/top/artists"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "limit": limit,
            "time_range": time_range
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            artists = []
            for item in response.json().get("items", []):
                artist_info = {
                    "name": item.get("name"),
                    "id": item.get("id"),
                    "image": item.get("images", [{}])[0].get("url") if item.get("images") else None,
                    "genres": item.get("genres", []),
                    "popularity": item.get("popularity", 0)
                }
                artists.append(artist_info)
            
            print(f"[SPOTIFY] Got {len(artists)} top artists with genres")
            return artists
            
        except Exception as e:
            print(f"[SPOTIFY] Error getting top artists with genres: {e}")
            return []
    
    def get_top_tracks_detailed(self, access_token: str, limit: int = 20, time_range: str = "long_term") -> List[Dict]:
        """Get user's top tracks with detailed info from Spotify"""
        url = f"{self.base_url}/me/top/tracks"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "limit": limit,
            "time_range": time_range
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            tracks = []
            for item in response.json().get("items", []):
                track_info = {
                    "name": item.get("name"),
                    "id": item.get("id"),
                    "artist": item.get("artists", [{}])[0].get("name") if item.get("artists") else "Unknown",
                    "artist_id": item.get("artists", [{}])[0].get("id") if item.get("artists") else None,
                    "album": item.get("album", {}).get("name") if item.get("album") else "Unknown",
                    "image": item.get("album", {}).get("images", [{}])[0].get("url") if item.get("album", {}).get("images") else None,
                    "popularity": item.get("popularity", 0)
                }
                tracks.append(track_info)
            
            print(f"[SPOTIFY] Got {len(tracks)} top tracks detailed")
            return tracks
            
        except Exception as e:
            print(f"[SPOTIFY] Error getting top tracks detailed: {e}")
            return []

    def get_user_data_fast(self, access_token: str) -> Dict:
        """Get all user data in parallel - optimized for speed"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Get user profile first
        profile = self.get_user_profile(access_token)
        if not profile:
            return {}
        
        # Get top artists and tracks in parallel
        artists_url = f"{self.base_url}/me/top/artists?limit=10&time_range=medium_term"
        tracks_url = f"{self.base_url}/me/top/tracks?limit=10&time_range=medium_term"
        
        try:
            # Make parallel requests
            artist_response = requests.get(artists_url, headers=headers, timeout=5)
            track_response = requests.get(tracks_url, headers=headers, timeout=5)
            
            artists = []
            if artist_response.status_code == 200:
                artist_data = artist_response.json()
                artists = [{"name": item["name"], "id": item["id"], "genres": item.get("genres", [])} 
                          for item in artist_data.get("items", [])]
            
            tracks = []
            if track_response.status_code == 200:
                track_data = track_response.json()
                tracks = [{"name": item["name"], "id": item["id"], "artist": item["artists"][0]["name"]} 
                         for item in track_data.get("items", [])]
            
            return {
                "profile": profile,
                "artists": artists,
                "tracks": tracks
            }
        except Exception as e:
            print(f"User data fetch error: {e}")
            return {"profile": profile, "artists": [], "tracks": []}
    
    def create_playlist(self, access_token: str, user_id: str, name: str, description: str = "") -> Optional[Dict]:
        """Create Spotify playlist - single API call"""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "name": name,
            "description": description,
            "public": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/users/{user_id}/playlists",
                headers=headers,
                json=data,
                timeout=5
            )
            response.raise_for_status()
            playlist_data = response.json()
            
            return {
                "playlist_id": playlist_data["id"],
                "playlist_url": playlist_data["external_urls"]["spotify"]
            }
        except Exception as e:
            print(f"Playlist creation error: {e}")
            return None
    
    def add_tracks_to_playlist(self, access_token: str, playlist_id: str, track_uris: List[str]) -> bool:
        """Add tracks to playlist - single API call"""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        data = {"uris": track_uris}
        
        try:
            response = requests.post(
                f"{self.base_url}/playlists/{playlist_id}/tracks",
                headers=headers,
                json=data,
                timeout=5
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Add tracks error: {e}")
            return False
    
    def search_artist(self, access_token: str, artist_name: str) -> Optional[Dict]:
        """Search for artist - single API call"""
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"q": artist_name, "type": "artist", "limit": 1}
        
        try:
            response = requests.get(f"{self.base_url}/search", headers=headers, params=params, timeout=10)
            
            # Check for specific error codes
            if response.status_code == 401:
                print(f"Artist search unauthorized (401) - token may be expired")
                return None
            elif response.status_code == 403:
                print(f"Artist search forbidden (403) - token may not have required scopes")
                return None
            elif response.status_code == 429:
                print(f"Artist search rate limited (429) - too many requests")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            artists = data.get("artists", {}).get("items", [])
            if artists:
                artist = artists[0]
                print(f"Found artist: {artist['name']} (ID: {artist['id']})")
                return {
                    "id": artist["id"],
                    "name": artist["name"],
                    "genres": artist.get("genres", [])
                }
            else:
                print(f"No artist found for: {artist_name}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Network error in artist search: {e}")
            return None
        except Exception as e:
            print(f"Artist search error: {e}")
            return None
    
    def get_artist_details(self, access_token: str, artist_id: str) -> Optional[Dict]:
        """Get detailed artist information including image and Spotify URL"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = requests.get(f"{self.base_url}/artists/{artist_id}", headers=headers, timeout=10)
            
            # Check for specific error codes
            if response.status_code == 401:
                print(f"Artist details unauthorized (401) - token may be expired")
                return None
            elif response.status_code == 403:
                print(f"Artist details forbidden (403) - token may not have required scopes")
                return None
            elif response.status_code == 429:
                print(f"Artist details rate limited (429) - too many requests")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Get the best quality image (usually the first one is the highest quality)
            image_url = None
            if data.get("images"):
                # Try to get the medium size image (around 300x300)
                for image in data["images"]:
                    if image.get("width", 0) >= 200 and image.get("width", 0) <= 400:
                        image_url = image.get("url")
                        break
                # If no medium image found, use the first one
                if not image_url and data["images"]:
                    image_url = data["images"][0].get("url")
            
            artist_details = {
                "id": data.get("id"),
                "name": data.get("name"),
                "image": image_url,
                "genres": data.get("genres", []),
                "popularity": data.get("popularity", 0),
                "followers": data.get("followers", {}).get("total", 0) if data.get("followers") else 0,
                "spotify_url": data.get("external_urls", {}).get("spotify", ""),
                "uri": data.get("uri", "")
            }
            
            print(f"Artist details fetched for {data.get('name')}: image={image_url is not None}, spotify_url={artist_details['spotify_url']}")
            return artist_details
            
        except requests.exceptions.RequestException as e:
            print(f"Network error in artist details fetch: {e}")
            return None
        except Exception as e:
            print(f"Artist details fetch error: {e}")
            return None
    
    def get_playlist_by_id(self, access_token: str, playlist_id: str) -> Optional[Dict]:
        """Get playlist details - single API call"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            response = requests.get(f"{self.base_url}/playlists/{playlist_id}", headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                "id": data["id"],
                "name": data["name"],
                "description": data.get("description", ""),
                "tracks": [{"name": item["track"]["name"], "artist": item["track"]["artists"][0]["name"]} 
                          for item in data.get("tracks", {}).get("items", []) if item["track"]]
            }
        except Exception as e:
            print(f"Playlist fetch error: {e}")
            return None
    
    def search_playlists(self, access_token: str, query: str) -> List[Dict]:
        """Search for playlists"""
        url = f"{self.base_url}/search"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "q": query,
            "type": "playlist",
            "limit": 10
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            
            playlists = []
            for item in response.json().get("playlists", {}).get("items", []):
                playlist_info = {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "image": item.get("images", [{}])[0].get("url") if item.get("images") else None,
                    "owner": item.get("owner", {}).get("display_name"),
                    "tracks_count": item.get("tracks", {}).get("total", 0)
                }
                playlists.append(playlist_info)
            
            return playlists
        except Exception as e:
            print(f"Playlist search error: {e}")
            return []
    
    def get_artist_id(self, artist_name: str, access_token: str) -> Optional[str]:
        """Get artist ID by name"""
        url = f"{self.base_url}/search"
        headers = {"Authorization": f"Bearer {access_token}"}
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
            
        except Exception as e:
            print(f"Artist search error: {e}")
        
        return None
    
    def get_artist_top_tracks(self, artist_id: str, access_token: str, limit: int = 5) -> List[Dict]:
        """Get artist's top tracks from Spotify"""
        url = f"{self.base_url}/artists/{artist_id}/top-tracks"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"country": "US", "limit": limit}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            tracks = data.get("tracks", [])
            
            # Ensure tracks is a list of dictionaries
            if not isinstance(tracks, list):
                print(f"Warning: tracks is not a list for artist {artist_id}")
                return []
            
            # Filter out any non-dictionary items
            valid_tracks = []
            for track in tracks:
                if isinstance(track, dict):
                    valid_tracks.append(track)
                else:
                    print(f"Warning: track is not a dictionary: {track}")
            
            return valid_tracks
        except Exception as e:
            print(f"Error getting artist top tracks: {e}")
            return []
    
    def get_audio_features(self, track_ids: List[str], access_token: str) -> List[Dict]:
        """Get audio features for multiple tracks (batch processing) with better error handling"""
        if not track_ids:
            return []
        
        # Note: Audio features endpoint is deprecated but still functional
        # We'll use it with proper error handling
        url = f"{self.base_url}/audio-features"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"ids": ",".join(track_ids)}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            # Check specific error codes
            if response.status_code == 403:
                print(f"Audio features access denied (403) - token may not have required scopes")
                print(f"Required scopes: user-read-private, user-read-currently-playing")
                return []
            elif response.status_code == 401:
                print(f"Audio features access unauthorized (401) - token may be expired")
                return []
            elif response.status_code == 429:
                print(f"Audio features rate limited (429) - too many requests")
                return []
            
            response.raise_for_status()
            audio_features = response.json().get("audio_features", [])
            
            # Filter out None values (tracks without audio features)
            valid_features = [features for features in audio_features if features is not None]
            
            if len(valid_features) != len(track_ids):
                print(f"Got {len(valid_features)} valid audio features out of {len(track_ids)} tracks")
            
            # Log successful audio features retrieval
            if valid_features:
                print(f"Successfully retrieved audio features for {len(valid_features)} tracks")
                # Log sample audio features for debugging
                if valid_features:
                    sample = valid_features[0]
                    print(f"Sample audio features - Danceability: {sample.get('danceability', 'N/A')}, "
                          f"Energy: {sample.get('energy', 'N/A')}, Valence: {sample.get('valence', 'N/A')}")
            
            return valid_features
            
        except requests.exceptions.RequestException as e:
            print(f"Network error getting audio features: {e}")
            return []
        except Exception as e:
            print(f"Error getting audio features: {e}")
            return []
    
    def get_spotify_artist_genres(self, artist_name: str, access_token: str) -> List[str]:
        """Get artist genres from Spotify"""
        try:
            # First search for the artist
            search_url = f"{self.base_url}/search"
            headers = {"Authorization": f"Bearer {access_token}"}
            params = {
                "q": artist_name,
                "type": "artist",
                "limit": 1
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            
            artists = response.json().get("artists", {}).get("items", [])
            if artists:
                return artists[0].get("genres", [])
            
            return []
        except Exception as e:
            print(f"Error getting artist genres for {artist_name}: {e}")
            return []
    
    def analyze_track_emotional_context(self, features: Dict, track_name: str, artist_name: str) -> str:
        """Analyze emotional context from audio features"""
        try:
            # Extract key audio features
            valence = features.get("valence", 0.5)  # Happiness (0-1)
            energy = features.get("energy", 0.5)    # Energy level (0-1)
            danceability = features.get("danceability", 0.5)  # Danceability (0-1)
            tempo = features.get("tempo", 120)      # BPM
            
            # Determine emotional context based on features
            if valence > 0.7 and energy > 0.7:
                return "happy_energetic"
            elif valence > 0.6 and danceability > 0.7:
                return "happy_danceable"
            elif valence < 0.4 and energy < 0.4:
                return "sad_melancholic"
            elif energy > 0.8 and tempo > 140:
                return "energetic_fast"
            elif valence > 0.6 and energy < 0.5:
                return "happy_calm"
            elif danceability > 0.8:
                return "danceable"
            elif valence < 0.3:
                return "sad_emotional"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"Error analyzing emotional context: {e}")
            return "neutral"
    
    def analyze_track_emotional_context_fallback(self, track_name: str, artist_name: str) -> str:
        """Enhanced fallback emotional context analysis when audio features are not available"""
        try:
            track_lower = track_name.lower()
            artist_lower = artist_name.lower()
            
            # Enhanced keywords for different emotional contexts
            happy_keywords = ['happy', 'joy', 'smile', 'sunshine', 'bright', 'cheerful', 'upbeat', 'positive', 'good', 'great', 'amazing', 'wonderful', 'fantastic', 'awesome']
            sad_keywords = ['sad', 'cry', 'tears', 'lonely', 'heartbreak', 'pain', 'sorrow', 'melancholy', 'blue', 'hurt', 'broken', 'missing', 'gone', 'lost', 'alone']
            romantic_keywords = ['love', 'romance', 'heart', 'kiss', 'forever', 'together', 'soulmate', 'passion', 'beautiful', 'darling', 'sweet', 'honey', 'baby', 'dear']
            energetic_keywords = ['energy', 'power', 'strong', 'fire', 'wild', 'crazy', 'intense', 'explosive', 'dance', 'party', 'rock', 'beat', 'rhythm', 'bass']
            calm_keywords = ['calm', 'peace', 'quiet', 'gentle', 'soft', 'smooth', 'relax', 'serene', 'easy', 'slow', 'gentle', 'tender', 'mellow']
            nostalgic_keywords = ['remember', 'memory', 'yesterday', 'old', 'past', 'childhood', 'nostalgic', 'throwback', 'vintage', 'classic']
            
            # Hindi/Urdu emotional keywords
            hindi_happy = ['khushi', 'masti', 'rang', 'nacho', 'gao', 'maza', 'fun']
            hindi_sad = ['dukh', 'dard', 'aansu', 'tanha', 'udaas', 'gham', 'yaad']
            hindi_romantic = ['pyaar', 'mohabbat', 'dil', 'jaan', 'sajan', 'prem', 'ishq']
            
            # Check track name for emotional keywords (English)
            for keyword in happy_keywords:
                if keyword in track_lower:
                    return "happy_energetic"
            
            for keyword in sad_keywords:
                if keyword in track_lower:
                    return "sad_melancholic"
            
            for keyword in romantic_keywords:
                if keyword in track_lower:
                    return "romantic"
            
            for keyword in energetic_keywords:
                if keyword in track_lower:
                    return "energetic_fast"
            
            for keyword in calm_keywords:
                if keyword in track_lower:
                    return "happy_calm"
            
            for keyword in nostalgic_keywords:
                if keyword in track_lower:
                    return "nostalgic"
            
            # Check for Hindi/Urdu keywords
            for keyword in hindi_happy:
                if keyword in track_lower:
                    return "happy_energetic"
            
            for keyword in hindi_sad:
                if keyword in track_lower:
                    return "sad_melancholic"
            
            for keyword in hindi_romantic:
                if keyword in track_lower:
                    return "romantic"
            
            # Check artist name for genre hints
            if any(genre in artist_lower for genre in ['rock', 'metal', 'punk', 'hardcore']):
                return "energetic_fast"
            elif any(genre in artist_lower for genre in ['jazz', 'classical', 'ambient', 'lofi']):
                return "happy_calm"
            elif any(genre in artist_lower for genre in ['pop', 'dance', 'electronic', 'edm', 'house']):
                return "happy_danceable"
            elif any(genre in artist_lower for genre in ['rap', 'hip', 'trap']):
                return "energetic_fast"
            elif any(genre in artist_lower for genre in ['country', 'folk', 'acoustic']):
                return "happy_calm"
            elif any(genre in artist_lower for genre in ['bollywood', 'indian', 'hindi', 'punjabi']):
                return "romantic"  # Default for Indian music
            
            return "neutral"
            
        except Exception as e:
            print(f"Error in fallback emotional context analysis: {e}")
            return "neutral"
    
    def get_artist_genre_fallback(self, artist_name: str) -> str:
        """Fallback genre detection when Spotify genres are not available"""
        try:
            artist_lower = artist_name.lower()
            
            # Enhanced artist genre mappings
            genre_mappings = {
                'pop': ['taylor swift', 'ed sheeran', 'ariana grande', 'justin bieber', 'dua lipa', 'harry styles', 'billie eilish', 'olivia rodrigo'],
                'rock': ['coldplay', 'imagine dragons', 'twenty one pilots', 'maroon 5', 'one republic', 'fall out boy', 'panic at the disco'],
                'hip_hop': ['post malone', 'travis scott', 'drake', 'kendrick lamar', 'j cole', 'eminem', 'snoop dogg'],
                'electronic': ['martin garrix', 'the chainsmokers', 'calvin harris', 'david guetta', 'marshmello', 'skrillex', 'deadmau5'],
                'indie': ['arctic monkeys', 'the 1975', 'vampire weekend', 'arcade fire', 'alt-j', 'glass animals'],
                'country': ['luke combs', 'morgan wallen', 'kane brown', 'blake shelton', 'carrie underwood'],
                'r&b': ['the weeknd', 'bruno mars', 'john legend', 'sam smith', 'sza', 'h.e.r.'],
                'bollywood': ['pritam', 'a.r. rahman', 'atif aslam', 'arijit singh', 'neha kakkar', 'shreya ghoshal', 'sunidhi chauhan'],
                'hindi_pop': ['badshah', 'harrdy sandhu', 'tony kakkar', 'vishal mishra', 'jubin nautiyal', 'palak muchhal'],
                'punjabi': ['diljit dosanjh', 'guru randhawa', 'ammy virk', 'jassie gill', 'gurdas maan', 'babbu maan'],
                'tamil_pop': ['yuvan shankar raja', 'harris jayaraj', 'g.v. prakash', 'anirudh ravichander'],
                'bhangra': ['gurdas maan', 'babbu maan', 'jassi sidhu', 'malkit singh']
            }
            
            # Check for exact matches
            for genre, artists in genre_mappings.items():
                if artist_lower in artists:
                    return genre
            
            # Check for partial matches
            for genre, artists in genre_mappings.items():
                for artist in artists:
                    if artist in artist_lower or artist_lower in artist:
                        return genre
            
            # Check for genre keywords in artist name
            if any(word in artist_lower for word in ['rock', 'metal', 'punk']):
                return 'rock'
            elif any(word in artist_lower for word in ['pop', 'mainstream']):
                return 'pop'
            elif any(word in artist_lower for word in ['rap', 'hip', 'trap']):
                return 'hip_hop'
            elif any(word in artist_lower for word in ['dj', 'electronic', 'edm']):
                return 'electronic'
            elif any(word in artist_lower for word in ['jazz', 'blues']):
                return 'jazz'
            elif any(word in artist_lower for word in ['classical', 'orchestra']):
                return 'classical'
            elif any(word in artist_lower for word in ['country', 'folk']):
                return 'country'
            
            return "unknown"
            
        except Exception as e:
            print(f"Error in fallback genre detection: {e}")
            return "unknown"
    
    def get_enhanced_user_preferences(self, access_token: str, context_type: str, language_preference: Dict, mood_preference: Dict) -> Dict:
        """Get enhanced user preferences for personalization"""
        try:
            # Get user's top artists and tracks
            top_artists = self.get_top_artists_with_genres(access_token, 10, "long_term")
            top_tracks = self.get_top_tracks_detailed(access_token, 10, "long_term")
            
            # Extract favorite artists and genres
            favorite_artists = [artist.get("name", "") for artist in top_artists]
            favorite_genres = []
            for artist in top_artists:
                favorite_genres.extend(artist.get("genres", []))
            
            # Remove duplicates
            favorite_genres = list(set(favorite_genres))
            
            return {
                "favorite_artists": favorite_artists,
                "favorite_tracks": [track.get("name", "") for track in top_tracks],
                "favorite_genres": favorite_genres,
                "context_type": context_type,
                "language_preference": language_preference,
                "mood_preference": mood_preference
            }
            
        except Exception as e:
            print(f"Error getting enhanced user preferences: {e}")
            return {
                "favorite_artists": [],
                "favorite_tracks": [],
                "favorite_genres": [],
                "context_type": context_type,
                "language_preference": language_preference,
                "mood_preference": mood_preference
            }
    
    def get_context_fallback_artists(self, context_type: str, language_preference: Dict) -> List[str]:
        """Get context-appropriate fallback artists"""
        primary_language = language_preference.get('primary_language', 'any')
        
        # Context-based fallback artists
        fallback_artists = {
            "upbeat": {
                "english": ["The Weeknd", "Ed Sheeran", "Taylor Swift", "Justin Bieber", "Ariana Grande"],
                "hindi": ["Arijit Singh", "Neha Kakkar", "Badshah", "Harrdy Sandhu", "Shreya Ghoshal"],
                "any": ["The Weeknd", "Ed Sheeran", "Arijit Singh", "Taylor Swift", "Neha Kakkar"]
            },
            "sad": {
                "english": ["Adele", "Sam Smith", "Lewis Capaldi", "Billie Eilish", "Lana Del Rey"],
                "hindi": ["Arijit Singh", "Atif Aslam", "Mohit Chauhan", "Sunidhi Chauhan", "Shreya Ghoshal"],
                "any": ["Adele", "Arijit Singh", "Sam Smith", "Atif Aslam", "Billie Eilish"]
            },
            "energetic": {
                "english": ["Martin Garrix", "The Chainsmokers", "Calvin Harris", "David Guetta", "Marshmello"],
                "hindi": ["Badshah", "Harrdy Sandhu", "Neha Kakkar", "Tony Kakkar", "Vishal Mishra"],
                "any": ["Martin Garrix", "Badshah", "The Chainsmokers", "Harrdy Sandhu", "Calvin Harris"]
            },
            "party": {
                "english": ["LMFAO", "Pitbull", "Flo Rida", "Black Eyed Peas", "Kesha"],
                "hindi": ["Badshah", "Harrdy Sandhu", "Neha Kakkar", "Tony Kakkar", "Vishal Mishra"],
                "any": ["LMFAO", "Badshah", "Pitbull", "Harrdy Sandhu", "Flo Rida"]
            }
        }
        
        context_artists = fallback_artists.get(context_type, fallback_artists["upbeat"])
        return context_artists.get(primary_language, context_artists["any"])
    
    def get_trending_tracks_for_context(self, context_type: str, access_token: str, limit: int = 15) -> List[Dict]:
        """Get trending tracks for specific context"""
        try:
            # Use Spotify's featured playlists to get trending tracks
            url = f"{self.base_url}/browse/featured-playlists"
            headers = {"Authorization": f"Bearer {access_token}"}
            params = {"limit": 5}
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            
            playlists = response.json().get("playlists", {}).get("items", [])
            trending_tracks = []
            
            for playlist in playlists[:3]:  # Use top 3 playlists
                playlist_id = playlist.get("id")
                if playlist_id:
                    # Get playlist tracks
                    tracks_url = f"{self.base_url}/playlists/{playlist_id}/tracks"
                    tracks_response = requests.get(tracks_url, headers=headers, params={"limit": 10}, timeout=5)
                    
                    if tracks_response.status_code == 200:
                        tracks_data = tracks_response.json()
                        for item in tracks_data.get("items", []):
                            track = item.get("track")
                            if track and len(trending_tracks) < limit:
                                track_obj = {
                                    "name": track.get("name", "Unknown Track"),
                                    "artist": track.get("artists", [{}])[0].get("name", "Unknown Artist"),
                                    "album_name": track.get("album", {}).get("name", "Unknown Album"),
                                    "release_year": track.get("album", {}).get("release_date", "Unknown")[:4] if track.get("album", {}).get("release_date") else "Unknown",
                                    "album_art_url": track.get("album", {}).get("images", [{}])[0].get("url") if track.get("album", {}).get("images") else "/placeholder.svg",
                                    "preview_url": track.get("preview_url"),
                                    "url": track.get("external_urls", {}).get("spotify", "#"),
                                    "context_score": 0.8
                                }
                                trending_tracks.append(track_obj)
            
            return trending_tracks
            
        except Exception as e:
            print(f"Error getting trending tracks: {e}")
            return []
    
    def get_hardcoded_fallback_tracks(self, context_type: str) -> List[Dict]:
        """Get hardcoded fallback tracks for specific context"""
        fallback_tracks = {
            "sad": [
                {
                    "name": "Someone Like You",
                    "artist": "Adele",
                    "album_name": "21",
                    "release_year": "2011",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273c8c4bb14fd21f320a8e2f8fd",
                    "preview_url": None,
                    "url": "https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV",
                    "context_score": 1.5
                },
                {
                    "name": "All of Me",
                    "artist": "John Legend",
                    "album_name": "Love in the Future",
                    "release_year": "2013",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                    "preview_url": None,
                    "url": "https://open.spotify.com/track/3U4isOIWM3VvDubwSI3y7a",
                    "context_score": 1.3
                }
            ],
            "upbeat": [
                {
                    "name": "Party Rock Anthem",
                    "artist": "LMFAO",
                    "album_name": "Sorry for Party Rocking",
                    "release_year": "2011",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273c8c4bb14fd21f320a8e2f8fd",
                    "preview_url": None,
                    "url": "https://open.spotify.com/track/0IkKz2J93C94Ei4BvDop7P",
                    "context_score": 1.0
                },
                {
                    "name": "Uptown Funk",
                    "artist": "Mark Ronson ft. Bruno Mars",
                    "album_name": "Uptown Special",
                    "release_year": "2014",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                    "preview_url": None,
                    "url": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS",
                    "context_score": 1.0
                }
            ]
        }
        
        return fallback_tracks.get(context_type, fallback_tracks["upbeat"])
    
    def check_audio_features_access(self, access_token: str) -> bool:
        """Check if the access token has permissions for audio features"""
        try:
            # Try to get audio features for a known track ID to test access
            test_track_id = "11dFghVXANMlKmJXsNCbNl"  # Example track ID from Spotify docs
            url = f"{self.base_url}/audio-features"
            headers = {"Authorization": f"Bearer {access_token}"}
            params = {"ids": test_track_id}
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                print("✅ Audio features access confirmed - token has required scopes")
                return True
            elif response.status_code == 403:
                print("❌ Audio features access denied - token missing required scopes")
                print("Required scopes: user-read-private, user-read-currently-playing")
                return False
            elif response.status_code == 401:
                print("❌ Audio features access unauthorized - token may be expired")
                return False
            else:
                print(f"⚠️ Audio features access test returned status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error checking audio features access: {e}")
            return False
    
    def get_token_scopes(self, access_token: str) -> List[str]:
        """Get the scopes associated with the current access token"""
        try:
            # Get user profile to check scopes (if available)
            url = f"{self.base_url}/me"
            headers = {"Authorization": f"Bearer {access_token}"}
            
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                user_data = response.json()
                # Note: Spotify doesn't return scopes in user profile
                # We'll infer from what endpoints work
                return self._infer_token_scopes(access_token)
            else:
                print(f"Could not get user profile: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error getting token scopes: {e}")
            return []
    
    def _infer_token_scopes(self, access_token: str) -> List[str]:
        """Infer token scopes by testing different endpoints"""
        scopes = []
        
        # Test different endpoints to infer scopes
        endpoints_to_test = {
            "user-read-private": f"{self.base_url}/me",
            "user-top-read": f"{self.base_url}/me/top/artists?limit=1",
            "user-read-recently-played": f"{self.base_url}/me/player/recently-played?limit=1",
            "user-read-currently-playing": f"{self.base_url}/me/player/currently-playing",
            "playlist-modify-public": f"{self.base_url}/me/playlists?limit=1"
        }
        
        for scope, endpoint in endpoints_to_test.items():
            try:
                headers = {"Authorization": f"Bearer {access_token}"}
                response = requests.get(endpoint, headers=headers, timeout=3)
                
                if response.status_code in [200, 204]:
                    scopes.append(scope)
                    print(f"✅ Scope confirmed: {scope}")
                else:
                    print(f"❌ Scope missing: {scope} (status: {response.status_code})")
                    
            except Exception as e:
                print(f"Error testing scope {scope}: {e}")
        
        return scopes

    def get_fallback_recommendations(self, context_type: str) -> List[Dict]:
        """Get fallback recommendations for when API calls fail"""
        fallback_data = {
            "party": [
                {"name": "Party Rock Anthem", "artist": "LMFAO", "album": "Sorry for Party Rocking", "year": 2011, "spotify_url": "https://open.spotify.com/track/0IkKz2J93C94Ei4BvDop7P", "affinity_score": 0.9},
                {"name": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "album": "Uptown Special", "year": 2014, "spotify_url": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS", "affinity_score": 0.9},
                {"name": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours", "year": 2020, "spotify_url": "https://open.spotify.com/track/0V3wPSX9ygBnCm8psKIegu", "affinity_score": 0.8},
                {"name": "Bad Guy", "artist": "Billie Eilish", "album": "WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?", "year": 2019, "spotify_url": "https://open.spotify.com/track/5QO79kh1waicV47BqGRL3g", "affinity_score": 0.8},
                {"name": "Stressed Out", "artist": "Twenty One Pilots", "album": "Blurryface", "year": 2015, "spotify_url": "https://open.spotify.com/track/3CRDbSIZ4r5MsZ0YwxuEkn", "affinity_score": 0.7}
            ],
            "sad": [
                {"name": "Someone Like You", "artist": "Adele", "album": "21", "year": 2011, "spotify_url": "https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV", "affinity_score": 0.9},
                {"name": "All of Me", "artist": "John Legend", "album": "Love in the Future", "year": 2013, "spotify_url": "https://open.spotify.com/track/3U4isOIWM3VvDubwSI3y7a", "affinity_score": 0.8},
                {"name": "Say You Won't Let Go", "artist": "James Arthur", "album": "Back from the Edge", "year": 2016, "spotify_url": "https://open.spotify.com/track/5uCax9HTNlzGybIStD3vDh", "affinity_score": 0.8},
                {"name": "Perfect", "artist": "Ed Sheeran", "album": "÷ (Divide)", "year": 2017, "spotify_url": "https://open.spotify.com/track/0tgVpDi06FyKpA1z0VMD4v", "affinity_score": 0.7},
                {"name": "Photograph", "artist": "Ed Sheeran", "album": "x", "year": 2014, "spotify_url": "https://open.spotify.com/track/6fxVffaTuwjgEk5h9QyRjy", "affinity_score": 0.7}
            ]
        }
        
        return fallback_data.get(context_type, fallback_data["party"])
    
    def _generate_state(self) -> str:
        """Generate random state parameter"""
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32))
    
    def _build_query_string(self, params: Dict) -> str:
        """Build query string from parameters"""
        return '&'.join([f"{k}={v}" for k, v in params.items()]) 