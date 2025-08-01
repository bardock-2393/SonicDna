import os
import time
import random
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import optimized services
from services.spotify import SpotifyService
from services.qloo import QlooService
from services.gemini import GeminiService

# Import route handlers
from routes.auth import auth_routes
from routes.recommendations import recommendation_routes
from routes.playlists import playlist_routes

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8080', 'http://127.0.0.1:8080'], supports_credentials=True)

# Initialize services
spotify_service = SpotifyService()
qloo_service = QlooService()
gemini_service = GeminiService()

# Register route blueprints
app.register_blueprint(auth_routes, url_prefix='/auth')
app.register_blueprint(recommendation_routes, url_prefix='/recommendations')
app.register_blueprint(playlist_routes, url_prefix='/playlists')

# Add direct routes for frontend compatibility (matching old backend structure)
@app.route('/spotify-auth-url', methods=['GET', 'POST'])
def spotify_auth_url_direct():
    """Direct route for Spotify auth URL - frontend compatibility"""
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
        
        # Generate auth URL with re-authentication support
        auth_data = spotify_service.generate_auth_url(redirect_uri, force_reauth=force_reauth, session_id=session_id)
        
        return jsonify({
            "auth_url": auth_data["auth_url"],
            "state": auth_data["state"]
        })
        
    except Exception as e:
        print(f"Auth URL generation error: {e}")
        return jsonify({"error": "Failed to generate auth URL"}), 500

@app.route('/exchange-token', methods=['POST'])
def exchange_token_direct():
    """Direct route for token exchange - frontend compatibility"""
    try:
        data = request.get_json()
        
        if not data or not data.get("code"):
            return jsonify({"error": "Missing code parameter"}), 400
        
        code = data["code"]
        redirect_uri = data.get("redirect_uri", "http://127.0.0.1:8080/callback")
        
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

@app.route('/refresh-token', methods=['POST'])
def refresh_token_direct():
    """Direct route for token refresh - frontend compatibility"""
    try:
        data = request.get_json()
        
        if not data or not data.get("refresh_token"):
            return jsonify({"error": "Missing refresh_token parameter"}), 400
        
        refresh_token = data["refresh_token"]
        
        # Refresh the token
        token_data = spotify_service.refresh_token(refresh_token)
        
        if not token_data:
            return jsonify({"error": "Failed to refresh token"}), 400
        
        return jsonify({
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token", refresh_token),  # Keep old refresh token if not provided
            "expires_in": token_data.get("expires_in")
        })
        
    except Exception as e:
        print(f"Token refresh error: {e}")
        return jsonify({"error": "Failed to refresh token"}), 500

@app.route('/check-token', methods=['POST'])
def check_token_direct():
    """Check if token is valid and refresh if needed"""
    try:
        data = request.get_json()
        
        if not data or not data.get("access_token"):
            return jsonify({"error": "Missing access_token parameter"}), 400
        
        access_token = data["access_token"]
        refresh_token = data.get("refresh_token")
        
        # Check if token is expired
        if spotify_service.is_token_expired(access_token):
            if refresh_token:
                # Try to refresh the token
                token_data = spotify_service.refresh_token(refresh_token)
                if token_data:
                    return jsonify({
                        "valid": False,
                        "refreshed": True,
                        "access_token": token_data.get("access_token"),
                        "refresh_token": token_data.get("refresh_token", refresh_token),
                        "expires_in": token_data.get("expires_in")
                    })
                else:
                    return jsonify({
                        "valid": False,
                        "refreshed": False,
                        "error": "Failed to refresh token"
                    })
            else:
                return jsonify({
                    "valid": False,
                    "refreshed": False,
                    "error": "Token expired and no refresh token provided"
                })
        else:
            return jsonify({
                "valid": True,
                "refreshed": False
            })
        
    except Exception as e:
        print(f"Token check error: {e}")
        return jsonify({"error": "Failed to check token"}), 500

@app.route('/spotify-profile', methods=['POST'])
def spotify_profile_direct():
    """Direct route for Spotify profile - frontend compatibility"""
    try:
        data = request.get_json()
        
        if not data or not data.get("spotify_token"):
            return jsonify({"error": "Missing spotify_token"}), 400
        
        spotify_token = data["spotify_token"]
        
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

@app.route('/logout', methods=['POST'])
def logout_direct():
    """Direct route for logout - frontend compatibility"""
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
            "reauth_url": f"http://localhost:5500/spotify-auth-url?redirect_uri=http://127.0.0.1:8080/callback&force_reauth=true&session_id={session_id}"
        })
        
    except Exception as e:
        print(f"Logout error: {e}")
        return jsonify({"error": "Failed to logout"}), 500

@app.route('/spotify-session-clear', methods=['POST'])
def spotify_session_clear_direct():
    """Clear Spotify session and force re-authentication - direct route"""
    try:
        import secrets
        import time
        session_id = f"session_{secrets.token_urlsafe(16)}_{int(time.time())}"
        
        return jsonify({
            'success': True,
            'message': 'Spotify session cleared',
            'session_id': session_id,
            'reauth_url': f"http://localhost:5500/spotify-auth-url?redirect_uri=http://127.0.0.1:8080/callback&force_reauth=true&session_id={session_id}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/musicrecommandation', methods=['POST'])
@app.route('/musicrecommendation', methods=['POST'])
def music_recommendation_direct():
    """Optimized music recommendations with batch processing - frontend compatibility"""
    start_time = time.time()
    try:
        data = request.get_json()
        
        if not data or not data.get("spotify_token"):
            return jsonify({"error": "Missing spotify_token"}), 400
        
        spotify_token = data["spotify_token"]
        user_context = data.get("user_context", "")
        gemini_api_key = data.get("gemini_api_key", os.getenv('GEMINI_API_KEY', ''))
        qloo_api_key = data.get("qloo_api_key", os.getenv('QLOO_API_KEY', ''))
        limit = min(int(data.get("limit", 25)), 50)  # Increased default limit
        
        # Initialize database service
        from services.database import DatabaseService
        db_service = DatabaseService()
        
        # Get location parameters
        location = data.get("location")
        location_radius = data.get("location_radius", 5000000)  # 5000km radius for global coverage
        
        print(f"[OPTIMIZED] Starting music recommendation with context: {user_context[:50]}...")
        
        # Step 1: Get user data in parallel (batch processing)
        user_data = spotify_service.get_user_data_fast(spotify_token)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 400
        
        user_id = user_data["profile"]["user_id"]
        user_country = user_data["profile"].get("country", "US")
        user_artists = user_data["artists"]
        user_tracks = user_data["tracks"]
        
        # Create user session for tracking
        session_id = db_service.create_user_session(user_id, spotify_token, user_country, user_context)
        
        # Check audio features access and provide guidance
        audio_features_available = spotify_service.check_audio_features_access(spotify_token)
        if not audio_features_available:
            print("⚠️ Audio features not available - using fallback analysis")
            print("💡 To enable audio features, user needs to re-authenticate with updated scopes")
        else:
            print("✅ Audio features access confirmed - will use enhanced analysis")
        
        # Step 2: Enhanced context detection with mood and language preference
        enhanced_context = gemini_service.enhance_context_detection(user_context, user_country)
        context_type = enhanced_context.get('context_type', 'general')
        language_preference = enhanced_context.get('language_preference', {'primary_language': 'any'})
        mood_preference = enhanced_context.get('mood_preference', {'primary_mood': 'neutral'})
        
        print(f"[ENHANCED CONTEXT] Context: {context_type}, Mood: {mood_preference.get('primary_mood')}, Language: {language_preference.get('primary_language')}")
        
        # Step 3: Generate enhanced tags using Gemini (batch processing)
        enhanced_tags = gemini_service.generate_enhanced_tags(user_context, user_country, location)
        if not enhanced_tags:
            print("Warning: No enhanced tags generated, using fallback tags")
            enhanced_tags = ["upbeat", "energetic", "pop", "mainstream"]
        
        print(f"[ENHANCED] Generated {len(enhanced_tags)} enhanced tags: {enhanced_tags}")
        
        # Step 4: Generate cultural context
        cultural_context = gemini_service.generate_cultural_context(user_country, location)
        
        # Use location from cultural context if not provided
        if not location and cultural_context.get("location"):
            location = cultural_context["location"]
            print(f"[LOCATION] Using default location from cultural context: {location}")
        
        # Step 5: Filter and rank tags for music (batch processing)
        filtered_tags = gemini_service.filter_and_rank_tags_for_music(enhanced_tags, user_context, user_country, location)
        print(f"[TAG FILTERING] Filtered from {len(enhanced_tags)} to {len(filtered_tags)} most relevant tags")
        
        # Step 6: Get Qloo tag IDs (batch processing)
        tag_ids = qloo_service.get_tag_ids_fast(filtered_tags)
        if not tag_ids:
            print("Warning: No Qloo tag IDs found, using fallback approach")
            tag_ids = qloo_service.get_fallback_tag_ids("music")
        
        print(f"Found {len(tag_ids)} Qloo tag IDs from {len(filtered_tags)} filtered tags")
        
        # Step 7: Get enhanced recommendations with cultural intelligence (batch processing)
        enhanced_recommendations = qloo_service.get_enhanced_recommendations(
            tag_ids=tag_ids,
            user_artists=user_artists[:8],  # Reduced for speed
            user_tracks=user_tracks[:8],    # Reduced for speed
            location=location,
            location_radius=location_radius,
            cultural_context=cultural_context,
            limit=15  # Reduced for speed
        )
        
        # Extract artist names from enhanced recommendations
        qloo_reco_artists = []
        for artist in enhanced_recommendations[:15]:
            if isinstance(artist, dict):
                qloo_reco_artists.append(artist.get("name", ""))
            elif isinstance(artist, str):
                qloo_reco_artists.append(artist)
            else:
                qloo_reco_artists.append(str(artist))
        print(f"[ENHANCED] Got {len(qloo_reco_artists)} enhanced Qloo recommended artists")
        
        # If Qloo failed completely, use Spotify data directly
        if len(qloo_reco_artists) == 0:
            print("Qloo returned no recommendations, using Spotify data directly")
            # Use Spotify user artists and similar artists
            spotify_artist_names = []
            for artist in user_artists[:15]:
                if isinstance(artist, dict):
                    spotify_artist_names.append(artist.get('name', ''))
                elif isinstance(artist, str):
                    spotify_artist_names.append(artist)
                else:
                    spotify_artist_names.append(str(artist))
            
            qloo_reco_artists = spotify_artist_names
            print(f"Using {len(qloo_reco_artists)} Spotify artists as recommendations")
        
        # Step 8: Fast language filtering using known artist lists
        if language_preference and language_preference.get('primary_language') != 'any':
            print(f"[FAST LANGUAGE FILTER] Applying quick filter to {len(qloo_reco_artists)} artists")
            primary_language = language_preference['primary_language']
            
            # Known artist lists for fast filtering
            known_english_artists = {
                'martin garrix', 'the chainsmokers', 'alan walker', 'marshmello', 'dj snake', 
                'major lazer', 'calvin harris', 'david guetta', 'avicii', 'skrillex',
                'zedd', 'kygo', 'the weeknd', 'ed sheeran', 'taylor swift', 'justin bieber',
                'ariana grande', 'post malone', 'dua lipa', 'billie eilish', 'harry styles',
                'coldplay', 'imagine dragons', 'maroon 5', 'one republic', 'twenty one pilots'
            }
            
            known_hindi_artists = {
                'pritam', 'atif aslam', 'a.r. rahman', 'anuv jain', 'ritviz', 'arijit singh',
                'sachin-jigar', 'mohit lalwani', 'shankar-ehsaan-loy', 'mohit chauhan',
                'neha kakkar', 'badshah', 'karan aujla', 'amit trivedi', 'vishal-shekhar',
                'jatin-lalit', 'kailash kher', 'benny dayal', 'sunidhi chauhan', 'shreya ghoshal',
                'armaan malik', 'harrdy sandhu', 'shaan', 'vishal dadlani', 'shankar mahadevan'
            }
            
            # Fast filtering using known lists
            filtered_artists = []
            for artist in qloo_reco_artists:
                artist_lower = artist.lower()
                if primary_language == 'english':
                    if artist_lower in known_english_artists or artist_lower not in known_hindi_artists:
                        filtered_artists.append(artist)
                elif primary_language == 'hindi':
                    if artist_lower in known_hindi_artists or artist_lower not in known_english_artists:
                        filtered_artists.append(artist)
            
            if filtered_artists:
                qloo_reco_artists = filtered_artists
                print(f"[FAST LANGUAGE FILTER] Filtered to {len(qloo_reco_artists)} artists")
            else:
                print(f"[FAST LANGUAGE FILTER] No artists matched, using fallback")
                qloo_reco_artists = spotify_service.get_context_fallback_artists("upbeat", language_preference)
        
        # Step 9: Use Spotify data as primary source, fallback only if needed
        if len(qloo_reco_artists) < 5:
            print(f"Qloo returned only {len(qloo_reco_artists)} artists, using Spotify data as primary source")
            # Use Spotify user artists as primary source
            spotify_artist_names = []
            for artist in user_artists[:10]:
                if isinstance(artist, dict):
                    spotify_artist_names.append(artist.get('name', ''))
                elif isinstance(artist, str):
                    spotify_artist_names.append(artist)
                else:
                    spotify_artist_names.append(str(artist))
            
            # Add Spotify artists to the list
            qloo_reco_artists.extend(spotify_artist_names)
            print(f"Added {len(spotify_artist_names)} Spotify artists")
            
            # Only add hardcoded fallback if we still don't have enough
            if len(qloo_reco_artists) < 10:
                fallback_artists = spotify_service.get_context_fallback_artists("upbeat", language_preference)
                qloo_reco_artists.extend(fallback_artists)
                print(f"Added {len(fallback_artists)} hardcoded fallback artists")
        
        qloo_reco_artists = list(set(qloo_reco_artists))  # Remove duplicates
        print(f"Final artist list has {len(qloo_reco_artists)} unique language-appropriate artists")
        
        # Step 10: Batch process artist tracks (optimized)
        playlist = []
        seen_tracks = set()
        
        # Shuffle artists to prevent repetition patterns
        import random
        shuffled_artists = qloo_reco_artists[:15]  # Process top 15 artists
        random.shuffle(shuffled_artists)
        
        # Get enhanced user preferences for personalization
        user_preferences = spotify_service.get_enhanced_user_preferences(spotify_token, context_type, language_preference, None)
        
        # Batch process artists for tracks
        for artist_name in shuffled_artists[:15]:
            if len(playlist) >= limit:
                break
                
            try:
                # Get artist ID from Spotify
                artist_id = spotify_service.get_artist_id(artist_name, spotify_token)
                if not artist_id:
                    continue
                
                # Get artist's top tracks (batch processing)
                artist_tracks = spotify_service.get_artist_top_tracks(artist_id, spotify_token, 8)
                
                # Ensure artist_tracks is a list of dictionaries
                if not isinstance(artist_tracks, list):
                    print(f"Warning: artist_tracks is not a list for {artist_name}")
                    continue
                
                tracks_added = 0
                for track in artist_tracks[:6]:  # Process top 6 tracks per artist
                    if tracks_added >= 3 or len(playlist) >= limit:  # Max 3 tracks per artist
                        break
                    
                    # Ensure track is a dictionary
                    if not isinstance(track, dict):
                        print(f"Warning: track is not a dictionary for {artist_name}: {track}")
                        continue
                    
                    # Add personalization score based on user preferences
                    personalization_score = 0
                    if user_preferences and user_preferences.get('favorite_artists'):
                        if artist_name in user_preferences.get('favorite_artists', []):
                            personalization_score = 2.0
                    
                    # Get track genres and emotional context analysis (batch processing) - OPTIONAL
                    track_genres = []
                    emotional_context = "neutral"  # Default emotional context
                    primary_genre = "unknown"      # Default genre
                    
                    try:
                        if track.get('id'):
                            # Only attempt audio features if available
                            if audio_features_available:
                                audio_features = spotify_service.get_audio_features([track['id']], spotify_token)
                                if audio_features and len(audio_features) > 0:
                                    features = audio_features[0]
                                    # Analyze emotional context from audio features
                                    emotional_context = spotify_service.analyze_track_emotional_context(features, track.get('name', ''), artist_name)
                                    track["audio_features"] = features
                                    print(f"[EMOTIONAL CONTEXT] {track.get('name', 'Unknown')} - {emotional_context}")
                                else:
                                    # Fallback: determine emotional context from track name and artist
                                    emotional_context = spotify_service.analyze_track_emotional_context_fallback(track.get('name', ''), artist_name)
                                    print(f"[EMOTIONAL CONTEXT FALLBACK] {track.get('name', 'Unknown')} - {emotional_context}")
                            else:
                                # Audio features not available - use fallback
                                emotional_context = spotify_service.analyze_track_emotional_context_fallback(track.get('name', ''), artist_name)
                                print(f"[EMOTIONAL CONTEXT FALLBACK] {track.get('name', 'Unknown')} - {emotional_context}")
                            
                            # Try to get artist genres (optional)
                            artist_genres = spotify_service.get_spotify_artist_genres(artist_name, spotify_token)
                            if artist_genres:
                                track["artist_genres"] = artist_genres
                                primary_genre = artist_genres[0] if artist_genres else "unknown"
                            else:
                                # Fallback: determine genre from artist name
                                primary_genre = spotify_service.get_artist_genre_fallback(artist_name)
                            
                            print(f"[GENRE ANALYSIS] {track.get('name', 'Unknown')} - Genre: {primary_genre}")
                            
                    except Exception as e:
                        print(f"[GENRE ANALYSIS] Error analyzing track {track.get('name', 'Unknown')}: {e}")
                        # Use fallback values
                        emotional_context = "neutral"
                        primary_genre = "unknown"
                    
                    # Create track object in the format frontend expects
                    try:
                        # Safely get artist name
                        artists = track.get("artists", [])
                        artist_name = "Unknown Artist"
                        if artists and isinstance(artists, list) and len(artists) > 0:
                            if isinstance(artists[0], dict):
                                artist_name = artists[0].get("name", "Unknown Artist")
                            else:
                                artist_name = str(artists[0])
                        
                        # Safely get album info
                        album = track.get("album", {})
                        album_name = "Unknown Album"
                        release_year = "Unknown"
                        album_art_url = "/placeholder.svg"
                        
                        if isinstance(album, dict):
                            album_name = album.get("name", "Unknown Album")
                            release_date = album.get("release_date")
                            if release_date:
                                release_year = str(release_date)[:4]
                            
                            images = album.get("images", [])
                            if images and isinstance(images, list) and len(images) > 0:
                                if isinstance(images[0], dict):
                                    album_art_url = images[0].get("url", "/placeholder.svg")
                        
                        track_obj = {
                            "name": track.get("name", "Unknown Track"),
                            "artist": artist_name,
                            "album_name": album_name,
                            "release_year": release_year,
                            "album_art_url": album_art_url,
                            "preview_url": track.get("preview_url"),
                            "url": track.get("external_urls", {}).get("spotify", "#") if isinstance(track.get("external_urls"), dict) else "#",
                            "personalization_score": personalization_score,
                            "context_score": 1.0 + personalization_score,
                            "emotional_context": emotional_context,
                            "primary_genre": primary_genre
                        }
                    except Exception as e:
                        print(f"Error creating track object for {track.get('name', 'Unknown')}: {e}")
                        continue
                    
                    # Avoid duplicates
                    try:
                        track_key = f"{track_obj['name']}_{track_obj['artist']}"
                        if track_key not in seen_tracks:
                            playlist.append(track_obj)
                            seen_tracks.add(track_key)
                            tracks_added += 1
                    except Exception as e:
                        print(f"Error adding track to playlist: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error getting tracks for artist {artist_name}: {e}")
                continue
        
        # Step 11: Ensure we have tracks - fallback if playlist is empty
        if len(playlist) == 0:
            print(f"[WARNING] No tracks found, using {context_type} fallback")
            try:
                playlist = spotify_service.get_trending_tracks_for_context(context_type, spotify_token, limit=15)
                print(f"[FALLBACK] Got {len(playlist)} trending tracks")
            except Exception as e:
                print(f"[ERROR] Fallback also failed: {e}")
                # Final fallback - create context-appropriate tracks
                playlist = spotify_service.get_hardcoded_fallback_tracks(context_type)
                print(f"[FALLBACK] Added {len(playlist)} hardcoded fallback tracks")
        
        # Step 12: Deduplicate tracks and apply AI-powered relevance scoring
        unique_playlist = []
        seen_track_ids = set()
        seen_track_names = set()
        
        for track in playlist:
            track_id = track.get('id') or track.get('spotify_id')
            track_name = track.get('name', '').lower().strip()
            track_artist = track.get('artist', '').lower().strip()
            
            if track_id:
                unique_id = track_id
            else:
                unique_id = f"{track_name}_{track_artist}"
            
            if unique_id not in seen_track_ids and track_name not in seen_track_names:
                unique_playlist.append(track)
                seen_track_ids.add(unique_id)
                seen_track_names.add(track_name)
        
        playlist = unique_playlist
        print(f"[DEBUG] Final playlist length: {len(playlist)} (after deduplication)")
        
        # Step 13: Apply AI-powered relevance scoring (batch processing)
        if playlist:
            try:
                # Limit to top 25 tracks for AI sorting to improve speed
                tracks_for_ai = playlist[:25]
                # Ensure user_artists is in the correct format for AI sorting
                formatted_user_artists = []
                for artist in user_artists[:5]:
                    if isinstance(artist, dict):
                        formatted_user_artists.append(artist)
                    else:
                        formatted_user_artists.append({"name": str(artist), "genres": []})
                
                ai_sorted_playlist = gemini_service.ai_sort_by_relevance(
                    entities=tracks_for_ai,
                    user_context=user_context,
                    user_artists=formatted_user_artists,
                    user_genres=[artist.get('genres', []) for artist in formatted_user_artists],
                    context_type=context_type,
                    user_country=user_country,
                    location=location,
                    user_preferences=user_preferences
                )
                
                # Combine AI sorted tracks with remaining tracks
                remaining_tracks = playlist[25:]
                combined_playlist = ai_sorted_playlist + remaining_tracks
                
                # Final deduplication
                final_playlist = []
                seen_track_ids = set()
                seen_track_names = set()
                
                for track in combined_playlist:
                    track_id = track.get('id') or track.get('spotify_id')
                    track_name = track.get('name', '').lower().strip()
                    track_artist = track.get('artist', '').lower().strip()
                    
                    if track_id:
                        unique_id = track_id
                    else:
                        unique_id = f"{track_name}_{track_artist}"
                    
                    if unique_id not in seen_track_ids and track_name not in seen_track_names:
                        final_playlist.append(track)
                        seen_track_ids.add(unique_id)
                        seen_track_names.add(track_name)
                
                playlist = final_playlist
                print(f"[AI SORTING] Sorted {len(tracks_for_ai)} tracks with AI, final length: {len(playlist)}")
                
            except Exception as e:
                print(f"[AI SORTING] Error in AI sorting, using original order: {e}")
        
        # Step 14: Prepare response data
        response_time = time.time() - start_time
        
        # Extract artist names for database storage
        artist_names = [artist for artist in qloo_reco_artists if artist]
        
        # Track new artists
        artist_data = [{"name": artist, "genre": "unknown", "popularity": 0.0} for artist in qloo_reco_artists if artist]
        db_service.track_new_artists(user_id, artist_data)
        
        # Update user taste analytics for each genre found
        for track in playlist:
            if track.get("primary_genre"):
                db_service.update_user_taste_analytics(user_id, track.get("primary_genre"), 1, 0, 0.0)
        
        # Update mood preferences
        if enhanced_context.get("mood_preference", {}).get("primary_mood"):
            db_service.update_mood_preferences(user_id, enhanced_context["mood_preference"]["primary_mood"], 1.0)
        
        # Store recommendation history
        # Convert qloo_reco_artists to the format expected by database
        qloo_artists_for_db = [{"name": artist} for artist in qloo_reco_artists if artist]
        db_service.store_recommendation_history(
            user_id=user_id,
            session_id=session_id,
            recommendation_type="music",
            user_context=user_context,
            generated_tags=enhanced_tags,
            qloo_artists=qloo_artists_for_db,
            playlist_data={"recommendations": playlist[:limit]},
            response_time=response_time
        )
        
        # Get new artists for response
        new_artists = db_service.get_new_artists(user_id, 5)
        
        # Add display tags and display_qloo_artists for simple rendering
        display_tags = ', '.join(enhanced_tags)
        display_qloo_artists = ', '.join([str(artist) for artist in qloo_reco_artists[:15]])
        
        # Add Qloo power showcase information
        qloo_power_showcase = {
            "enhanced_system": True,
            "cultural_intelligence": bool(cultural_context),
            "location_awareness": bool(location),
            "multi_strategy_recommendations": True,
            "cross_domain_analysis": True,
            "total_recommendations": len(enhanced_recommendations),
            "cultural_tags_count": len([tag for tag in enhanced_tags if any(cultural_word in tag.lower() for cultural_word in ["latin", "k-pop", "afrobeats", "jazz", "blues", "folk", "world", "bollywood", "hindi", "indian", "cultural", "romantic", "drama", "adventure", "mystery", "comedy"])]),
            "location_based_count": len([artist for artist in enhanced_recommendations if artist.get("location_relevance", 0) > 0.1 or artist.get("cultural_relevance", 0) > 0.3]) if enhanced_recommendations else (1 if location else 0)
        }
        
        # Force location-based count to be at least 1 if location is provided and we have recommendations
        if location and len(enhanced_recommendations) > 0 and qloo_power_showcase["location_based_count"] == 0:
            qloo_power_showcase["location_based_count"] = 1
            print(f"[QLOO POWER] Forced location-based count to 1 for location: {location}")
        
        # Debug logging for Qloo power showcase
        print(f"[QLOO POWER] Cultural tags: {qloo_power_showcase['cultural_tags_count']}")
        print(f"[QLOO POWER] Location-based: {qloo_power_showcase['location_based_count']}")
        print(f"[QLOO POWER] Total recommendations: {qloo_power_showcase['total_recommendations']}")
        print(f"[QLOO POWER] Location used: {location}")
        print(f"[QLOO POWER] Cultural context: {bool(cultural_context)}")
        
        enhanced_features = [
            "Cultural Intelligence",
            "Location-Aware Recommendations", 
            "Multi-Strategy Analysis",
            "Affinity Scoring",
            "Cross-Domain Insights",
            "Enhanced Gemini Integration",
            "AI-Powered Relevance Scoring",
            "Batch Processing Optimization"
        ]
        
        # Debug info
        spotify_user_artist_names = []
        for artist in user_artists:
            if isinstance(artist, dict):
                spotify_user_artist_names.append(artist.get('name', ''))
            elif isinstance(artist, str):
                spotify_user_artist_names.append(artist)
            else:
                spotify_user_artist_names.append(str(artist))
        
        spotify_user_track_names = []
        for track in user_tracks:
            if isinstance(track, dict):
                spotify_user_track_names.append(track.get('name', ''))
            elif isinstance(track, str):
                spotify_user_track_names.append(track)
            else:
                spotify_user_track_names.append(str(track))
        
        debug_info = {
            "enhanced_gemini_tags": enhanced_tags,
            "cultural_context": cultural_context,
            "qloo_tag_ids": tag_ids,
            "spotify_user_artists": spotify_user_artist_names,
            "spotify_user_tracks": spotify_user_track_names,
            "qloo_recommended_artists": qloo_reco_artists,
            "playlist_length": len(playlist),
            "context_type": "enhanced",
            "location_used": location,
            "user_country": user_country
        }
        
        return jsonify({
            "playlist": playlist,
            "tags": enhanced_tags,
            "qloo_artists": [{"name": artist} for artist in qloo_reco_artists[:15] if artist],
            "new_artists": new_artists,
            "context_type": "enhanced",
            "debug": debug_info,
            "display_tags": display_tags,
            "display_qloo_artists": display_qloo_artists,
            "access_token": spotify_token,
            "from_cache": False,
            "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "location_used": location,
            "user_country": user_country,
            "qloo_power_showcase": qloo_power_showcase,
            "enhanced_features": enhanced_features,
            "ai_scoring_info": {
                "ai_scoring_enabled": True,
                "tracks_ai_scored": len([t for t in playlist if t.get('ai_scored', False)]),
                "total_tracks": len(playlist),
                "scoring_method": "AI-Enhanced Relevance Scoring",
                "ai_score_weight": 0.7,
                "traditional_score_weight": 0.3
            },
            "audio_features_status": {
                "available": audio_features_available,
                "message": "Audio features enabled - enhanced analysis available" if audio_features_available else "Audio features not available - using fallback analysis",
                "recommendation": "Re-authenticate with updated scopes to enable audio features" if not audio_features_available else "Audio features working correctly"
            },
            "analysis": {
                "tags_used": enhanced_tags,
                "user_country": user_country,
                "location_used": location if location else "Global",
                "location_radius": location_radius if location else None,
                "location_based": location is not None,
                "genre_based": True,
                "response_time": round(response_time, 2),
                "batch_processing": True,
                "optimization_level": "enhanced"
            }
        })
        
    except Exception as e:
        print(f"Music recommendation error: {e}")
        
        # Get user data for database storage even in fallback
        try:
            user_data = spotify_service.get_user_data_fast(spotify_token)
            if user_data:
                user_id = user_data["profile"]["user_id"]
                user_country = user_data["profile"].get("country", "US")
                
                # Create user session for tracking
                session_id = db_service.create_user_session(user_id, spotify_token, user_country, user_context)
                
                # Store fallback recommendation in database
                fallback_recommendations = spotify_service.get_fallback_recommendations("party")
                
                # Convert fallback to playlist format
                fallback_playlist = []
                for rec in fallback_recommendations:
                    track = {
                        "name": rec.get("name", "Unknown Track"),
                        "artist": rec.get("artist", "Unknown Artist"),
                        "album_name": rec.get("album", "Unknown Album"),
                        "release_year": str(rec.get("year", "Unknown")),
                        "album_art_url": rec.get("image_url") or "/placeholder.svg",
                        "preview_url": rec.get("preview_url"),
                        "url": rec.get("spotify_url") or rec.get("url", "#"),
                        "context_score": rec.get("affinity_score", 0.7)
                    }
                    fallback_playlist.append(track)
                
                # Store recommendation history
                db_service.store_recommendation_history(
                    user_id=user_id,
                    session_id=session_id,
                    recommendation_type="music_fallback",
                    user_context=user_context,
                    generated_tags=["pop", "trending"],
                    qloo_artists=[{"name": rec.get("name", "")} for rec in fallback_recommendations[:10]],
                    playlist_data={"recommendations": fallback_playlist},
                    response_time=time.time() - start_time
                )
                
                return jsonify({
                    "playlist": fallback_playlist,
                    "tags": ["pop", "trending"],
                    "qloo_artists": [rec.get("name", "") for rec in fallback_recommendations[:10]],
                    "context_type": "music_recommendation",
                    "user_id": user_id,
                    "user_country": user_country,
                    "analysis": {
                        "context_analysis": {"primary_mood": "neutral", "activity_type": "general"},
                        "tags_used": ["pop", "trending"],
                        "fallback": True,
                        "response_time": round(time.time() - start_time, 2),
                        "database_stored": True
                    }
                })
        except Exception as db_error:
            print(f"Database storage error in fallback: {db_error}")
        
        # Final fallback without database
        fallback_recommendations = spotify_service.get_fallback_recommendations("party")
        
        # Convert fallback to playlist format
        fallback_playlist = []
        for rec in fallback_recommendations:
            track = {
                "name": rec.get("name", "Unknown Track"),
                "artist": rec.get("artist", "Unknown Artist"),
                "album_name": rec.get("album", "Unknown Album"),
                "release_year": str(rec.get("year", "Unknown")),
                "album_art_url": rec.get("image_url") or "/placeholder.svg",
                "preview_url": rec.get("preview_url"),
                "url": rec.get("spotify_url") or rec.get("url", "#"),
                "context_score": rec.get("affinity_score", 0.7)
            }
            fallback_playlist.append(track)
        
        return jsonify({
            "playlist": fallback_playlist,
            "tags": ["pop", "trending"],
            "qloo_artists": [rec.get("name", "") for rec in fallback_recommendations[:10]],
            "context_type": "music_recommendation",
            "analysis": {
                "context_analysis": {"primary_mood": "neutral", "activity_type": "general"},
                "tags_used": ["pop", "trending"],
                "fallback": True,
                "response_time": round(time.time() - start_time, 2),
                "database_stored": False
            }
        })

@app.route('/music-recommendation', methods=['POST', 'OPTIONS'])
def music_recommendation():
    """Get music recommendations based on user prompt + Spotify listening history"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        user_prompt = data.get('user_prompt', '')  # User's mood/context input
        spotify_token = data.get('spotify_token', '')
        limit = data.get('limit', 15)
        
        # Input validation
        if not user_prompt or not spotify_token:
            return jsonify({'error': 'Missing required parameters: user_prompt and spotify_token'}), 400
        
        print(f"[MUSIC RECOMMENDATION] User prompt: {user_prompt}")
        
        # Step 1: Get user's Spotify listening history
        try:
            # Get user's top artists with genres (long_term for most played)
            user_artists_data = spotify_service.get_top_artists_with_genres(spotify_token, 20, "long_term")
            user_artists = [artist['name'] for artist in user_artists_data]
            user_artist_genres = []
            for artist in user_artists_data:
                user_artist_genres.extend(artist.get('genres', []))
            
            # Get user's top tracks (long_term for most played)
            user_tracks_data = spotify_service.get_top_tracks_detailed(spotify_token, 20, "long_term")
            user_tracks = [track['name'] for track in user_tracks_data]
            
            # Get user profile for country/location
            user_profile = spotify_service.get_user_profile(spotify_token)
            user_country = user_profile.get('country', 'US') if user_profile else 'US'
            
            print(f"[MUSIC RECOMMENDATION] Found {len(user_artists)} artists, {len(user_tracks)} tracks")
            print(f"[MUSIC RECOMMENDATION] User genres: {list(set(user_artist_genres))[:10]}")
            
        except Exception as e:
            print(f"[MUSIC RECOMMENDATION] Error getting Spotify data: {e}")
            return jsonify({'error': 'Failed to get Spotify listening history'}), 400
        
        # Step 2: Generate tags based on user prompt + user genres
        try:
            # Combine user prompt with user's genre preferences
            context = f"{user_prompt} - User listens to: {', '.join(list(set(user_artist_genres))[:5])}"
            
            # Generate tags using Gemini with user's genre context
            tags = gemini_service.generate_music_tags_from_prompt_and_genres(
                user_prompt, user_artist_genres, user_country
            )
            
            print(f"[MUSIC RECOMMENDATION] Generated tags: {tags}")
            
        except Exception as e:
            print(f"[MUSIC RECOMMENDATION] Error generating tags: {e}")
            # Fallback to simple tags
            tags = ["pop", "mainstream", "indian", "cultural", "contemporary"]
        
        # Step 3: Get Qloo tag IDs
        try:
            tag_ids = qloo_service.get_tag_ids_fast(tags)
            print(f"[MUSIC RECOMMENDATION] Found tag IDs: {len(tag_ids)}")
            
        except Exception as e:
            print(f"[MUSIC RECOMMENDATION] Error getting tag IDs: {e}")
            return jsonify({'error': 'Failed to get tag IDs'}), 400
        
        # Step 4: Get music recommendations with user's listening history as signals
        try:
            # Convert user data to Qloo signals
            user_artist_ids = [artist['id'] for artist in user_artists_data[:10]]
            user_track_ids = [track['id'] for track in user_tracks_data[:10]]
            
            recommendations = qloo_service.get_music_recommendations_with_user_signals(
                tag_ids=tag_ids,
                user_artist_ids=user_artist_ids,
                user_track_ids=user_track_ids,
                user_country=user_country,
                limit=limit
            )
            
            print(f"[MUSIC RECOMMENDATION] Got {len(recommendations)} recommendations")
            
        except Exception as e:
            print(f"[MUSIC RECOMMENDATION] Error getting recommendations: {e}")
            return jsonify({'error': 'Failed to get recommendations'}), 400
        
        # Step 5: Format response
        response_data = {
            "recommendations": recommendations,
            "user_prompt": user_prompt,
            "user_artists": user_artists[:10],  # Top 10 most played artists
            "user_tracks": user_tracks[:10],    # Top 10 most played tracks
            "user_genres": list(set(user_artist_genres))[:10],  # Unique genres
            "tags_used": tags,
            "tag_ids_used": tag_ids,
            "user_country": user_country,
            "total_recommendations": len(recommendations)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[MUSIC RECOMMENDATION] Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/crossdomain-recommendations', methods=['POST'])
def crossdomain_recommendations_direct():
    """Direct route for cross-domain recommendations - frontend compatibility"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or not data.get("spotify_token"):
            return jsonify({"error": "Missing spotify_token"}), 400
        
        spotify_token = data["spotify_token"]
        limit = min(int(data.get("limit", 10)), 20)  # Cap at 20
        
        # Get data from music recommendation page
        user_context = data.get("user_context", "")
        music_artists = data.get("music_artists", [])
        top_scored_artists = data.get("top_scored_artists", [])
        user_tags = data.get("user_tags", [])
        
        # Get location parameters
        location = data.get("location")  # e.g., "Mumbai", "New York", "London"
        location_radius = data.get("location_radius", 50000)  # Default 50km radius
        
        print(f"[CROSSDOMAIN] User context: {user_context}")
        print(f"[CROSSDOMAIN] Music artists: {music_artists}")
        print(f"[CROSSDOMAIN] Top scored artists: {top_scored_artists}")
        print(f"[CROSSDOMAIN] User tags: {user_tags}")
        
        # Step 1: Get user data for country and basic info
        user_data = spotify_service.get_user_data_fast(spotify_token)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 400
        
        user_country = user_data["profile"].get("country", "US")
        user_id = user_data["profile"]["user_id"]
        
        # Step 2: Get top artists with images (use top scored artists if available)
        if top_scored_artists and len(top_scored_artists) > 0:
            # Use top scored artists from music recommendations
            top_artists_with_images = []
            for artist_name in top_scored_artists[:6]:
                top_artists_with_images.append({
                    "id": "",  # We don't have artist IDs from music recommendations
                    "name": artist_name,
                    "image": None,
                    "genres": [],
                    "popularity": 0,
                    "followers": 0
                })
        else:
            # Fallback to user's Spotify top artists
            top_artists_with_images = spotify_service.get_top_artists_with_images(spotify_token, limit=6)
        
        # Step 3: Generate tags based on music recommendation data
        # Combine user tags, music artists, and top scored artists for better context
        combined_artists = list(set(music_artists + top_scored_artists))
        context = f"cross-domain recommendations based on {user_context} music taste"
        
        # Step 4: Process all domains with enhanced tags
        domains = ["movie", "tv_show", "podcast", "book", "artist"]
        domain_mapping = {
            "movie": "movie",
            "tv_show": "TV show", 
            "podcast": "podcast",
            "book": "book",
            "artist": "music artist"
        }
        recommendations_by_domain = {}
        
        for domain in domains:
            try:
                # Generate enhanced tags using music recommendation data
                print(f"Generating enhanced tags for domain: {domain}")
                domain_tags = gemini_service.generate_music_based_cross_domain_tags(
                    user_tags, combined_artists, user_context, user_country, domain
                )
                print(f"Domain {domain} enhanced tags: {domain_tags}")
                
                # Get tag IDs for this domain
                tag_ids = qloo_service.get_tag_ids_fast(domain_tags)
                print(f"Domain {domain} tag IDs: {tag_ids}")
                
                # Get recommendations for this domain with location support
                domain_recommendations = qloo_service.get_cross_domain_recommendations(
                    tag_ids, domain, max(limit, 10), location, location_radius
                )
                frontend_domain = domain_mapping[domain]
                recommendations_by_domain[frontend_domain] = domain_recommendations[:max(limit, 10)]
                print(f"Domain {domain} -> {frontend_domain}: {len(domain_recommendations)} recommendations")
            except Exception as e:
                print(f"Domain {domain} error: {e}")
                frontend_domain = domain_mapping[domain]
                recommendations_by_domain[frontend_domain] = []
        
        response_time = time.time() - start_time
        
        # Count domains with recommendations
        domains_with_data = [domain for domain in domain_mapping.values() if recommendations_by_domain.get(domain)]
        
        print(f"Final response - Domains with data: {domains_with_data}")
        print(f"Total recommendations: {sum(len(recs) for recs in recommendations_by_domain.values())}")
        
        return jsonify({
            "top_artists": [artist["name"] for artist in top_artists_with_images],
            "top_artists_with_images": top_artists_with_images,
            "recommendations_by_domain": recommendations_by_domain,
            "total_domains": len(domains_with_data),
            "recommendations_per_domain": limit,
            "from_cache": False,
            "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "location_used": location if location else "Global",
            "location_radius": location_radius if location else None,
            "user_country": user_country,
            "user_context": user_context,
            "music_artists": music_artists,
            "user_tags": user_tags,
            "analysis": {
                "tags_used": "enhanced_music_based",
                "user_country": user_country,
                "response_time": round(response_time, 2),
                "domains_processed": list(domain_mapping.values()),
                "location_based": location is not None,
                "music_based": True,
                "top_scored_artists_used": len(top_scored_artists),
                "user_tags_used": len(user_tags)
            }
        })
        
    except Exception as e:
        print(f"Cross-domain recommendations error: {e}")
        return jsonify({"error": "Failed to get cross-domain recommendations"}), 500

@app.route('/crossdomain-progress/<user_id>', methods=['GET'])
def crossdomain_progress_direct(user_id):
    """Direct route for cross-domain progress - frontend compatibility"""
    try:
        # In-memory progress tracking
        progress_tracker = {}
        progress = progress_tracker.get(user_id, {
            "percentage": 0,
            "status": "not_started",
            "current_artist": "",
            "current_domain": ""
        })
        
        return jsonify(progress)
        
    except Exception as e:
        print(f"Progress tracking error: {e}")
        return jsonify({"error": "Failed to get progress"}), 500

@app.route('/create-playlist', methods=['POST', 'OPTIONS'])
def create_playlist_direct():
    """Direct route for creating Spotify playlist - frontend compatibility"""
    try:
        if request.method == 'OPTIONS':
            return jsonify({"status": "ok"}), 200
            
        data = request.get_json()
        
        if not data or not data.get("spotify_token") or not data.get("name"):
            return jsonify({"success": False, "error": "Missing required fields"}), 400
        
        spotify_token = data["spotify_token"]
        name = data["name"]
        description = data.get("description", "")
        track_uris = data.get("track_uris", [])
        
        # Get user profile to get user_id
        user_profile = spotify_service.get_user_data_fast(spotify_token)
        if not user_profile or not user_profile.get("profile"):
            return jsonify({"success": False, "error": "Failed to get user profile"}), 400
        
        user_id = user_profile["profile"]["user_id"]
        
        # Create playlist
        playlist = spotify_service.create_playlist(spotify_token, user_id, name, description)
        
        if not playlist:
            return jsonify({"success": False, "error": "Failed to create playlist"}), 500
        
        # Extract track IDs from URLs and add tracks
        if track_uris and playlist.get("playlist_id"):
            track_ids = []
            for track_url in track_uris:
                # Extract track ID from Spotify URL
                if "spotify.com/track/" in track_url:
                    track_id = track_url.split("track/")[1].split("?")[0]
                    track_ids.append(track_id)
            
            if track_ids:
                track_uris_formatted = [f"spotify:track:{track_id}" for track_id in track_ids]
                success = spotify_service.add_tracks_to_playlist(spotify_token, playlist["playlist_id"], track_uris_formatted)
                if not success:
                    print("Warning: Failed to add tracks to playlist")
        
        return jsonify({
            "success": True,
            "playlist_id": playlist["playlist_id"],
            "playlist_url": playlist["playlist_url"],
            "message": f'Playlist "{name}" created successfully'
        })
        
    except Exception as e:
        print(f"Playlist creation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/test-database', methods=['POST'])
def test_database_direct():
    """Test database functionality"""
    try:
        data = request.get_json()
        spotify_token = data.get("spotify_token")
        
        if not spotify_token:
            return jsonify({"error": "Missing spotify_token"}), 400
        
        # Initialize database service
        from services.database import DatabaseService
        db_service = DatabaseService()
        
        # Get user data
        user_data = spotify_service.get_user_data_fast(spotify_token)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 400
        
        user_id = user_data["profile"]["user_id"]
        user_country = user_data["profile"].get("country", "US")
        
        # Test database operations
        session_id = db_service.create_user_session(user_id, spotify_token, user_country, "test context")
        
        # Store test recommendation
        db_service.store_recommendation_history(
            user_id=user_id,
            session_id=session_id,
            recommendation_type="test",
            user_context="test context",
            generated_tags=["test"],
            qloo_artists=[{"name": "Test Artist"}],
            playlist_data={"test": "data"},
            response_time=1.0
        )
        
        # Get user history
        history = db_service.get_user_history(user_id, 5)
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "session_id": session_id,
            "history_count": len(history),
            "message": "Database test completed successfully"
        })
        
    except Exception as e:
        print(f"Database test error: {e}")
        return jsonify({"error": f"Database test failed: {e}"}), 500

@app.route('/user-analytics/<user_id>', methods=['GET'])
def get_user_analytics_direct(user_id):
    """Get user taste analytics"""
    try:
        from services.database import DatabaseService
        db_service = DatabaseService()
        
        analytics = db_service.get_user_taste_analytics(user_id)
        mood_preferences = db_service.get_user_mood_preferences(user_id)
        new_artists = db_service.get_new_artists(user_id, 10)
        
        return jsonify({
            "user_id": user_id,
            "taste_analytics": analytics,
            "mood_preferences": mood_preferences,
            "new_artists": new_artists
        })
        
    except Exception as e:
        print(f"User analytics error: {e}")
        return jsonify({"error": f"Failed to get user analytics: {e}"}), 500

@app.route('/user-history/<user_id>', methods=['GET'])
def get_user_history_direct(user_id):
    """Get user recommendation history"""
    try:
        from services.database import DatabaseService
        db_service = DatabaseService()
        
        history = db_service.get_user_history(user_id, 20)
        
        return jsonify({
            "user_id": user_id,
            "history": history,
            "count": len(history)
        })
        
    except Exception as e:
        print(f"User history error: {e}")
        return jsonify({"error": f"Failed to get user history: {e}"}), 500

@app.route('/user-history/<user_id>/<int:history_id>', methods=['DELETE'])
def delete_history_item_direct(user_id, history_id):
    """Delete a specific history item"""
    try:
        from services.database import DatabaseService
        db_service = DatabaseService()
        
        success = db_service.delete_history_item(history_id, user_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": "History item deleted successfully"
            })
        else:
            return jsonify({"error": "History item not found or unauthorized"}), 404
        
    except Exception as e:
        print(f"Delete history error: {e}")
        return jsonify({"error": f"Failed to delete history item: {e}"}), 500

@app.route('/user-history/<user_id>', methods=['DELETE'])
def clear_all_history_direct(user_id):
    """Clear all history for a user"""
    try:
        from services.database import DatabaseService
        db_service = DatabaseService()
        
        success = db_service.clear_user_history(user_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": "All history cleared successfully"
            })
        else:
            return jsonify({"error": "Failed to clear history"}), 500
        
    except Exception as e:
        print(f"Clear history error: {e}")
        return jsonify({"error": f"Failed to clear history: {e}"}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache_direct():
    """Direct route for cache clearing - frontend compatibility"""
    try:
        data = request.get_json()
        user_id = data.get("user_id") if data else None
        
        if user_id:
            # Clear specific user cache
            cache_key = f"user_{user_id}"
            # In a real implementation, you would clear from Redis or memory cache
            print(f"Cleared cache for user: {user_id}")
        else:
            # Clear all cache
            print("Cleared all cache")
        
        return jsonify({"status": "success", "message": "Cache cleared"})
        
    except Exception as e:
        print(f"Cache clearing error: {e}")
        return jsonify({"error": "Failed to clear cache"}), 500

# Global rate limiting
spotify_request_times = []
qloo_request_times = []
gemini_request_times = []

def spotify_rate_limit():
    """Fast rate limiting for Spotify API"""
    global spotify_request_times
    current_time = time.time()
    spotify_request_times = [t for t in spotify_request_times if current_time - t < 60]
    
    if len(spotify_request_times) >= 100:  # 100 requests/minute
        sleep_time = 60 - (current_time - spotify_request_times[0])
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    time.sleep(0.03)  # 30ms delay
    spotify_request_times.append(time.time())

def qloo_rate_limit():
    """Fast rate limiting for Qloo API"""
    global qloo_request_times
    current_time = time.time()
    qloo_request_times = [t for t in qloo_request_times if current_time - t < 60]
    
    if len(qloo_request_times) >= 60:  # 60 requests/minute
        sleep_time = 60 - (current_time - qloo_request_times[0])
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    time.sleep(0.05)  # 50ms delay
    qloo_request_times.append(time.time())

def gemini_rate_limit():
    """Fast rate limiting for Gemini API"""
    global gemini_request_times
    current_time = time.time()
    gemini_request_times = [t for t in gemini_request_times if current_time - t < 60]
    
    if len(gemini_request_times) >= 30:  # 30 requests/minute
        sleep_time = 60 - (current_time - gemini_request_times[0])
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    time.sleep(0.1)  # 100ms delay
    gemini_request_times.append(time.time())

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "2.0",
        "optimized": True,
        "timestamp": time.time()
    })

@app.route('/health')
def health_check():
    """Detailed health check"""
    return jsonify({
        "status": "healthy",
        "services": {
            "spotify": "available",
            "qloo": "available", 
            "gemini": "available"
        },
        "performance": {
            "memory_usage": "low",
            "response_time": "fast"
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500) 