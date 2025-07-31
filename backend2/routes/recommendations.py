from flask import Blueprint, request, jsonify
from services.spotify import SpotifyService
from services.qloo import QlooService
from services.gemini import GeminiService
from services.database import DatabaseService
from utils.helpers import (
    validate_input_data, sanitize_string, rank_recommendations_fast,
    apply_cultural_intelligence_fast, get_fallback_recommendations
)
import time
import hashlib

recommendation_routes = Blueprint('recommendations', __name__)
spotify_service = SpotifyService()
qloo_service = QlooService()
gemini_service = GeminiService()
db_service = DatabaseService()

# In-memory progress tracking
progress_tracker = {}

@recommendation_routes.route('/musicrecommendation', methods=['POST'])
@recommendation_routes.route('/musicrecommandation', methods=['POST'])
def music_recommendation():
    """Optimized music recommendation engine using Qloo + Gemini with database integration"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        user_context = sanitize_string(data.get("user_context", ""))
        limit = min(int(data.get("limit", 15)), 50)  # Cap at 50
        
        # Step 1: Get user data and analyze context in parallel
        user_data = spotify_service.get_user_data_fast(spotify_token)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 400
        
        user_id = user_data["profile"]["user_id"]
        user_country = user_data["profile"].get("country", "US")
        
        # Step 2: Create user session for tracking
        session_id = db_service.create_user_session(user_id, spotify_token, user_country, user_context)
        
        # Step 3: Check cache first
        cache_key = hashlib.md5(f"{user_id}_{user_context}_{user_country}".encode()).hexdigest()
        cached_result = db_service.get_cached_recommendation(cache_key)
        if cached_result:
            print(f"[CACHE HIT] Returning cached result for user {user_id}")
            return jsonify(cached_result)
        
        # Step 3: Analyze context with Gemini
        context_analysis = gemini_service.analyze_context_fast(user_context)
        
        # Step 4: Generate optimized tags
        tags = gemini_service.generate_optimized_tags(user_context, user_country)
        
        # Step 5: Get Qloo recommendations
        tag_ids = qloo_service.get_tag_ids_fast(tags)
        recommendations = qloo_service.get_recommendations_fast(tag_ids, limit)
        
        # Step 6: Apply cultural intelligence and ranking
        recommendations = apply_cultural_intelligence_fast(recommendations, user_country)
        
        user_preferences = {
            "artists": [artist["name"] for artist in user_data["artists"]],
            "tracks": [track["name"] for track in user_data["tracks"]]
        }
        
        recommendations = rank_recommendations_fast(recommendations, user_preferences)
        
        # Step 7: Track new artists
        artist_data = [{"name": rec.get("name", ""), "genre": rec.get("genre", ""), "popularity": rec.get("popularity", 0.0)} for rec in recommendations if rec.get("name")]
        db_service.track_new_artists(user_id, artist_data)
        
        # Step 8: Update user taste analytics for each genre
        for rec in recommendations:
            if rec.get("genre"):
                db_service.update_user_taste_analytics(user_id, rec.get("genre"), 1, 0, 0.0)
        
        # Step 9: Update mood preferences
        if context_analysis.get("primary_mood"):
            db_service.update_mood_preferences(user_id, context_analysis["primary_mood"], 1.0)
        
        # Step 10: Create playlist if requested
        playlist = None
        if data.get("create_playlist", False):
            playlist_name = f"SoniqueDNA - {context_analysis['activity_type'].title()}"
            playlist_description = f"AI-generated playlist for {context_analysis['primary_mood']} mood"
            
            playlist = spotify_service.create_playlist(spotify_token, user_id, playlist_name, playlist_description)
        
        response_time = time.time() - start_time
        
        # Step 11: Store in history
        db_service.store_recommendation_history(
            user_id=user_id,
            session_id=session_id,
            recommendation_type="music",
            user_context=user_context,
            generated_tags=tags,
            qloo_artists=recommendations,
            playlist_data={"recommendations": recommendations[:limit]},
            response_time=response_time
        )
        
        # Step 12: Cache the result
        new_artists = db_service.get_new_artists(user_id, 5)  # Get recent new artists
        response_data = {
            "playlist": playlist,
            "recommendations": recommendations[:limit],
            "new_artists": new_artists,
            "analysis": {
                "context_analysis": context_analysis,
                "tags_used": tags,
                "user_country": user_country,
                "response_time": round(response_time, 2),
                "cache_hit": False
            }
        }
        
        artist_names = [rec.get("name", "") for rec in recommendations if rec.get("name")]
        db_service.store_cached_recommendation(cache_key, user_context, user_country, artist_names, response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Music recommendation error: {e}")
        
        # Fallback to trending recommendations
        fallback_recommendations = get_fallback_recommendations("party")
        
        return jsonify({
            "playlist": None,
            "recommendations": fallback_recommendations,
            "new_artists": [],
            "analysis": {
                "context_analysis": {"primary_mood": "neutral", "activity_type": "general"},
                "tags_used": ["pop", "trending"],
                "fallback": True,
                "response_time": round(time.time() - start_time, 2)
            }
        })

@recommendation_routes.route('/crossdomain-recommendations', methods=['POST'])
def crossdomain_recommendations():
    """Optimized multi-domain recommendations"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        limit = min(int(data.get("limit", 10)), 20)  # Cap at 20
        
        # Step 1: Get user data once
        user_data = spotify_service.get_user_data_fast(spotify_token)
        if not user_data:
            return jsonify({"error": "Failed to get user data"}), 400
        
        user_country = user_data["profile"].get("country", "US")
        user_id = user_data["profile"]["user_id"]
        
        # Step 2: Create user session for tracking
        session_id = db_service.create_user_session(user_id, spotify_token, user_country, "cross-domain recommendations")
        
        # Step 3: Prepare top artists with images
        top_artists_with_images = []
        for artist in user_data.get("artists", [])[:6]:
            top_artists_with_images.append({
                "id": artist.get("id", ""),
                "name": artist.get("name", ""),
                "image": None,  # Will be populated if available
                "genres": artist.get("genres", []),
                "popularity": 0,
                "followers": 0
            })
        
        # Step 3: Generate unified tags
        context = "cross-domain recommendations"
        tags = gemini_service.generate_optimized_tags(context, user_country)
        tag_ids = qloo_service.get_tag_ids_fast(tags)
        
        # Step 4: Process all domains in parallel
        domains = ["artist", "movie", "brand", "destination"]
        recommendations_by_domain = {}
        
        for domain in domains:
            try:
                domain_recommendations = qloo_service.get_cross_domain_recommendations(tag_ids, domain, limit)
                recommendations_by_domain[domain] = domain_recommendations[:limit]
            except Exception as e:
                print(f"Domain {domain} error: {e}")
                recommendations_by_domain[domain] = []
        
        response_time = time.time() - start_time
        
        # Store cross-domain recommendation in history
        all_recommendations = []
        for domain, recs in recommendations_by_domain.items():
            all_recommendations.extend(recs)
        
        db_service.store_recommendation_history(
            user_id=user_id,
            session_id=session_id,
            recommendation_type="crossdomain",
            user_context="cross-domain recommendations",
            generated_tags=tags,
            qloo_artists=all_recommendations,
            playlist_data={"recommendations_by_domain": recommendations_by_domain},
            response_time=response_time
        )
        
        return jsonify({
            "top_artists": [artist["name"] for artist in top_artists_with_images],
            "top_artists_with_images": top_artists_with_images,
            "recommendations_by_domain": recommendations_by_domain,
            "total_domains": len([d for d in domains if recommendations_by_domain.get(d)]),
            "analysis": {
                "tags_used": tags,
                "user_country": user_country,
                "response_time": round(response_time, 2)
            }
        })
        
    except Exception as e:
        print(f"Cross-domain recommendations error: {e}")
        return jsonify({"error": "Failed to get cross-domain recommendations"}), 500

@recommendation_routes.route('/crossdomain-progress/<user_id>', methods=['GET'])
def get_crossdomain_progress(user_id):
    """Real-time progress tracking"""
    try:
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

@recommendation_routes.route('/artist-priority-recommendations', methods=['POST'])
def artist_priority_recommendations():
    """Fast artist-based recommendations"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not validate_input_data(data, ["spotify_token", "artist_name"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        spotify_token = sanitize_string(data["spotify_token"])
        artist_name = sanitize_string(data["artist_name"])
        context = sanitize_string(data.get("context", ""))
        
        # Step 1: Search for artist
        artist_info = spotify_service.search_artist(spotify_token, artist_name)
        if not artist_info:
            return jsonify({"error": "Artist not found"}), 404
        
        # Step 2: Generate focused tags
        tags = gemini_service.generate_optimized_tags(f"{artist_name} {context}")
        tag_ids = qloo_service.get_tag_ids_fast(tags)
        
        # Step 3: Get recommendations with artist signals
        recommendations = qloo_service.get_artist_recommendations_by_tags(
            tag_ids, 
            [artist_info["id"]], 
            limit=15
        )
        
        # Step 4: Analyze artist context
        artist_analysis = gemini_service.analyze_artist_context(artist_name, context)
        
        response_time = time.time() - start_time
        
        return jsonify({
            "recommendations": recommendations,
            "artist_analysis": artist_analysis,
            "artist_info": artist_info,
            "response_time": round(response_time, 2)
        })
        
    except Exception as e:
        print(f"Artist recommendations error: {e}")
        return jsonify({"error": "Failed to get artist recommendations"}), 500 

@recommendation_routes.route('/history/<user_id>', methods=['GET'])
def get_user_history(user_id):
    """Get user's recommendation history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        history = db_service.get_user_history(user_id, limit)
        return jsonify({"history": history})
    except Exception as e:
        print(f"History fetch error: {e}")
        return jsonify({"error": "Failed to get history"}), 500

@recommendation_routes.route('/history/<user_id>/<int:history_id>', methods=['POST'])
def replay_recommendation(user_id, history_id):
    """Replay a specific recommendation from history"""
    try:
        history_item = db_service.get_history_item(history_id)
        if not history_item:
            return jsonify({"error": "History item not found"}), 404
        
        # Re-run recommendation with same context
        data = request.get_json()
        spotify_token = data.get("spotify_token")
        
        if not spotify_token:
            return jsonify({"error": "Missing spotify_token"}), 400
        
        # Use the stored context to generate new recommendations
        user_context = history_item.get("user_context", "")
        
        # Create a new request with the historical context
        replay_data = {
            "spotify_token": spotify_token,
            "user_context": user_context,
            "limit": 15
        }
        
        # Create a new request with the historical context
        request._json = replay_data
        return music_recommendation()
        
    except Exception as e:
        print(f"Replay recommendation error: {e}")
        return jsonify({"error": "Failed to replay recommendation"}), 500

@recommendation_routes.route('/new-artists/<user_id>', methods=['GET'])
def get_new_artists(user_id):
    """Get recently discovered new artists for user"""
    try:
        days = request.args.get('days', 7, type=int)
        print(f"Fetching new artists for user {user_id} in last {days} days")
        new_artists = db_service.get_new_artists(user_id, days)
        print(f"Found {len(new_artists)} new artists: {[artist['artist_name'] for artist in new_artists]}")
        return jsonify({"new_artists": new_artists})
    except Exception as e:
        print(f"New artists fetch error: {e}")
        return jsonify({"error": "Failed to get new artists"}), 500

@recommendation_routes.route('/taste-analytics/<user_id>', methods=['GET'])
def get_user_taste_analytics(user_id):
    """Get user taste analytics for graph visualization"""
    try:
        analytics = db_service.get_user_taste_analytics(user_id)
        return jsonify(analytics)
    except Exception as e:
        print(f"Taste analytics fetch error: {e}")
        return jsonify({"error": "Failed to get taste analytics"}), 500

@recommendation_routes.route('/artists/track', methods=['POST'])
def track_artists():
    """Track artists for new detection"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        artists = data.get('artists', [])
        
        if not user_id or not artists:
            return jsonify({"error": "Missing user_id or artists"}), 400
        
        new_artists = db_service.track_new_artists(user_id, artists)
        return jsonify({"new_artists": new_artists})
    except Exception as e:
        print(f"Artist tracking error: {e}")
        return jsonify({"error": "Failed to track artists"}), 500

@recommendation_routes.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear recommendation cache"""
    try:
        data = request.get_json()
        user_id = data.get("user_id") if data else None
        
        db_service.clear_user_cache(user_id)
        
        if user_id:
            print(f"Cleared cache for user: {user_id}")
        else:
            print("Cleared all cache")
        
        return jsonify({"status": "success", "message": "Cache cleared"})
        
    except Exception as e:
        print(f"Cache clearing error: {e}")
        return jsonify({"error": "Failed to clear cache"}), 500

@recommendation_routes.route('/analytics/clear/<user_id>', methods=['POST'])
def clear_analytics(user_id):
    """Clear analytics data for a user"""
    try:
        db_service.clear_user_analytics(user_id)
        return jsonify({"status": "success", "message": "Analytics cleared"})
    except Exception as e:
        print(f"Analytics clearing error: {e}")
        return jsonify({"error": "Failed to clear analytics"}), 500

@recommendation_routes.route('/artist-details', methods=['POST'])
def get_artist_details():
    """Get detailed artist information from Spotify"""
    try:
        data = request.get_json()
        
        if not data or not data.get("spotify_token") or not data.get("artist_name"):
            return jsonify({"error": "Missing spotify_token or artist_name"}), 400
        
        spotify_token = data["spotify_token"]
        artist_name = data["artist_name"]
        
        print(f"Searching for artist: {artist_name}")
        # First search for the artist to get their ID
        artist_search = spotify_service.search_artist(spotify_token, artist_name)
        
        if not artist_search:
            print(f"Artist '{artist_name}' not found in Spotify search")
            return jsonify({"error": f"Artist '{artist_name}' not found"}), 404
        
        print(f"Found artist in search: {artist_search.get('name', 'Unknown')} (ID: {artist_search.get('id', 'Unknown')})")
        # Get detailed artist information
        artist_details = spotify_service.get_artist_details(spotify_token, artist_search["id"])
        
        if not artist_details:
            return jsonify({"error": "Failed to get artist details"}), 500
        
        return jsonify({"artist": artist_details})
        
    except Exception as e:
        print(f"Artist details error: {e}")
        return jsonify({"error": "Failed to get artist details"}), 500

@recommendation_routes.route('/analytics/populate-sample/<user_id>', methods=['POST'])
def populate_sample_analytics(user_id):
    """Populate sample analytics data for a user"""
    try:
        db_service.populate_sample_analytics(user_id)
        return jsonify({"status": "success", "message": "Sample analytics populated"})
    except Exception as e:
        print(f"Sample analytics population error: {e}")
        return jsonify({"error": "Failed to populate sample analytics"}), 500 