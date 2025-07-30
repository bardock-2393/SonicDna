import urllib.parse
import requests
import base64
import time
import random
from flask import Flask, jsonify, request, render_template
import os
import json
import hashlib
from datetime import datetime, timedelta
from flask_cors import CORS
from functools import wraps
import concurrent.futures
import threading
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import client modules to avoid duplication
from utils import retry_on_failure
from gemini_client import detect_artist_language_with_gemini
from spotify_client import (
    exchange_code_for_token, 
    get_spotify_user_id, 
    get_spotify_user_profile,
    get_spotify_top_artists,
    get_spotify_top_artists_with_images,
    get_spotify_top_tracks,
    get_spotify_recently_played,
    get_spotify_artist_id,
    get_spotify_artist_genres,
    create_spotify_playlist,
    get_spotify_track_uri,
    get_spotify_top_tracks_detailed,
    get_spotify_playlist_by_id,
    validate_token
)
from qloo_client import get_qloo_artist_id

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:8080', 'http://127.0.0.1:8080'], supports_credentials=True)

# In-memory cache for recommendations
recommendation_cache = {}
CACHE_DURATION_MINUTES = 15  # Cache for 30 minutes

# Rate limiting configuration
SPOTIFY_RATE_LIMIT_DELAY = 0.05  # 50ms between requests (reduced for speed)
SPOTIFY_MAX_REQUESTS_PER_MINUTE = 100  # Increased for better performance
spotify_request_times = []

# Gemini API rate limiting configuration
GEMINI_RATE_LIMIT_DELAY = 1.0  # Base delay between calls in seconds
GEMINI_MAX_RETRIES = 3
GEMINI_BACKOFF_FACTOR = 2.0
GEMINI_CACHE_DURATION = 3600  # 1 hour cache

# Redis configuration
REDIS_URL = "redis://default:iArh1BsXVMS8qcvz5dxx6DI5Le0H4svu@redis-12272.c85.us-east-1-2.ec2.redns.redis-cloud.com:12272"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Global cache for Gemini API responses (fallback if Redis fails)
gemini_cache = {}
gemini_last_call_time = 0
gemini_lock = threading.Lock()

def spotify_rate_limit():
    """Rate limiting for Spotify API calls"""
    global spotify_request_times
    current_time = time.time()
    
    # Remove requests older than 1 minute
    spotify_request_times = [t for t in spotify_request_times if current_time - t < 60]
    
    # If we've made too many requests, wait
    if len(spotify_request_times) >= SPOTIFY_MAX_REQUESTS_PER_MINUTE:
        sleep_time = 60 - (current_time - spotify_request_times[0])
        if sleep_time > 0:
            print(f"[RATE LIMIT] Waiting {sleep_time:.2f} seconds for rate limit reset")
            time.sleep(sleep_time)
    
    # Add jitter to avoid thundering herd
    jitter = random.uniform(0, 0.05)
    time.sleep(SPOTIFY_RATE_LIMIT_DELAY + jitter)
    
    # Record this request
    spotify_request_times.append(time.time())

def retry_on_spotify_rate_limit(max_retries=3, base_delay=0.5):
    """Retry decorator specifically for Spotify API with exponential backoff for 429 errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    # Apply rate limiting before each request
                    spotify_rate_limit()
                    
                    result = func(*args, **kwargs)
                    return result
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        # Get retry-after header if available
                        retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                delay = int(retry_after)
                                print(f"[RATE LIMIT] Spotify API returned 429, waiting {delay} seconds as specified")
                                time.sleep(delay)
                            except ValueError:
                                delay = base_delay * (2 ** attempt)
                                print(f"[RATE LIMIT] Invalid Retry-After header, using exponential backoff: {delay}s")
                                time.sleep(delay)
                        else:
                            # Exponential backoff
                            delay = base_delay * (2 ** attempt)
                            print(f"[RATE LIMIT] Spotify API returned 429, attempt {attempt + 1}/{max_retries}, waiting {delay}s")
                            time.sleep(delay)
                        
                        if attempt == max_retries - 1:
                            print(f"[RATE LIMIT] Max retries reached for Spotify API call")
                            return None
                        continue
                    else:
                        # For other HTTP errors, don't retry
                        raise e
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"[ERROR] Max retries reached: {e}")
                        return None
                    delay = base_delay * (2 ** attempt)
                    print(f"[ERROR] Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator

def gemini_rate_limit():
    """Rate limiting decorator for Gemini API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global gemini_last_call_time
            
            with gemini_lock:
                current_time = time.time()
                time_since_last_call = current_time - gemini_last_call_time
                
                if time_since_last_call < GEMINI_RATE_LIMIT_DELAY:
                    sleep_time = GEMINI_RATE_LIMIT_DELAY - time_since_last_call
                    # Add some jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.5)
                    sleep_time += jitter
                    print(f"[GEMINI RATE LIMIT] Waiting {sleep_time:.2f}s before next call")
                    time.sleep(sleep_time)
                
                gemini_last_call_time = time.time()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_on_gemini_rate_limit(max_retries=GEMINI_MAX_RETRIES, base_delay=GEMINI_RATE_LIMIT_DELAY):
    """Retry decorator with exponential backoff for Gemini API rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit exceeded
                        last_exception = e
                        if attempt < max_retries:
                            delay = base_delay * (GEMINI_BACKOFF_FACTOR ** attempt)
                            jitter = random.uniform(0, delay * 0.1)
                            total_delay = delay + jitter
                            print(f"[GEMINI RETRY] Rate limit hit, retrying in {total_delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                            time.sleep(total_delay)
                            continue
                    else:
                        raise e
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (GEMINI_BACKOFF_FACTOR ** attempt)
                        print(f"[GEMINI RETRY] Error occurred, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(delay)
                        continue
                    else:
                        raise e
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

def get_gemini_cache_key(*args, **kwargs):
    """Generate a cache key for Gemini API calls"""
    # Create a hash of the function arguments
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()

def is_gemini_cache_valid(cache_entry):
    """Check if a Gemini cache entry is still valid"""
    if not cache_entry:
        return False
    
    cache_time = cache_entry.get('timestamp', 0)
    current_time = time.time()
    return (current_time - cache_time) < GEMINI_CACHE_DURATION

def get_cached_gemini_response(cache_key):
    """Get cached Gemini response if available and valid"""
    try:
        # Try Redis first
        cached_data = redis_client.get(f"gemini:{cache_key}")
        if cached_data:
            response = json.loads(cached_data)
            print(f"[GEMINI CACHE] Using Redis cached response for key: {cache_key[:8]}...")
            return response
    except Exception as e:
        print(f"[GEMINI CACHE] Redis error, falling back to memory cache: {e}")
        # Fallback to memory cache
        cache_entry = gemini_cache.get(cache_key)
        if is_gemini_cache_valid(cache_entry):
            print(f"[GEMINI CACHE] Using memory cached response for key: {cache_key[:8]}...")
            return cache_entry.get('response')
    return None

def cache_gemini_response(cache_key, response):
    """Cache a Gemini API response"""
    try:
        # Try Redis first
        redis_client.setex(
            f"gemini:{cache_key}", 
            GEMINI_CACHE_DURATION, 
            json.dumps(response)
        )
        print(f"[GEMINI CACHE] Cached response in Redis for key: {cache_key[:8]}...")
    except Exception as e:
        print(f"[GEMINI CACHE] Redis error, falling back to memory cache: {e}")
        # Fallback to memory cache
        gemini_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        print(f"[GEMINI CACHE] Cached response in memory for key: {cache_key[:8]}...")

def clear_expired_gemini_cache():
    """Clear expired entries from Gemini cache"""
    try:
        # Redis automatically expires keys, so we just clear memory cache
        current_time = time.time()
        expired_keys = [
            key for key, entry in gemini_cache.items()
            if (current_time - entry.get('timestamp', 0)) >= GEMINI_CACHE_DURATION
        ]
        for key in expired_keys:
            del gemini_cache[key]
        if expired_keys:
            print(f"[GEMINI CACHE] Cleared {len(expired_keys)} expired memory cache entries")
    except Exception as e:
        print(f"[GEMINI CACHE] Error clearing cache: {e}")

# --- Caching Utilities ---
def generate_cache_key(user_id, context, recommendation_type="music"):
    """Generate a unique cache key based on user and context"""
    key_string = f"{user_id}_{context}_{recommendation_type}"
    return hashlib.md5(key_string.encode()).hexdigest()

def is_cache_valid(cache_entry):
    """Check if cache entry is still valid"""
    if not cache_entry:
        return False
    
    cache_time = cache_entry.get('timestamp')
    if not cache_time:
        return False
    
    # Convert string timestamp back to datetime if needed
    if isinstance(cache_time, str):
        cache_time = datetime.fromisoformat(cache_time)
    
    expiry_time = cache_time + timedelta(minutes=CACHE_DURATION_MINUTES)
    return datetime.now() < expiry_time

def get_cached_recommendation(user_id, context, recommendation_type="music"):
    """Get cached recommendation if valid"""
    cache_key = generate_cache_key(user_id, context, recommendation_type)
    
    try:
        # Try Redis first
        cached_data = redis_client.get(f"recommendation:{cache_key}")
        if cached_data:
            cache_entry = json.loads(cached_data)
            if is_cache_valid(cache_entry):
                print(f"[CACHE HIT] Using Redis cached {recommendation_type} recommendation for user {user_id}")
                return cache_entry['data']
    except Exception as e:
        print(f"[CACHE] Redis error, falling back to memory cache: {e}")
        # Fallback to memory cache
        cache_entry = recommendation_cache.get(cache_key)
        if is_cache_valid(cache_entry):
            print(f"[CACHE HIT] Using memory cached {recommendation_type} recommendation for user {user_id}")
            return cache_entry['data']
    
    return None

def cache_recommendation(user_id, context, data, recommendation_type="music"):
    """Cache recommendation data"""
    cache_key = generate_cache_key(user_id, context, recommendation_type)
    cache_entry = {
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'context': context,
        'type': recommendation_type
    }
    
    try:
        # Try Redis first
        redis_client.setex(
            f"recommendation:{cache_key}", 
            CACHE_DURATION_MINUTES * 60, 
            json.dumps(cache_entry)
        )
        print(f"[CACHE STORE] Cached {recommendation_type} recommendation in Redis for user {user_id}")
    except Exception as e:
        print(f"[CACHE] Redis error, falling back to memory cache: {e}")
        # Fallback to memory cache
        recommendation_cache[cache_key] = cache_entry
        print(f"[CACHE STORE] Cached {recommendation_type} recommendation in memory for user {user_id}")

def clear_user_cache(user_id):
    """Clear all cached recommendations for a specific user"""
    try:
        # Clear from Redis
        pattern = f"recommendation:*"
        keys = redis_client.keys(pattern)
        cleared_count = 0
        
        for key in keys:
            try:
                cached_data = redis_client.get(key)
                if cached_data:
                    cache_entry = json.loads(cached_data)
                    if cache_entry.get('user_id') == user_id:
                        redis_client.delete(key)
                        cleared_count += 1
            except Exception as e:
                print(f"[CACHE CLEAR] Error checking Redis key {key}: {e}")
        
        print(f"[CACHE CLEAR] Cleared {cleared_count} Redis cached entries for user {user_id}")
    except Exception as e:
        print(f"[CACHE CLEAR] Redis error: {e}")
    
    # Also clear from memory cache
    keys_to_remove = []
    for key, entry in recommendation_cache.items():
        if entry.get('user_id') == user_id:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del recommendation_cache[key]
    
    print(f"[CACHE CLEAR] Cleared {len(keys_to_remove)} memory cached entries for user {user_id}")

def cleanup_expired_cache():
    """Remove expired cache entries"""
    try:
        # Redis automatically expires keys, so we just clean memory cache
        keys_to_remove = []
        for key, entry in recommendation_cache.items():
            if not is_cache_valid(entry):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del recommendation_cache[key]
        
        if keys_to_remove:
            print(f"[CACHE CLEANUP] Removed {len(keys_to_remove)} expired memory cache entries")
    except Exception as e:
        print(f"[CACHE CLEANUP] Error: {e}")

# --- Default Config (can be overridden by request) ---
# Load from environment variables with fallbacks
DEFAULT_QLOO_API_KEY = os.getenv('QLOO_API_KEY', "86722dJnD5PS3iDwgGGp03p8rJgOODG9iridsg1Y5KY")
DEFAULT_GEMINI_API_KEY = os.getenv('GEMINI_API_KEY',"AIzaSyBNb-EtpmciV73x0VzhQdHUtaJysd4aRKM")  # Remove hardcoded invalid key
DEFAULT_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', "5b5e4ceb834347e6a6c3b998cfaf0088")
DEFAULT_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', "9c9aadd2b18e49859df887e5e9cc6ede")
DEFAULT_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', "http://127.0.0.1:8080/callback")
DEFAULT_SCOPE = os.getenv('SPOTIFY_SCOPE', "user-read-private user-read-email user-top-read user-read-recently-played playlist-modify-public playlist-modify-private")


def filter_artists_by_language(artists, language_preference, access_token):
    """Filter artists by language preference using batch requests for efficiency"""
    if not artists:
        return []
    
    # Apply rate limiting once for the batch operation
    spotify_rate_limit()
    
    # Get known language mappings for common artists to avoid API calls
    known_language_artists = {
        # English artists
        "ed sheeran": "english", "taylor swift": "english", "justin bieber": "english", 
        "maroon 5": "english", "harry styles": "english", "the chainsmokers": "english",
        "coldplay": "english", "adele": "english", "bruno mars": "english",
        "post malone": "english", "drake": "english", "ariana grande": "english",
        "billie eilish": "english", "dua lipa": "english", "the weeknd": "english",
        
        # Hindi artists
        "pritam": "hindi", "vishal-shekhar": "hindi", "a.r. rahman": "hindi",
        "arijit singh": "hindi", "shreya ghoshal": "hindi", "atif aslam": "hindi",
        "anuv jain": "hindi", "sachin-jigar": "hindi", "shankar-ehsaan-loy": "hindi",
        "yo yo honey singh": "hindi", "sajid-wajid": "hindi", "mithoon": "hindi",
        "jeet gannguli": "hindi", "armaan malik": "hindi", "sunidhi chauhan": "hindi",
        
        # Spanish artists
        "bad bunny": "spanish", "j balvin": "spanish", "maluma": "spanish",
        "shakira": "spanish", "enrique iglesias": "spanish", "ricky martin": "spanish",
        "camila cabello": "spanish", "rosalia": "spanish", "karol g": "spanish"
    }
    
    filtered_artists = []
    artists_to_check = []
    
    # First pass: check known artists
    for artist in artists:
        artist_name = artist.get('name', '').lower()
        if artist_name in known_language_artists:
            artist_language = known_language_artists[artist_name]
            if artist_language == language_preference:
                filtered_artists.append(artist)
                print(f"[LANGUAGE FILTER] ✓ {artist.get('name')} - Known {artist_language} artist")
            else:
                print(f"[LANGUAGE FILTER] ✗ {artist.get('name')} - Known {artist_language} artist, excluding for {language_preference} request")
        else:
            artists_to_check.append(artist)
    
    # If we have enough artists from known list, skip API calls
    if len(filtered_artists) >= 10:
        print(f"[LANGUAGE FILTER] Using {len(filtered_artists)} known artists, skipping API calls")
        return filtered_artists[:10]
    
    # Batch check remaining artists (limit to 20 to avoid too many API calls)
    if artists_to_check and len(artists_to_check) <= 20:
        try:
            # Get artist IDs for batch request
            artist_ids = []
            for artist in artists_to_check:
                artist_id = artist.get('id')
                if artist_id:
                    artist_ids.append(artist_id)
            
            if artist_ids:
                # Make batch request for artist details
                batch_url = "https://api.spotify.com/v1/artists"
                headers = {"Authorization": f"Bearer {access_token}"}
                params = {"ids": ",".join(artist_ids[:20])}  # Limit to 20 artists
                
                response = requests.get(batch_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Process batch results
                for artist_data in data.get('artists', []):
                    if not artist_data:
                        continue
                        
                    artist_name = artist_data.get('name', '')
                    genres = artist_data.get('genres', [])
                    
                    # Simple language detection based on genres and name
                    artist_language =  detect_artist_language(artist_name, genres, gemini_api_key)
                    
                    if artist_language == language_preference:
                        # Find the original artist object
                        for original_artist in artists_to_check:
                            if original_artist.get('id') == artist_data.get('id'):
                                filtered_artists.append(original_artist)
                                print(f"[LANGUAGE FILTER] ✓ {artist_name} - Detected {artist_language} artist")
                                break
                    else:
                        print(f"[LANGUAGE FILTER] ✗ {artist_name} - Detected {artist_language} artist, excluding for {language_preference} request")
                        
        except Exception as e:
            print(f"[LANGUAGE FILTER] Batch API error: {e}")
            # Fallback: include unknown artists with low confidence
            for artist in artists_to_check[:5]:
                filtered_artists.append(artist)
                print(f"[LANGUAGE FILTER] ? {artist.get('name')} - Unknown, including for {language_preference}")
    
    print(f"[LANGUAGE FILTER] Filtered {len(artists)} artists to {len(filtered_artists)} matching language preference")
    return filtered_artists

def detect_artist_language(artist_name, genres, gemini_api_key=None):
    """Detect artist language using Gemini AI or fallback to hard-coded detection"""
    if gemini_api_key and gemini_api_key != "your_gemini_api_key_here" and gemini_api_key != DEFAULT_GEMINI_API_KEY:
        try:
            return detect_artist_language_with_gemini(artist_name, genres, gemini_api_key)
        except Exception as e:
            print(f"[LANGUAGE DETECTION] Gemini failed for {artist_name}: {e}, using fallback")
    
    # Fallback to hard-coded detection
    artist_name_lower = artist_name.lower()
    genres_lower = [g.lower() for g in genres]
    
    # Hindi/Indian indicators
    hindi_indicators = ['bollywood', 'indian', 'hindi', 'punjabi', 'bhangra', 'indie', 'desi']
    if any(indicator in artist_name_lower for indicator in hindi_indicators) or \
       any(indicator in genres_lower for indicator in hindi_indicators):
        return "hindi"
    
    # Spanish indicators
    spanish_indicators = ['reggaeton', 'latin', 'spanish', 'mexican', 'colombian', 'puerto rican']
    if any(indicator in artist_name_lower for indicator in spanish_indicators) or \
       any(indicator in genres_lower for indicator in spanish_indicators):
        return "spanish"
    
    # Default to English for unknown
    return "english"

def detect_specific_mood_activity(user_context, gemini_api_key):
    """
    Detect specific mood/activity from user context using Gemini
    Returns: {'primary_mood': 'dance', 'secondary_moods': ['party', 'energetic'], 'confidence': 0.9, 'activity_type': 'dancing'}
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[MOOD DETECTION] No valid Gemini API key provided, using fallback detection")
        # Fallback to keyword-based detection
        context_lower = user_context.lower()
        if any(word in context_lower for word in ['dance', 'party', 'club']):
            return {
                'primary_mood': 'dance',
                'secondary_moods': ['party', 'energetic'],
                'activity_type': 'dancing',
                'energy_level': 'high',
                'confidence': 0.8,
                'mood_keywords': ['dance', 'party'],
                'explicit_mood_request': True
            }
        elif any(word in context_lower for word in ['study', 'work', 'focus']):
            return {
                'primary_mood': 'study',
                'secondary_moods': ['focused', 'motivated'],
                'activity_type': 'studying',
                'energy_level': 'medium',
                'confidence': 0.8,
                'mood_keywords': ['study', 'focus'],
                'explicit_mood_request': True
            }
        else:
            return {
                'primary_mood': 'general',
                'secondary_moods': [],
                'activity_type': 'general',
                'energy_level': 'medium',
                'confidence': 0.5,
                'mood_keywords': [],
                'explicit_mood_request': False
            }
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
Analyze this user context and detect specific mood/activity preferences: '{user_context}'

Detect:
1. **Primary Mood/Activity**: What specific mood or activity does the user want music for?
2. **Secondary Moods**: What other related moods are acceptable?
3. **Activity Type**: What type of activity is this?
4. **Energy Level**: What energy level is needed?

Common mood/activity indicators:
- Dance: "dance", "dancing", "club", "party", "nightclub", "disco"
- Workout: "workout", "gym", "exercise", "running", "fitness", "training"
- Study: "study", "work", "focus", "concentration", "office", "productivity"
- Sleep: "sleep", "bedtime", "relax", "calm", "soothing", "night"
- Driving: "driving", "road trip", "commute", "travel", "car"
- Party: "party", "celebration", "birthday", "wedding", "festival"
- Romance: "romance", "love", "date", "romantic", "intimate"
- Sad: "sad", "breakup", "heartbreak", "melancholic", "emotional"
- Happy: "happy", "joyful", "cheerful", "upbeat", "positive"

Return ONLY a JSON object with this exact format:
{{
    "primary_mood": "dance",
    "secondary_moods": ["party", "energetic"],
    "activity_type": "dancing",
    "energy_level": "high",
    "confidence": 0.9,
    "mood_keywords": ["dance", "party"],
    "explicit_mood_request": true
}}

If no specific mood is detected, use:
{{
    "primary_mood": "general",
    "secondary_moods": [],
    "activity_type": "general",
    "energy_level": "medium",
    "confidence": 0.5,
    "mood_keywords": [],
    "explicit_mood_request": false
}}
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Check if response has content
        if not data.get('candidates') or not data['candidates'][0].get('content') or not data['candidates'][0]['content'].get('parts'):
            print(f"[MOOD DETECTION] Empty response from Gemini API")
            raise Exception("Empty response from Gemini API")
        
        result_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Check if result_text is empty
        if not result_text:
            print(f"[MOOD DETECTION] Empty text response from Gemini API")
            raise Exception("Empty text response from Gemini API")
        
        print(f"[MOOD DETECTION] Raw response: {result_text[:200]}...")
        
        # Parse JSON response - handle code blocks and extract JSON
        import json
        import re
        
        # Clean the response - remove markdown code blocks if present
        cleaned_text = result_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]  # Remove ```json
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]  # Remove ```
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]  # Remove ```
        
        cleaned_text = cleaned_text.strip()
        
        try:
            mood_pref = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"[MOOD DETECTION] JSON parsing failed: {e}")
            print(f"[MOOD DETECTION] Attempting to extract JSON from response...")
            
            # Try to extract JSON using regex
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                try:
                    mood_pref = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise Exception("Could not parse JSON from response")
            else:
                raise Exception("No JSON found in response")
        
        print(f"[MOOD DETECTION] Detected: {mood_pref['primary_mood']} ({mood_pref['activity_type']}) - Energy: {mood_pref['energy_level']}")
        return mood_pref
        
    except Exception as e:
        print(f"Error detecting mood preference: {e}")
        # Fallback: Simple keyword-based detection
        user_context_lower = user_context.lower()
        
        # Simple keyword detection for moods
        if any(word in user_context_lower for word in ['dance', 'dancing', 'club', 'party', 'nightclub', 'disco']):
            return {
                "primary_mood": "dance",
                "secondary_moods": ["party", "energetic"],
                "activity_type": "dancing",
                "energy_level": "high",
                "confidence": 0.8,
                "mood_keywords": ["dance", "party"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['workout', 'gym', 'exercise', 'running', 'fitness', 'training']):
            return {
                "primary_mood": "workout",
                "secondary_moods": ["energetic", "motivational"],
                "activity_type": "exercise",
                "energy_level": "high",
                "confidence": 0.8,
                "mood_keywords": ["workout", "energy"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['study', 'work', 'focus', 'concentration', 'office', 'productivity']):
            return {
                "primary_mood": "study",
                "secondary_moods": ["calm", "focus"],
                "activity_type": "studying",
                "energy_level": "low",
                "confidence": 0.8,
                "mood_keywords": ["study", "focus"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['sleep', 'bedtime', 'relax', 'calm', 'soothing', 'night']):
            return {
                "primary_mood": "sleep",
                "secondary_moods": ["calm", "relaxing"],
                "activity_type": "sleeping",
                "energy_level": "very_low",
                "confidence": 0.8,
                "mood_keywords": ["sleep", "calm"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['driving', 'road trip', 'commute', 'travel', 'car']):
            return {
                "primary_mood": "driving",
                "secondary_moods": ["upbeat", "energetic"],
                "activity_type": "driving",
                "energy_level": "medium_high",
                "confidence": 0.8,
                "mood_keywords": ["driving", "travel"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['party', 'celebration', 'birthday', 'wedding', 'festival']):
            return {
                "primary_mood": "party",
                "secondary_moods": ["upbeat", "energetic"],
                "activity_type": "partying",
                "energy_level": "high",
                "confidence": 0.8,
                "mood_keywords": ["party", "celebration"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['romance', 'love', 'date', 'romantic', 'intimate']):
            return {
                "primary_mood": "romance",
                "secondary_moods": ["romantic", "intimate"],
                "activity_type": "romance",
                "energy_level": "low",
                "confidence": 0.8,
                "mood_keywords": ["romance", "love"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['sad', 'breakup', 'heartbreak', 'melancholic', 'emotional']):
            return {
                "primary_mood": "sad",
                "secondary_moods": ["melancholic", "emotional"],
                "activity_type": "reflection",
                "energy_level": "low",
                "confidence": 0.8,
                "mood_keywords": ["sad", "emotional"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['happy', 'joyful', 'cheerful', 'upbeat', 'positive']):
            return {
                "primary_mood": "happy",
                "secondary_moods": ["joyful", "positive"],
                "activity_type": "enjoyment",
                "energy_level": "medium_high",
                "confidence": 0.8,
                "mood_keywords": ["happy", "joyful"],
                "explicit_mood_request": True
            }
        elif any(word in user_context_lower for word in ['motivational', 'motivating', 'inspirational', 'energetic']):
            return {
                "primary_mood": "motivational",
                "secondary_moods": ["energetic", "inspiring"],
                "activity_type": "motivation",
                "energy_level": "high",
                "confidence": 0.8,
                "mood_keywords": ["motivational", "energetic"],
                "explicit_mood_request": True
            }
        else:
            return {
                "primary_mood": "general",
                "secondary_moods": [],
                "activity_type": "general",
                "energy_level": "medium",
                "confidence": 0.5,
                "mood_keywords": [],
                "explicit_mood_request": False
            }

def get_artist_mood_capability(artist_name, access_token, target_mood):
    """
    Analyze if an artist is capable of producing music for a specific mood/activity
    Returns: {'can_produce_mood': True, 'confidence': 0.8, 'indicators': ['dance', 'electronic'], 'best_tracks': [...]}
    """
    try:
        # Get artist genres and top tracks
        genres = get_spotify_artist_genres(artist_name, access_token)
        artist_id = get_spotify_artist_id(artist_name, access_token)
        
        if not artist_id:
            return {
                'can_produce_mood': False,
                'confidence': 0.3,
                'indicators': [],
                'best_tracks': []
            }
        
        # Get artist's tracks
        tracks = get_artist_tracks_smart(artist_id, access_token, limit=20)
        
        # Enhanced mood-genre mapping
        mood_genre_map = {
            'dance': {
                'genres': ['dance', 'electronic', 'house', 'techno', 'edm', 'club', 'disco', 'pop', 'hip hop', 'rap'],
                'keywords': ['dance', 'party', 'club', 'electronic', 'upbeat', 'energetic'],
                'energy_level': 'high'
            },
            'workout': {
                'genres': ['hip hop', 'rap', 'electronic', 'pop', 'rock', 'high energy', 'dance'],
                'keywords': ['workout', 'gym', 'energy', 'motivational', 'pump', 'intense'],
                'energy_level': 'high'
            },
            'study': {
                'genres': ['ambient', 'instrumental', 'acoustic', 'classical', 'jazz', 'chill', 'lo-fi'],
                'keywords': ['study', 'focus', 'calm', 'ambient', 'instrumental', 'acoustic'],
                'energy_level': 'low'
            },
            'sleep': {
                'genres': ['ambient', 'chill', 'acoustic', 'instrumental', 'classical', 'lullaby'],
                'keywords': ['sleep', 'calm', 'soothing', 'relax', 'ambient', 'chill'],
                'energy_level': 'very_low'
            },
            'driving': {
                'genres': ['pop', 'rock', 'electronic', 'hip hop', 'road trip', 'driving'],
                'keywords': ['driving', 'road trip', 'upbeat', 'energetic', 'travel'],
                'energy_level': 'medium_high'
            },
            'party': {
                'genres': ['pop', 'dance', 'electronic', 'hip hop', 'rap', 'party', 'upbeat'],
                'keywords': ['party', 'celebration', 'upbeat', 'energetic', 'fun'],
                'energy_level': 'high'
            },
            'romance': {
                'genres': ['r&b', 'pop', 'ballad', 'acoustic', 'romantic', 'love'],
                'keywords': ['romance', 'love', 'romantic', 'ballad', 'acoustic'],
                'energy_level': 'low'
            },
            'sad': {
                'genres': ['sad', 'melancholic', 'emotional', 'ballad', 'acoustic', 'indie'],
                'keywords': ['sad', 'melancholic', 'emotional', 'heartbreak', 'ballad'],
                'energy_level': 'low'
            },
            'happy': {
                'genres': ['pop', 'happy', 'upbeat', 'cheerful', 'positive'],
                'keywords': ['happy', 'joyful', 'cheerful', 'upbeat', 'positive'],
                'energy_level': 'medium_high'
            }
        }
        
        mood_config = mood_genre_map.get(target_mood, {
            'genres': [],
            'keywords': [],
            'energy_level': 'medium'
        })
        
        # Analyze genres
        genre_score = 0
        genre_matches = []
        for genre in genres:
            genre_lower = genre.lower()
            for target_genre in mood_config['genres']:
                if target_genre in genre_lower:
                    genre_score += 1
                    genre_matches.append(genre)
        
        # Analyze track names and titles for mood keywords
        track_score = 0
        mood_tracks = []
        for track in tracks:
            track_name = track.get('name', '').lower()
            album_name = track.get('album_name', '').lower()
            
            for keyword in mood_config['keywords']:
                if keyword in track_name or keyword in album_name:
                    track_score += 1
                    mood_tracks.append(track)
                    break
        
        # Calculate confidence
        total_score = genre_score + track_score
        confidence = min(total_score / 10.0, 1.0)  # Normalize to 0-1
        
        # Determine if artist can produce the mood
        can_produce = confidence >= 0.4  # Threshold for mood capability
        
        result = {
            'can_produce_mood': can_produce,
            'confidence': round(confidence, 2),
            'indicators': genre_matches,
            'best_tracks': mood_tracks[:5],  # Top 5 mood-matching tracks
            'genre_score': genre_score,
            'track_score': track_score,
            'total_tracks_analyzed': len(tracks)
        }
        
        print(f"[MOOD ANALYSIS] {artist_name} - {target_mood}: {can_produce} (confidence: {confidence})")
        return result
        
    except Exception as e:
        print(f"Error analyzing mood capability for {artist_name}: {e}")
        return {
            'can_produce_mood': False,
            'confidence': 0.3,
            'indicators': [],
            'best_tracks': []
        }

def filter_artists_by_mood_capability(artists, mood_preference, access_token):
    """
    Filter artists based on their capability to produce music for the specific mood/activity
    Returns: filtered list of artists that can produce the requested mood
    """
    if not mood_preference or mood_preference.get('primary_mood') == 'general':
        return artists
    
    filtered_artists = []
    primary_mood = mood_preference['primary_mood']
    secondary_moods = mood_preference.get('secondary_moods', [])
    confidence_threshold = 0.4
    
    print(f"[MOOD FILTER] Filtering artists for mood: {primary_mood}")
    print(f"[MOOD FILTER] Secondary moods: {secondary_moods}")
    
    for artist in artists:
        try:
            # Check primary mood capability
            mood_analysis = get_artist_mood_capability(artist, access_token, primary_mood)
            
            if mood_analysis['can_produce_mood'] and mood_analysis['confidence'] >= confidence_threshold:
                filtered_artists.append(artist)
                print(f"[MOOD FILTER] ✓ {artist} can produce {primary_mood} (confidence: {mood_analysis['confidence']})")
                continue
            
            # Check secondary moods if primary mood fails
            for secondary_mood in secondary_moods:
                secondary_analysis = get_artist_mood_capability(artist, access_token, secondary_mood)
                if secondary_analysis['can_produce_mood'] and secondary_analysis['confidence'] >= confidence_threshold:
                    filtered_artists.append(artist)
                    print(f"[MOOD FILTER] ✓ {artist} can produce {secondary_mood} (confidence: {secondary_analysis['confidence']})")
                    break
            else:
                print(f"[MOOD FILTER] ✗ {artist} cannot produce {primary_mood} or secondary moods")
                
        except Exception as e:
            print(f"Error filtering artist {artist} for mood: {e}")
            # Include artist if we can't determine mood capability (conservative approach)
            filtered_artists.append(artist)
    
    print(f"[MOOD FILTER] Filtered {len(artists)} artists to {len(filtered_artists)} mood-capable artists")
    return filtered_artists

def get_mood_specific_tracks(artist_name, mood_preference, access_token, limit=10):
    """
    Get tracks from an artist that specifically match the requested mood/activity
    Returns: list of tracks that match the mood
    """
    try:
        artist_id = get_spotify_artist_id(artist_name, access_token)
        if not artist_id:
            return []
        
        # Get all tracks from artist
        all_tracks = get_artist_tracks_smart(artist_id, access_token, limit=30)
        
        # Analyze each track for mood match
        mood_tracks = []
        primary_mood = mood_preference.get('primary_mood', 'general')
        
        for track in all_tracks:
            track_name = track.get('name', '').lower()
            album_name = track.get('album_name', '').lower()
            
            # Check if track matches mood keywords
            mood_keywords = mood_preference.get('mood_keywords', [])
            for keyword in mood_keywords:
                if keyword in track_name or keyword in album_name:
                    track['mood_match'] = primary_mood
                    track['mood_confidence'] = 0.8
                    mood_tracks.append(track)
                    break
        
        # If no explicit mood matches, use genre-based filtering
        if not mood_tracks:
            mood_analysis = get_artist_mood_capability(artist_name, access_token, primary_mood)
            if mood_analysis['can_produce_mood']:
                # Use top tracks from the artist (assuming they can produce the mood)
                mood_tracks = all_tracks[:limit]
                for track in mood_tracks:
                    track['mood_match'] = primary_mood
                    track['mood_confidence'] = mood_analysis['confidence']
        
        return mood_tracks[:limit]
        
    except Exception as e:
        print(f"Error getting mood-specific tracks for {artist_name}: {e}")
        return []

def enhance_context_detection_with_mood_and_language(user_context, gemini_api_key):
    """
    Enhanced context detection that considers both mood/activity and language preferences
    Returns: {'context_type': 'party', 'mood_preference': {...}, 'language_preference': {...}, 'enhanced_context': {...}}
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[ENHANCED CONTEXT] No valid Gemini API key provided, using fallback detection")
        # Return fallback enhanced context
        return {
            'context_type': 'general',
            'mood_preference': {
                'primary_mood': 'general',
                'secondary_moods': [],
                'activity_type': 'general',
                'energy_level': 'medium',
                'confidence': 0.5,
                'mood_keywords': [],
                'explicit_mood_request': False
            },
            'language_preference': {
                'primary_language': 'any',
                'secondary_languages': [],
                'confidence': 0.5,
                'language_keywords': [],
                'explicit_language_request': False
            },
            'user_context': user_context,
            'has_mood_request': False,
            'has_language_request': False,
            'primary_mood': 'general',
            'primary_language': 'any',
            'activity_type': 'general',
            'energy_level': 'medium',
            'confidence': 0.5
        }
    
    try:
        # Get basic context type
        context_type = detect_context_type_llm(user_context, gemini_api_key)
        
        # Get mood preference
        mood_preference = detect_specific_mood_activity(user_context, gemini_api_key)
        
        # Get language preference
        language_preference = detect_language_preference(user_context, gemini_api_key)
        
        # Create enhanced context
        enhanced_context = {
            'context_type': context_type,
            'mood_preference': mood_preference,
            'language_preference': language_preference,
            'user_context': user_context,
            'has_mood_request': mood_preference.get('explicit_mood_request', False),
            'has_language_request': language_preference.get('explicit_language_request', False),
            'primary_mood': mood_preference.get('primary_mood', 'general'),
            'primary_language': language_preference.get('primary_language', 'any'),
            'activity_type': mood_preference.get('activity_type', 'general'),
            'energy_level': mood_preference.get('energy_level', 'medium'),
            'confidence': min(mood_preference.get('confidence', 0.5), language_preference.get('confidence', 0.5))
        }
        
        print(f"[ENHANCED CONTEXT] Context: {context_type}, Mood: {mood_preference['primary_mood']}, Language: {language_preference['primary_language']}")
        print(f"[ENHANCED CONTEXT] Activity: {mood_preference['activity_type']}, Energy: {mood_preference['energy_level']}")
        return enhanced_context
        
    except Exception as e:
        print(f"Error in enhanced context detection: {e}")
        return {
            'context_type': 'general',
            'mood_preference': {
                'primary_mood': 'general',
                'secondary_moods': [],
                'activity_type': 'general',
                'energy_level': 'medium',
                'confidence': 0.5,
                'mood_keywords': [],
                'explicit_mood_request': False
            },
            'language_preference': {
                'primary_language': 'any',
                'secondary_languages': [],
                'confidence': 0.5,
                'language_keywords': [],
                'explicit_language_request': False
            },
            'user_context': user_context,
            'has_mood_request': False,
            'has_language_request': False,
            'primary_mood': 'general',
            'primary_language': 'any',
            'activity_type': 'general',
            'energy_level': 'medium',
            'confidence': 0.5
        }

# --- Gemini: Context to Tags ---
def call_gemini_for_enhanced_tags(user_context, gemini_api_key, user_country=None, location=None):
    """Enhanced Gemini function that leverages Qloo's cultural intelligence capabilities"""
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[ENHANCED TAGS] No valid Gemini API key provided, using fallback tags")
        # Return fallback tags based on context
        context_lower = user_context.lower()
        if any(word in context_lower for word in ['dance', 'party', 'club']):
            return ['upbeat', 'energetic', 'dance', 'party', 'electronic', 'pop']
        elif any(word in context_lower for word in ['study', 'work', 'focus']):
            return ['calm', 'focus', 'ambient', 'instrumental', 'study']
        elif any(word in context_lower for word in ['sad', 'breakup', 'heartbreak']):
            return ['sad', 'melancholic', 'emotional', 'ballad', 'acoustic']
        else:
            return ['pop', 'upbeat', 'energetic', 'happy', 'motivational']
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    # Enhanced prompt that showcases Qloo's power
    prompt = f"""
You are a music recommendation expert working with Qloo's cultural intelligence platform. 
Analyze this user context: '{user_context}'

User location: {location or 'Global'}
User country: {user_country or 'Unknown'}

Generate a CONCISE and FOCUSED list of music tags that would work with Qloo's API. Consider:

1. **Primary Emotional Context**: What is the dominant emotion in this situation?
2. **Cultural Relevance**: What cultural elements are most relevant for {user_country}?
3. **Activity Context**: What type of activity or situation is this?
4. **Musical Style**: What musical characteristics would best match this context?

Return ONLY a comma-separated list of 8-12 tags that Qloo can understand, prioritizing:

EMOTIONAL/MOOD TAGS (50% of tags - 4-6 tags):
- For sad/breakup contexts: sad, melancholic, heartbroken, emotional, introspective, lonely, cathartic
- For happy/party contexts: upbeat, energetic, happy, celebratory, festive, vibrant
- For calm/relaxing contexts: calm, soothing, peaceful, relaxing, ambient

GENRE TAGS (30% of tags - 2-4 tags):
- For India: bollywood, hindi pop, indian classical, ghazal, sufi
- For US: pop, rock, hip hop, r&b, electronic
- For Korea: k-pop, korean pop
- For Japan: j-pop, japanese rock
- For Latin America: latin pop, reggaeton

ACTIVITY/STYLE TAGS (15% of tags - 1-2 tags):
- For party: dance, electronic, high energy
- For workout: high energy, energetic, motivational
- For study/work: ambient, instrumental, acoustic
- For driving: road trip, driving music, upbeat

CULTURAL/LOCATION TAGS (5% of tags - 1 tag):
- For India: mumbai, india, desi
- For US: american, country
- For Korea: korean
- For Japan: japanese

IMPORTANT GUIDELINES:
- Focus on tags that Qloo's music database would recognize
- Prioritize emotional and mood tags that match the user's situation
- Include cultural tags appropriate for the user's country
- Avoid generic tags that don't add value
- Ensure tags are specific enough to be useful but broad enough to find matches
- Keep the list concise and focused - quality over quantity
- Avoid redundant or similar tags
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        tag_str = data['candidates'][0]['content']['parts'][0]['text']
        tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        
        # Filter and validate tags to ensure quality
        filtered_tags = []
        for tag in tags:
            # Remove quotes and extra punctuation
            tag = tag.strip('"\'.,!?')
            # Skip if too short or too long
            if len(tag) < 2 or len(tag) > 30:
                continue
            # Skip if it's just a number or single character
            if tag.isdigit() or len(tag) == 1:
                continue
            # Skip if it's a generic filler word
            if tag.lower() in ['and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                continue
            filtered_tags.append(tag)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in filtered_tags:
            if tag.lower() not in seen:
                seen.add(tag.lower())
                unique_tags.append(tag)
        
        # Limit to target range (8-12 tags) to prevent information overload
        if len(unique_tags) > 12:
            unique_tags = unique_tags[:12]
            print(f"[ENHANCED GEMINI] Limited to 12 tags to prevent information overload")
        
        print(f"[ENHANCED GEMINI] Generated {len(unique_tags)} focused tags: {unique_tags}")
        return unique_tags
    except Exception as e:
        print("Error calling Gemini for enhanced tags:", e)
        return []



def generate_cultural_context(user_context, user_country, location, gemini_api_key):
    """Generate cultural context using Gemini for Qloo's location-based intelligence"""
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[CULTURAL CONTEXT] No valid Gemini API key provided, using fallback cultural context")
        return _get_fallback_cultural_context(location, user_country)
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
For music recommendations in {location} ({user_country}), analyze this context: '{user_context}'
What cultural elements, local music trends, or regional preferences should be considered?
Return ONLY a short, clean list of cultural keywords (max 200 characters) that Qloo can use for location-based recommendations.
Format: "keyword1, keyword2, keyword3" - no newlines, no special characters.
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Check if response has content
        if not data.get('candidates') or not data['candidates'][0].get('content') or not data['candidates'][0]['content'].get('parts'):
            print(f"[CULTURAL CONTEXT] Empty response from Gemini API")
            return _get_fallback_cultural_context(location, user_country)
        
        cultural_context = data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Check if result is empty
        if not cultural_context:
            print(f"[CULTURAL CONTEXT] Empty text response from Gemini API")
            return _get_fallback_cultural_context(location, user_country)
        
        # Clean and truncate the cultural context
        cultural_context = cultural_context.replace('\n', ' ').replace('\r', ' ')
        cultural_context = ' '.join(cultural_context.split())  # Remove extra whitespace
        cultural_context = cultural_context[:200]  # Truncate to 200 characters
        
        print(f"[CULTURAL CONTEXT] Generated: {cultural_context}")
        return cultural_context
        
    except Exception as e:
        print(f"Error generating cultural context: {e}")
        return _get_fallback_cultural_context(location, user_country)

def _get_fallback_cultural_context(location, user_country):
    """Fallback cultural context when Gemini API fails"""
    # Simple fallback based on location/country
    if user_country == "IN" or "india" in location.lower():
        return "bollywood, indian pop, mumbai, cosmopolitan, international"
    elif user_country == "US":
        return "american pop, rock, hip hop, country, diverse"
    elif user_country == "KR":
        return "k-pop, korean pop, seoul, modern, trendy"
    elif user_country == "JP":
        return "j-pop, japanese rock, tokyo, anime, modern"
    elif user_country == "ES":
        return "spanish pop, latin, reggaeton, flamenco, mediterranean"
    else:
        return "international, pop, modern, diverse"

# --- Spotify Auth & User Data ---
# exchange_code_for_token function moved to spotify_client.py to avoid duplication

def revoke_spotify_token(access_token, client_id, client_secret):
    """Revoke a Spotify access token to force re-authentication"""
    # Spotify doesn't have a direct token revocation endpoint like some other OAuth providers
    # Instead, we'll use a different approach to force re-authentication
    
    # Method 1: Try to invalidate the token by making an invalid request
    try:
        # Make a request to Spotify API with the token to see if it's still valid
        headers = {"Authorization": f"Bearer {access_token}"}
        resp = requests.get("https://api.spotify.com/v1/me", headers=headers)
        
        if resp.status_code == 401:
            # Token is already invalid, which is good
            print("Token is already invalid")
            return True
        elif resp.status_code == 200:
            # Token is still valid, we need to force re-auth
            print("Token is still valid, will force re-authentication")
            return True
        else:
            print(f"Unexpected response from Spotify: {resp.status_code}")
            return True
    except Exception as e:
        print(f"Error checking token validity: {e}")
        return True

def force_spotify_reauth():
    """Force Spotify to require re-authentication by using a unique state parameter"""
    import secrets
    # Generate a unique state parameter to force new OAuth flow
    unique_state = secrets.token_urlsafe(32)
    return unique_state

# get_spotify_user_id function moved to spotify_client.py to avoid duplication

# get_spotify_user_profile function moved to spotify_client.py to avoid duplication

# get_spotify_top_artists function moved to spotify_client.py to avoid duplication

# get_spotify_top_artists_with_images function moved to spotify_client.py to avoid duplication

# get_spotify_top_tracks function moved to spotify_client.py to avoid duplication

# get_spotify_top_tracks_detailed function moved to spotify_client.py to avoid duplication

# --- Qloo Section ---
class QlooAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://hackathon.api.qloo.com"
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def search_entity(self, query: str):
        try:
            url = f"{self.base_url}/search"
            params = {"query": query, "limit": 10}
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                if results:
                    for result in results:
                        if result.get('name', '').lower() == query.lower():
                            return result
                    return results[0]
            return None
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return None

    def get_recommendations(self, entity_type: str, tag_id: str, artist_ids=None, take: int = 1):
        params = {
            "filter.type": entity_type,
            "filter.tags": tag_id,
            "take": take
        }
        if artist_ids:
            params["signal.interests.entities"] = ",".join(artist_ids)
        try:
            url = f"{self.base_url}/v2/insights"
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None

# Enhanced Qloo client with full API capabilities
class EnhancedQlooClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://hackathon.api.qloo.com/v2"
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    
    def search_entities(self, query: str, entity_type: str = "artist", limit: int = 10):
        """Search for entities using Qloo's search API"""
        url = f"{self.base_url}/entities/search"
        params = {
            "q": query,
            "filter.type": f"urn:entity:{entity_type}",
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            entities = response.json().get("results", {}).get("entities", [])
            print(f"[QLOO SEARCH] Found {len(entities)} entities for '{query}'")
            return entities
        except Exception as e:
            print(f"Error searching Qloo entities: {e}")
            return []
    
    def get_enhanced_recommendations(self, tag_ids, user_artists, user_tracks, 
                                   location=None, location_radius=None, 
                                   cultural_context=None, limit=20):
        """Get enhanced recommendations using Qloo's full capabilities"""
        url = f"{self.base_url}/insights"
        
        # Enhanced parameters for better recommendations
        params = {
            "filter.type": "urn:entity:artist",
            "filter.tags": ",".join(tag_ids[:5]),  # Use exactly 5 most relevant tags
            "filter.popularity.min": 0.1,  # Lower threshold for more diverse results
            "limit": limit * 2,  # Reduced multiplier to avoid too many results
            "sort": "relevance"  # Sort by relevance
        }
        
        # Add location-based intelligence
        if location:
            params["signal.location.query"] = location
            # Fix radius parameter - ensure it's a valid float between 0 and 800000
            if location_radius:
                try:
                    radius = float(location_radius)
                    if 0 <= radius <= 800000:
                        params["signal.location.radius"] = radius
                    else:
                        params["signal.location.radius"] = 50000  # Default safe value
                except (ValueError, TypeError):
                    params["signal.location.radius"] = 50000  # Default safe value
            else:
                params["signal.location.radius"] = 50000  # Default safe value
            print(f"[QLOO ENHANCED] Using location intelligence: {location} with radius: {params.get('signal.location.radius')}")
        
        # Add cultural context signals - Clean and truncate to avoid URL issues
        if cultural_context:
            # Clean the cultural context - remove newlines, extra spaces, and truncate
            cleaned_context = cultural_context.replace('\n', ' ').replace('\r', ' ')
            cleaned_context = ' '.join(cleaned_context.split())  # Remove extra whitespace
            cleaned_context = cleaned_context[:500]  # Truncate to 500 characters
            
            params["signal.cultural.context"] = cleaned_context
            print(f"[QLOO ENHANCED] Using cleaned cultural context: {cleaned_context[:100]}...")
        
        # Enhanced user signals - Limit to avoid URL length issues
        signals = []
        for artist in user_artists[:8]:  # Reduced from 10 to 8
            signals.append(f"urn:entity:artist:{artist}")
        for track in user_tracks[:8]:  # Reduced from 10 to 8
            signals.append(f"urn:entity:track:{track}")
        
        if signals:
            params["signals"] = ",".join(signals)
            print(f"[QLOO ENHANCED] Using {len(signals)} user signals")
        
        try:
            # Log the URL for debugging (without sensitive data)
            debug_url = f"{url}?filter.type={params.get('filter.type')}&limit={params.get('limit')}"
            print(f"[QLOO ENHANCED] Making request to: {debug_url}")
            
            response = requests.get(url, headers=self.headers, params=params)
            
            # Log response status for debugging
            print(f"[QLOO ENHANCED] Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[QLOO ENHANCED] Error response: {response.text[:200]}")
                # Fallback to simpler request without cultural context
                return self._get_fallback_recommendations(tag_ids, user_artists, user_tracks, location, limit)
            
            response.raise_for_status()
            
            entities = response.json().get("results", {}).get("entities", [])
            recommended_artists = []
            
            for entity in entities:
                if entity.get("subtype") == "urn:entity:artist":
                    artist_info = {
                        "name": entity.get("name"),
                        "id": entity.get("id"),
                        "popularity": entity.get("popularity", 0),
                        "affinity_score": entity.get("affinity_score", 0),
                        "cultural_relevance": entity.get("cultural_relevance", 0),
                        "tags": entity.get("tags", [])
                    }
                    recommended_artists.append(artist_info)
            
            # Sort by relevance score
            recommended_artists.sort(key=lambda x: x.get("affinity_score", 0), reverse=True)
            
            print(f"[QLOO ENHANCED] Found {len(recommended_artists)} enhanced recommendations")
            return recommended_artists
            
        except Exception as e:
            print(f"Error getting enhanced Qloo recommendations: {e}")
            # Fallback to simpler request
            return self._get_fallback_recommendations(tag_ids, user_artists, user_tracks, location, limit)
    
    def _get_fallback_recommendations(self, tag_ids, user_artists, user_tracks, location, limit):
        """Fallback method for when enhanced recommendations fail"""
        print(f"[QLOO FALLBACK] Using simplified request")
        
        url = f"{self.base_url}/insights"
        
        # Simplified parameters without cultural context
        params = {
            "filter.type": "urn:entity:artist",
            "filter.tags": ",".join(tag_ids[:10]),  # Even fewer tags
            "filter.popularity.min": 0.1,
            "limit": limit,
            "sort": "relevance"
        }
        
        # Add location if available
        if location:
            params["signal.location.query"] = location
        
        # Add minimal user signals
        signals = []
        for artist in user_artists[:5]:  # Only 5 artists
            signals.append(f"urn:entity:artist:{artist}")
        
        if signals:
            params["signals"] = ",".join(signals)
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            entities = response.json().get("results", {}).get("entities", [])
            recommended_artists = []
            
            for entity in entities:
                if entity.get("subtype") == "urn:entity:artist":
                    artist_info = {
                        "name": entity.get("name"),
                        "id": entity.get("id"),
                        "popularity": entity.get("popularity", 0),
                        "affinity_score": entity.get("affinity_score", 0),
                        "cultural_relevance": entity.get("cultural_relevance", 0),
                        "tags": entity.get("tags", [])
                    }
                    recommended_artists.append(artist_info)
            
            recommended_artists.sort(key=lambda x: x.get("affinity_score", 0), reverse=True)
            
            print(f"[QLOO FALLBACK] Found {len(recommended_artists)} fallback recommendations")
            return recommended_artists
            
        except Exception as e:
            print(f"Error in fallback Qloo recommendations: {e}")
            return []
    
    def get_cultural_insights(self, location, domain="music"):
        """Get cultural insights for a location using proper Qloo API parameters"""
        url = f"{self.base_url}/insights"
        
        # Use proper entity type mapping
        entity_type_map = {
            "music": "urn:entity:artist",
            "artist": "urn:entity:artist",
            "movie": "urn:entity:movie",
            "brand": "urn:entity:brand",
            "destination": "urn:entity:destination"
        }
        
        entity_type = entity_type_map.get(domain, "urn:entity:artist")
        
        params = {
            "filter.type": entity_type,
            "signal.location.query": location,
            "take": 10,
            "feature.explainability": "true"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            insights = data.get("results", {}).get("entities", [])
            print(f"[QLOO CULTURAL] Found {len(insights)} cultural insights for {location}")
            return insights
        except Exception as e:
            print(f"Error getting cultural insights: {e}")
            # Return empty list instead of failing
            return []

def filter_and_rank_tags_for_music(tags, user_context, user_country, location, gemini_api_key):
    """
    Filter and rank tags to get the 5 most relevant tags for music recommendations.
    Uses AI to analyze relevance and cultural appropriateness.
    """
    if not tags or len(tags) <= 5:
        return tags[:5] if tags else []
    
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[TAG FILTERING] No valid Gemini API key provided, using first 5 tags")
        return tags[:5]
    
    # Create cache key for this specific request
    cache_key = f"tag_filter_{hash(tuple(sorted(tags)))}_{hash(user_context)}_{user_country}_{location}"
    
    # Check if we have a cached result
    cached_result = recommendation_cache.get(cache_key)
    if cached_result and is_cache_valid(cached_result):
        print(f"[TAG FILTERING] Using cached result for tag filtering")
        return cached_result['data']
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    # Create a prompt to rank tags by relevance
    prompt = f"""
You are a music recommendation expert. Given this user context: "{user_context}"
User location: {location or 'Global'}
User country: {user_country or 'Unknown'}

I have these music tags: {', '.join(tags)}

Please analyze and rank these tags by their relevance for music recommendations. Consider:

1. **Emotional Relevance**: How well does the tag match the user's emotional state?
2. **Cultural Appropriateness**: Is the tag culturally relevant for {user_country}?
3. **Musical Specificity**: How specific and useful is the tag for finding music?
4. **Qloo Compatibility**: Will this tag work well with Qloo's music database?

Return ONLY the 5 most relevant tags in order of importance (most relevant first), separated by commas.
Focus on tags that will give the best music recommendations for this specific context.

Example format: tag1, tag2, tag3, tag4, tag5
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        ranked_tags_str = data['candidates'][0]['content']['parts'][0]['text']
        
        # Parse the ranked tags
        ranked_tags = [tag.strip().strip('"\'.,!?') for tag in ranked_tags_str.split(",") if tag.strip()]
        
        # Filter to only include tags that were in the original list
        filtered_ranked_tags = []
        original_tags_lower = [tag.lower() for tag in tags]
        
        for ranked_tag in ranked_tags:
            if ranked_tag.lower() in original_tags_lower:
                # Find the original tag (preserving case)
                for original_tag in tags:
                    if original_tag.lower() == ranked_tag.lower():
                        filtered_ranked_tags.append(original_tag)
                        break
        
        # Ensure we have exactly 5 tags
        if len(filtered_ranked_tags) > 5:
            filtered_ranked_tags = filtered_ranked_tags[:5]
            print(f"[TAG FILTERING] Truncated to exactly 5 tags")
        elif len(filtered_ranked_tags) < 5:
            # Add remaining tags if we don't have 5
            remaining_tags = [tag for tag in tags if tag not in filtered_ranked_tags]
            filtered_ranked_tags.extend(remaining_tags[:5 - len(filtered_ranked_tags)])
            print(f"[TAG FILTERING] Added {5 - len(filtered_ranked_tags)} additional tags to reach 5")
        
        # Final validation
        if len(filtered_ranked_tags) != 5:
            print(f"[TAG FILTERING] WARNING: Expected 5 tags but got {len(filtered_ranked_tags)}")
            # Ensure exactly 5 tags
            while len(filtered_ranked_tags) < 5 and tags:
                for tag in tags:
                    if tag not in filtered_ranked_tags:
                        filtered_ranked_tags.append(tag)
                        break
                if len(filtered_ranked_tags) == len(tags):
                    break
            filtered_ranked_tags = filtered_ranked_tags[:5]
        
        print(f"[TAG FILTERING] AI selected 5 most relevant tags: {filtered_ranked_tags}")
        print(f"[TAG FILTERING] Tag selection process: {len(tags)} original → {len(filtered_ranked_tags)} filtered")
        
        # Cache the result
        cache_recommendation(cache_key, filtered_ranked_tags, "tag_filtering")
        
        return filtered_ranked_tags
        
    except Exception as e:
        print(f"[TAG FILTERING] Error ranking tags: {e}")
        # Fallback: use simple heuristic to select 5 most relevant tags
        fallback_tags = []
        
        # Priority order for fallback selection
        priority_keywords = ['emotional', 'sad', 'happy', 'upbeat', 'energetic', 'calm', 'romantic', 'melancholic']
        cultural_keywords = ['bollywood', 'hindi', 'indian', 'pop', 'rock', 'electronic']
        activity_keywords = ['dance', 'workout', 'study', 'driving', 'party']
        
        # First, try to find priority emotional/mood tags
        for keyword in priority_keywords:
            for tag in tags:
                if keyword in tag.lower() and tag not in fallback_tags and len(fallback_tags) < 5:
                    fallback_tags.append(tag)
        
        # Then, add cultural tags
        for keyword in cultural_keywords:
            for tag in tags:
                if keyword in tag.lower() and tag not in fallback_tags and len(fallback_tags) < 5:
                    fallback_tags.append(tag)
        
        # Finally, add activity tags
        for keyword in activity_keywords:
            for tag in tags:
                if keyword in tag.lower() and tag not in fallback_tags and len(fallback_tags) < 5:
                    fallback_tags.append(tag)
        
        # If we still don't have 5 tags, add remaining tags
        for tag in tags:
            if tag not in fallback_tags and len(fallback_tags) < 5:
                fallback_tags.append(tag)
        
        print(f"[TAG FILTERING] Fallback selected tags: {fallback_tags}")
        
        # Cache the fallback result
        cache_recommendation(cache_key, fallback_tags, "tag_filtering")
        
        return fallback_tags


def get_qloo_tag_ids(tags, qloo_api_key):
    """Convert tags to Qloo tag IDs with enhanced emotional tag mapping"""
    tag_ids = []
    successful_tags = []  # Track which tags were successfully converted
    headers = {
        "X-API-KEY": qloo_api_key,
        "Content-Type": "application/json"
    }
    
    # Enhanced emotional tag mapping with better Qloo category targeting
    emotional_tag_mapping = {
        'heartbroken': 'sad',
        'alone': 'lonely',
        'remembering': 'nostalgic',
        'missing someone': 'longing',
        'crying': 'emotional',
        'vulnerable': 'emotional',
        'reflective': 'introspective',
        'pensive': 'contemplative',
        'cathartic': 'emotional',
        'longing': 'romantic',
        'healing': 'calming',
        'recovery': 'healing',
        'moving on': 'recovery',
        'happy': 'upbeat',
        'joyful': 'upbeat',
        'excited': 'energetic',
        'energetic': 'high_energy',
        'party': 'party_anthems',
        'celebration': 'festive',
        'dance': 'dance_pop',
        'upbeat': 'upbeat',
        'melancholic': 'melancholic',
        'sad': 'sad',
        'emotional': 'emotional',
        'romantic': 'romantic',
        'love': 'romantic',
        'passionate': 'romantic',
        'intimate': 'romantic',
        'nostalgic': 'nostalgic',
        'sentimental': 'sentimental',
        'soothing': 'calm',
        'calm': 'calm',
        'peaceful': 'calm',
        'relaxing': 'calm',
        'chill': 'chill',
        'mellow': 'chill',
        'acoustic': 'acoustic',
        'unplugged': 'acoustic',
        'ballad': 'ballad',
        'slow': 'slowcore',
        'downtempo': 'downtempo',
        'ambient': 'ambient'
    }
    
    # Cultural tag mapping to avoid restaurant/place categories
    cultural_tag_mapping = {
        'punjabi': 'bhangra',
        'bengali': 'indian_classical',
        'tamil': 'indian_classical',
        'telugu': 'indian_classical',
        'malayalam': 'indian_classical',
        'kannada': 'indian_classical',
        'urdu': 'ghazal',
        'marathi': 'indian_classical',
        'desi': 'indian_pop',
        'india': 'indian_pop',
        'mumbai': 'bollywood',
        'delhi': 'bollywood',
        'bangalore': 'indian_pop'
    }
    
    # Activity tag mapping
    activity_tag_mapping = {
        'workout': 'high_energy',
        'gym': 'high_energy',
        'running': 'high_energy',
        'studying': 'ambient',
        'work': 'ambient',
        'driving': 'road_trip',
        'travel': 'road_trip',
        'cooking': 'easy_listening',
        'cleaning': 'easy_listening',
        'sleeping': 'ambient',
        'meditation': 'ambient',
        'yoga': 'ambient'
    }
    
    print(f"[QLOO TAGS] Converting {len(tags)} tags to Qloo tag IDs...")
    
    for tag in tags:
        try:
            # Try original tag first
            params = {"filter.query": tag, "limit": 1}
            resp = requests.get("https://hackathon.api.qloo.com/v2/tags", headers=headers, params=params)
            tag_id = None
            
            if resp.ok and resp.json().get("results", {}).get("tags"):
                tag_id = resp.json()["results"]["tags"][0]["id"]
                # Check if the tag is in a relevant category (avoid restaurant/place categories for music)
                if any(category in tag_id.lower() for category in ['music', 'genre', 'style', 'audience', 'character', 'theme', 'plot', 'subgenre']):
                    print(f"[QLOO TAGS] ✓ '{tag}' → {tag_id}")
                else:
                    # Tag found but in irrelevant category, try mapping
                    tag_id = None
            
            # If not found or in irrelevant category, try mappings
            if not tag_id:
                mapped_tag = None
                
                # Try emotional mapping first
                if tag in emotional_tag_mapping:
                    mapped_tag = emotional_tag_mapping[tag]
                elif tag in cultural_tag_mapping:
                    mapped_tag = cultural_tag_mapping[tag]
                elif tag in activity_tag_mapping:
                    mapped_tag = activity_tag_mapping[tag]
                
                if mapped_tag:
                    params = {"filter.query": mapped_tag, "limit": 1}
                    resp = requests.get("https://hackathon.api.qloo.com/v2/tags", headers=headers, params=params)
                    if resp.ok and resp.json().get("results", {}).get("tags"):
                        tag_id = resp.json()["results"]["tags"][0]["id"]
                        print(f"[QLOO TAGS] ✓ '{tag}' → '{mapped_tag}' → {tag_id}")
                    else:
                        print(f"[QLOO TAGS] ✗ '{tag}' (mapped to '{mapped_tag}') - not found in Qloo database")
                else:
                    print(f"[QLOO TAGS] ✗ '{tag}' - not found in Qloo database")
            
            if tag_id:
                tag_ids.append(tag_id)
                successful_tags.append(tag)
            
            time.sleep(0.2)
        except Exception as e:
            print(f"[QLOO TAGS] ✗ Error getting tag for '{tag}': {e}")
            continue
    
    print(f"[QLOO TAGS] Successfully converted {len(successful_tags)}/{len(tags)} tags")
    print(f"[QLOO TAGS] Successful tags: {successful_tags}")
    return tag_ids

def get_qloo_artist_recommendations(tag_ids, artists, tracks, qloo_api_key, limit=15, location=None, location_radius=None):
    """
    Get Qloo artist recommendations with optional location support
    
    Args:
        tag_ids: List of Qloo tag IDs
        artists: List of user's Spotify artists
        tracks: List of user's Spotify tracks
        qloo_api_key: Qloo API key
        limit: Number of recommendations to return
        location: Optional location query (e.g., "Mumbai", "New York", "London")
        location_radius: Optional radius in meters for location-based filtering
    """
    headers = {
        "X-API-KEY": qloo_api_key,
        "Content-Type": "application/json"
    }
    params = {
        "filter.type": "urn:entity:artist",
        "filter.tags": ",".join(tag_ids[:5]),  # Use exactly 5 most relevant tags
        "filter.popularity.min": 0.2,  # Lower threshold for more diverse artists
        "limit": limit * 2  # Get more artists to filter for relevance
    }
    
    # Add location-based signals if provided
    if location:
        params["signal.location.query"] = location
        if location_radius:
            params["signal.location.radius"] = location_radius
        print(f"[QLOO] Using location-based recommendations for: {location}")
    
    signals = []
    for artist in artists[:8]:
        signals.append(f"urn:entity:artist:{artist}")
    for track in tracks[:8]:
        signals.append(f"urn:entity:track:{track}")
    if signals:
        params["signals"] = ",".join(signals)
    
    try:
        resp = requests.get("https://hackathon.api.qloo.com/v2/insights", headers=headers, params=params)
        resp.raise_for_status()
        recommended_artists = []
        entities = resp.json().get("results", {}).get("entities", [])
        for entity in entities:
            if entity.get("subtype") == "urn:entity:artist":
                recommended_artists.append(entity.get("name"))
        
        if location:
            print(f"[QLOO] Found {len(recommended_artists)} location-based artist recommendations for {location}")
        else:
            print(f"[QLOO] Found {len(recommended_artists)} global artist recommendations")
            
        return recommended_artists
    except Exception as e:
        print(f"Error getting Qloo recommendations: {e}")
        return []

# --- Gemini/GPT-based Context Detection ---
def detect_context_type_llm(user_context, gemini_api_key):
    """Detect context type using LLM for better accuracy"""
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[CONTEXT] No valid Gemini API key provided, using fallback context detection")
        # Fallback to keyword-based detection
        context_lower = user_context.lower()
        if any(word in context_lower for word in ['dance', 'party', 'club', 'celebration']):
            return 'party'
        elif any(word in context_lower for word in ['workout', 'gym', 'exercise', 'fitness']):
            return 'workout'
        elif any(word in context_lower for word in ['study', 'work', 'focus', 'office']):
            return 'study'
        elif any(word in context_lower for word in ['drive', 'road', 'travel', 'commute']):
            return 'driving'
        elif any(word in context_lower for word in ['sleep', 'bed', 'relax', 'calm']):
            return 'sleep'
        elif any(word in context_lower for word in ['love', 'romance', 'date', 'romantic']):
            return 'romance'
        elif any(word in context_lower for word in ['sad', 'breakup', 'heartbreak', 'alone']):
            return 'breakup'
        else:
            return 'general'
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
Analyze this user context and determine the primary music context type: '{user_context}'

Choose from these context types:
- breakup: Sad, heartbreak, emotional, melancholic, alone, missing someone, crying, vulnerable, healing
- party: Celebration, upbeat, energetic, social gathering, dancing, fun, friends, clubbing, festive
- workout: Exercise, gym, running, high energy, motivational, fitness, training, sports
- study: Work, studying, focus, concentration, calm, ambient, office, productivity
- driving: Travel, road trip, commuting, upbeat, energetic, car, journey
- sleep: Bedtime, relaxation, calm, ambient, soothing, night, rest
- romance: Love, romantic, intimate, passionate, dating, relationship, couple
- celebration: Happy, festive, joyous, special occasion, birthday, wedding, success
- nostalgia: Remembering, past, memories, sentimental, old times, childhood
- meditation: Spiritual, calm, peaceful, zen, mindfulness, yoga, wellness

Consider the emotional state, activity, and social context. Return ONLY the context type (e.g., "breakup" or "party")
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        context_type = data['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        
        # Validate context type
        valid_contexts = ['breakup', 'party', 'workout', 'study', 'driving', 'sleep', 'romance', 'celebration', 'nostalgia', 'meditation']
        if context_type not in valid_contexts:
            # Try to map to closest valid context
            if any(word in context_type for word in ['sad', 'heart', 'break', 'alone', 'cry']):
                context_type = 'breakup'
            elif any(word in context_type for word in ['happy', 'celebrate', 'dance', 'fun', 'party']):
                context_type = 'party'
            elif any(word in context_type for word in ['work', 'study', 'focus', 'office']):
                context_type = 'study'
            elif any(word in context_type for word in ['exercise', 'gym', 'run', 'workout']):
                context_type = 'workout'
            else:
                context_type = 'general'
        
        print(f"[CONTEXT] Detected context type: {context_type}")
        return context_type
    except Exception as e:
        print(f"Error detecting context type: {e}")
        return "general"

# --- Spotify Track Search & Filtering ---
# get_spotify_artist_id function moved to spotify_client.py to avoid duplication

# get_spotify_artist_genres function moved to spotify_client.py to avoid duplication

# get_qloo_artist_id function moved to qloo_client.py to avoid duplication

@retry_on_failure(max_retries=3, delay=1)
def get_qloo_tag_id(tag_name, qloo_api_key):
    """Get Qloo tag ID by name"""
    try:
        url = "https://hackathon.api.qloo.com/v2/search"
        headers = {"X-API-KEY": qloo_api_key, "Content-Type": "application/json"}
        params = {"query": tag_name, "type": "tag"}
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        entities = data.get("results", {}).get("entities", [])
        
        if entities:
            return entities[0].get("entity_id", "").split(":")[-1]
        return None
    except Exception as e:
        print(f"Error getting Qloo tag ID for {tag_name}: {e}")
    return None

@retry_on_spotify_rate_limit()
def get_audio_features(track_ids, access_token):
    if not track_ids:
        return []
    chunks = [track_ids[i:i+100] for i in range(0, len(track_ids), 100)]
    all_features = []
    headers = {"Authorization": f"Bearer {access_token}"}
    for chunk in chunks:
        try:
            url = "https://api.spotify.com/v1/audio-features"
            params = {"ids": ",".join(chunk)}
            resp = requests.get(url, headers=headers, params=params)
            if resp.ok:
                features = resp.json().get("audio_features", [])
                all_features.extend(features)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error getting audio features: {e}")
            continue
    return all_features

def analyze_track_emotional_context(audio_features, track_name, artist_name):
    """Analyze emotional context of a track based on audio features and metadata"""
    if not audio_features:
        return "unknown"
    
    # Extract key audio features
    valence = audio_features.get('valence', 0.5)  # Happiness (0-1)
    energy = audio_features.get('energy', 0.5)    # Intensity (0-1)
    tempo = audio_features.get('tempo', 120)      # Speed (BPM)
    danceability = audio_features.get('danceability', 0.5)  # Dance-ability (0-1)
    acousticness = audio_features.get('acousticness', 0.5)  # Acoustic vs electronic (0-1)
    instrumentalness = audio_features.get('instrumentalness', 0.5)  # Instrumental vs vocal (0-1)
    
    # Analyze track name for emotional keywords
    track_name_lower = track_name.lower()
    emotional_keywords = {
        'sad': ['sad', 'cry', 'tears', 'heartbreak', 'lonely', 'alone', 'missing', 'lost', 'gone', 'broken', 'hurt', 'pain'],
        'happy': ['happy', 'joy', 'smile', 'laugh', 'fun', 'party', 'celebrate', 'dance', 'love', 'beautiful', 'wonderful'],
        'romantic': ['love', 'romance', 'heart', 'kiss', 'sweet', 'darling', 'baby', 'honey', 'forever', 'together'],
        'energetic': ['fire', 'burn', 'hot', 'wild', 'crazy', 'power', 'strong', 'fight', 'war', 'battle'],
        'calm': ['peace', 'calm', 'quiet', 'soft', 'gentle', 'smooth', 'easy', 'relax', 'sleep', 'dream']
    }
    
    # Check track name for emotional keywords
    detected_emotion = None
    for emotion, keywords in emotional_keywords.items():
        if any(keyword in track_name_lower for keyword in keywords):
            detected_emotion = emotion
            break
    
    # Analyze audio features for emotional context
    if valence < 0.3 and energy < 0.4:
        audio_emotion = "sad_melancholic"
    elif valence > 0.7 and energy > 0.6:
        audio_emotion = "happy_energetic"
    elif valence > 0.6 and danceability > 0.6:
        audio_emotion = "happy_danceable"
    elif acousticness > 0.7 and energy < 0.4:
        audio_emotion = "calm_acoustic"
    elif energy > 0.8 and tempo > 140:
        audio_emotion = "energetic_fast"
    elif valence < 0.4 and energy > 0.6:
        audio_emotion = "intense_emotional"
    else:
        audio_emotion = "neutral_balanced"
    
    # Combine track name analysis with audio features
    if detected_emotion:
        if detected_emotion == "sad" and audio_emotion in ["sad_melancholic", "calm_acoustic"]:
            final_emotion = "sad_melancholic"
        elif detected_emotion == "happy" and audio_emotion in ["happy_energetic", "happy_danceable"]:
            final_emotion = "happy_energetic"
        elif detected_emotion == "romantic" and audio_emotion in ["calm_acoustic", "neutral_balanced"]:
            final_emotion = "romantic_calm"
        else:
            final_emotion = f"{detected_emotion}_{audio_emotion}"
    else:
        final_emotion = audio_emotion
    
    return final_emotion

def filter_artists_by_mood(artists, context_type, access_token):
    """Filter a list of artist names by whether their genres match the current context/mood."""
    mood_genre_map = {
        'party': ['pop', 'dance', 'electronic', 'party', 'upbeat', 'bollywood', 'bhangra', 'club', 'hip hop', 'rap'],
        'breakup': ['sad', 'melancholic', 'emotional', 'ghazal', 'indian classical', 'sufi', 'ballad', 'acoustic', 'slow'],
        'workout': ['high energy', 'electronic', 'hip hop', 'pop', 'dance', 'rock'],
        'study': ['ambient', 'instrumental', 'acoustic', 'calm', 'chill', 'easy listening'],
        'driving': ['road trip', 'driving', 'upbeat', 'pop', 'rock', 'electronic'],
        'sleep': ['calm', 'ambient', 'soothing', 'chill', 'acoustic', 'instrumental'],
        'romantic': ['romantic', 'love', 'ballad', 'r&b', 'pop', 'acoustic'],
        'celebration': ['happy', 'festive', 'pop', 'dance', 'party', 'upbeat'],
        'nostalgia': ['nostalgic', 'retro', 'oldies', 'classic', 'indian classical', 'ghazal'],
        'meditation': ['calm', 'ambient', 'spiritual', 'sufi', 'instrumental', 'acoustic'],
        'general': []
    }
    target_genres = [g.lower() for g in mood_genre_map.get(context_type, [])]
    filtered = []
    for artist in artists:
        genres = get_spotify_artist_genres(artist, access_token)
        genres_lower = [g.lower() for g in genres]
        if any(tg in g for g in genres_lower for tg in target_genres):
            filtered.append(artist)
    return filtered

def get_artist_tracks_smart(artist_id, access_token, limit=15):
    headers = {"Authorization": f"Bearer {access_token}"}
    all_tracks = []
    try:
        # Get top tracks
        top_tracks_url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks"
        top_params = {"market": "US"}
        top_resp = requests.get(top_tracks_url, headers=headers, params=top_params, timeout=5)
        if top_resp.ok:
            top_tracks = top_resp.json().get("tracks", [])
            for track in top_tracks:
                album = track.get("album", {})
                album_name = album.get("name", "Unknown")
                album_images = album.get("images", [])
                album_art_url = album_images[0]["url"] if album_images else None
                release_date = album.get("release_date", "")
                release_year = release_date[:4] if release_date else None
                preview_url = track.get("preview_url")
                all_tracks.append({
                    "id": track["id"],
                    "name": track["name"],
                    "url": track["external_urls"]["spotify"],
                    "popularity": track.get("popularity", 0),
                    "type": "popular",
                    "artist": track["artists"][0]["name"] if track.get("artists") else "Unknown",
                    "album_name": album_name,
                    "album_art_url": album_art_url,
                    "release_year": release_year,
                    "preview_url": preview_url
                })

        # Get recent album tracks
        albums_url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
        albums_params = {"include_groups": "album,single", "limit": 5, "market": "US"}
        albums_resp = requests.get(albums_url, headers=headers, params=albums_params, timeout=5)
        if albums_resp.ok:
            albums = albums_resp.json().get("items", [])
            for album in albums:
                tracks_url = f"https://api.spotify.com/v1/albums/{album['id']}/tracks"
                tracks_params = {"limit": 10, "market": "US"}
                tracks_resp = requests.get(tracks_url, headers=headers, params=tracks_params, timeout=5)
                if tracks_resp.ok:
                    tracks = tracks_resp.json().get("items", [])
                    for track in tracks:
                        if not any(t["id"] == track["id"] for t in all_tracks):
                            # Fetch album details for these tracks
                            album_name = album.get("name", "Unknown")
                            album_images = album.get("images", [])
                            album_art_url = album_images[0]["url"] if album_images else None
                            release_date = album.get("release_date", "")
                            release_year = release_date[:4] if release_date else None
                            preview_url = track.get("preview_url")
                            all_tracks.append({
                                "id": track["id"],
                                "name": track["name"],
                                "url": track["external_urls"]["spotify"],
                                "popularity": 0,
                                "type": "album",
                                "artist": track["artists"][0]["name"] if track.get("artists") else "Unknown",
                                "album_name": album_name,
                                "album_art_url": album_art_url,
                                "release_year": release_year,
                                "preview_url": preview_url
                            })
                time.sleep(0.1)
    except Exception as e:
        print(f"Error getting tracks for artist: {e}")
    return all_tracks

# --- Trending/Contextual Fallbacks ---
def get_trending_tracks_for_context(context_type, access_token, limit=10):
    """Fetch trending or popular tracks from Spotify for a given context using dynamic search and caching."""
    
    # Generate cache key for this context
    cache_key = f"trending_tracks_{context_type}_{limit}"
    
    # Check if we have cached results
    cached_result = get_cached_recommendation("system", context_type, "trending_tracks")
    if cached_result and is_cache_valid(cached_result):
        print(f"[TRENDING] Using cached tracks for {context_type}")
        return cached_result['data'][:limit]
    
    # Apply rate limiting
    spotify_rate_limit()
    
    # Map context to search queries for dynamic playlist discovery
    context_to_search_queries = {
        "soccer": ["motivation", "workout", "energy", "sports"],
        "workout": ["workout", "gym", "fitness", "energy", "motivation"],
        "melancholic": ["sad", "melancholic", "emotional", "chill"],
        "relaxing": ["chill", "relaxing", "ambient", "peaceful"],
        "upbeat": ["upbeat", "happy", "energetic", "pop hits"],
        "driving": ["driving", "road trip", "rock", "classic rock"],
        "cooking": ["cooking", "kitchen", "chill", "background"],
        "studying": ["study", "focus", "instrumental", "ambient"],
        "party": ["party", "dance", "club", "upbeat"],
        "romantic": ["romantic", "love", "chill", "intimate"],
        "morning": ["morning", "wake up", "energetic", "positive"],
        "evening": ["evening", "sunset", "chill", "relaxing"],
        "night": ["night", "late night", "chill", "ambient"],
        "rainy": ["rainy", "rain", "chill", "melancholic"],
        "sunny": ["sunny", "summer", "happy", "upbeat"],
        "breakup": ["breakup", "sad", "emotional", "heartbreak"]
    }
    
    search_queries = context_to_search_queries.get(context_type, ["upbeat", "popular"])
    
    all_tracks = []
    playlist_ids_used = set()
    
    for query in search_queries:
        if len(all_tracks) >= limit:
            break
            
        try:
            # Search for playlists matching the query (reduced from 5 to 3)
            playlists = search_spotify_playlists(access_token, query, limit=3)
            
            for playlist in playlists:
                if len(all_tracks) >= limit:
                    break
                    
                playlist_id = playlist['id']
                if playlist_id in playlist_ids_used:
                    continue
                    
                playlist_ids_used.add(playlist_id)
                
                # Get tracks from this playlist
                url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
                headers = {"Authorization": f"Bearer {access_token}"}
                params = {"limit": min(limit, 20)}
                
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                for item in data.get("items", []):
                    if len(all_tracks) >= limit:
                        break
                        
                    track = item.get("track", {})
                    if not track:
                        continue
                        
                    album = track.get("album", {})
                    album_name = album.get("name", "Unknown")
                    album_images = album.get("images", [])
                    album_art_url = album_images[0]["url"] if album_images else None
                    release_date = album.get("release_date", "")
                    release_year = release_date[:4] if release_date else None
                    preview_url = track.get("preview_url")
                    
                    # Calculate context relevance score
                    context_score = calculate_context_relevance_score(track, context_type, query)
                    
                    track_info = {
                        "id": track.get("id"),
                        "name": track.get("name", "Unknown"),
                        "url": track.get("external_urls", {}).get("spotify", ""),
                        "popularity": track.get("popularity", 0),
                        "type": "trending",
                        "artist": track.get("artists", [{}])[0].get("name", "Unknown"),
                        "album_name": album_name,
                        "album_art_url": album_art_url,
                        "release_year": release_year,
                        "preview_url": preview_url,
                        "context_score": context_score,
                        "source_playlist": playlist['name'],
                        "search_query": query
                    }
                    
                    # Avoid duplicates
                    if not any(t['id'] == track_info['id'] for t in all_tracks):
                        all_tracks.append(track_info)
                
                # Add smaller delay between playlist requests
                time.sleep(0.02)
                
        except Exception as e:
            print(f"[TRENDING] Error searching for {query}: {e}")
            continue
    
    # If we still don't have enough tracks, try fallback to global top hits
    if len(all_tracks) < limit:
        try:
            fallback_playlists = search_spotify_playlists(access_token, "top hits", limit=3)
            for playlist in fallback_playlists:
                if len(all_tracks) >= limit:
                    break
                    
                playlist_id = playlist['id']
                if playlist_id in playlist_ids_used:
                    continue
                    
                url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
                headers = {"Authorization": f"Bearer {access_token}"}
                params = {"limit": min(limit - len(all_tracks), 10)}
                
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                for item in data.get("items", []):
                    if len(all_tracks) >= limit:
                        break
                        
                    track = item.get("track", {})
                    if not track:
                        continue
                        
                    album = track.get("album", {})
                    album_name = album.get("name", "Unknown")
                    album_images = album.get("images", [])
                    album_art_url = album_images[0]["url"] if album_images else None
                    release_date = album.get("release_date", "")
                    release_year = release_date[:4] if release_date else None
                    preview_url = track.get("preview_url")
                    
                    track_info = {
                        "id": track.get("id"),
                        "name": track.get("name", "Unknown"),
                        "url": track.get("external_urls", {}).get("spotify", ""),
                        "popularity": track.get("popularity", 0),
                        "type": "trending_fallback",
                        "artist": track.get("artists", [{}])[0].get("name", "Unknown"),
                        "album_name": album_name,
                        "album_art_url": album_art_url,
                        "release_year": release_year,
                        "preview_url": preview_url,
                        "context_score": 0.7,  # Lower score for fallback
                        "source_playlist": playlist['name'],
                        "search_query": "fallback"
                    }
                    
                    if not any(t['id'] == track_info['id'] for t in all_tracks):
                        all_tracks.append(track_info)
                        
        except Exception as e:
            print(f"[TRENDING] Fallback error: {e}")
    
    # Sort by context relevance and popularity
    all_tracks.sort(key=lambda x: (x['context_score'], x['popularity']), reverse=True)
    
    # Cache the results for 1 hour
    if all_tracks:
        cache_recommendation("system", context_type, all_tracks, "trending_tracks")
        print(f"[TRENDING] Cached {len(all_tracks)} tracks for {context_type}")
    
    print(f"[TRENDING] Successfully fetched {len(all_tracks)} trending tracks for {context_type}")
    return all_tracks[:limit]

def calculate_context_relevance_score(track, context_type, search_query):
    """Calculate how relevant a track is to the given context"""
    score = 0.5  # Base score
    
    # Boost based on popularity
    popularity = track.get("popularity", 0)
    score += (popularity / 100) * 0.3
    
    # Boost based on search query match
    track_name = track.get("name", "").lower()
    artist_name = track.get("artists", [{}])[0].get("name", "").lower()
    
    if search_query.lower() in track_name or search_query.lower() in artist_name:
        score += 0.2
    
    # Context-specific boosts
    context_boosts = {
        "workout": ["energy", "motivation", "gym", "fitness"],
        "relaxing": ["chill", "ambient", "peaceful", "calm"],
        "party": ["dance", "club", "upbeat", "energetic"],
        "studying": ["instrumental", "ambient", "focus", "study"],
        "romantic": ["love", "romantic", "intimate", "soft"]
    }
    
    if context_type in context_boosts:
        for keyword in context_boosts[context_type]:
            if keyword in track_name or keyword in artist_name:
                score += 0.1
                break
    
    return min(score, 1.0)  # Cap at 1.0

# --- Enhanced Scoring ---
def filter_tracks_by_context(tracks, access_token, context_type, user_artists=None):
    if not tracks:
        return []
    track_ids = [t["id"] for t in tracks]
    audio_features = get_audio_features(track_ids, access_token)
    filtered_tracks = []
    seen_artists = set()
    for i, track in enumerate(tracks):
        context_score = 0
        if i < len(audio_features) and audio_features[i]:
            features = audio_features[i]
            # Add more nuanced scoring
            if context_type == "soccer":
                context_score += features["energy"] * 2.0
                context_score += features["danceability"] * 1.5
                context_score += features["valence"] * 1.5
                context_score += min(features["tempo"] / 120, 1.5) * 1.2
                if features["loudness"] > -10:
                    context_score += 0.8
                if features["speechiness"] > 0.4:
                    context_score -= 0.5
                context_score += features.get("liveness", 0) * 0.5
            elif context_type == "workout":
                context_score += features["energy"] * 2.5
                context_score += features["danceability"] * 1.8
                context_score += min(features["tempo"] / 130, 1.8) * 1.5
                if features["loudness"] > -8:
                    context_score += 1.0
                context_score += features.get("liveness", 0) * 0.5
            elif context_type == "melancholic":
                if features["valence"] < 0.6:
                    context_score += 2 * (0.6 - features["valence"])
                if features["energy"] < 0.8:
                    context_score += 1 * (0.8 - features["energy"])
                context_score += features["acousticness"] * 1.5
                context_score += features["instrumentalness"] * 1.2
                context_score += (1 - features.get("liveness", 0)) * 0.5
            elif context_type == "breakup":
                if features["valence"] < 0.5:
                    context_score += 2.5 * (0.5 - features["valence"])
                if features["energy"] < 0.7:
                    context_score += 1.5 * (0.7 - features["energy"])
                context_score += features["acousticness"] * 1.8
                context_score += features["instrumentalness"] * 1.0
                if features["valence"] < 0.4:
                    context_score += 1.0  # Extra boost for very sad songs
                context_score += (1 - features.get("liveness", 0)) * 0.5
            elif context_type == "relaxing":
                context_score += (1 - features["energy"]) * 1.5
                context_score += features["acousticness"] * 1.8
                context_score += features["instrumentalness"] * 1.2
                if features["valence"] > 0.3 and features["valence"] < 0.7:
                    context_score += 1.0
                context_score += (1 - features.get("liveness", 0)) * 0.5
            elif context_type == "upbeat":
                context_score += features["valence"] * 2.0
                context_score += features["danceability"] * 1.8
                context_score += features["energy"] * 1.5
                context_score += features.get("liveness", 0) * 0.5
            elif context_type == "driving":
                context_score += features["energy"] * 1.8
                context_score += features["danceability"] * 1.2
                context_score += min(features["tempo"] / 140, 1.5) * 1.0
                if features["loudness"] > -12:
                    context_score += 0.6
                context_score += features.get("liveness", 0) * 0.3
            elif context_type == "cooking":
                context_score += features["valence"] * 1.5
                context_score += features["danceability"] * 1.2
                context_score += features["acousticness"] * 1.0
                if features["energy"] < 0.7:
                    context_score += 0.8
                context_score += (1 - features.get("liveness", 0)) * 0.3
            elif context_type == "studying":
                context_score += features["instrumentalness"] * 2.0
                context_score += (1 - features["energy"]) * 1.5
                context_score += features["acousticness"] * 1.2
                if features["speechiness"] < 0.1:
                    context_score += 1.0
                context_score += (1 - features.get("liveness", 0)) * 0.8
            elif context_type == "party":
                context_score += features["energy"] * 2.2
                context_score += features["danceability"] * 2.0
                context_score += features["valence"] * 1.8
                context_score += min(features["tempo"] / 130, 1.8) * 1.2
                if features["loudness"] > -10:
                    context_score += 0.8
            elif context_type == "romantic":
                context_score += features["valence"] * 1.8
                context_score += features["acousticness"] * 1.5
                context_score += (1 - features["energy"]) * 1.2
                if features["valence"] > 0.4 and features["valence"] < 0.8:
                    context_score += 1.0
                context_score += (1 - features.get("liveness", 0)) * 0.5
            elif context_type == "morning":
                context_score += features["valence"] * 1.8
                context_score += features["energy"] * 1.2
                context_score += features["danceability"] * 1.0
                if features["valence"] > 0.5:
                    context_score += 0.8
            elif context_type == "evening":
                context_score += features["valence"] * 1.2
                context_score += features["acousticness"] * 1.0
                context_score += (1 - features["energy"]) * 0.8
                if features["valence"] > 0.3 and features["valence"] < 0.7:
                    context_score += 0.6
            elif context_type == "night":
                context_score += features["energy"] * 1.5
                context_score += features["danceability"] * 1.3
                context_score += features["valence"] * 1.0
                if features["loudness"] > -12:
                    context_score += 0.5
            elif context_type == "rainy":
                context_score += (1 - features["valence"]) * 1.8
                context_score += features["acousticness"] * 1.5
                context_score += (1 - features["energy"]) * 1.2
                context_score += features["instrumentalness"] * 1.0
                context_score += (1 - features.get("liveness", 0)) * 0.5
            elif context_type == "sunny":
                context_score += features["valence"] * 2.0
                context_score += features["energy"] * 1.5
                context_score += features["danceability"] * 1.3
                if features["valence"] > 0.6:
                    context_score += 0.8
        else:
            context_score = 0.5
        if track["type"] == "popular":
            context_score += 0.3
        # Enhanced relevance scoring - reduce popularity weight and add user preference matching
        popularity_weight = 0.3  # Reduced from 0.5
        context_score += track.get("popularity", 0) / 100 * popularity_weight
        
        # User preference matching - check if track artist matches user's top artists
        track_artist = track.get("artist", "").lower()
        for user_artist in user_artists:
            user_artist_lower = user_artist.lower()
            if user_artist_lower in track_artist or track_artist in user_artist_lower:
                context_score += 1.5  # High boost for user's preferred artists
                break
        
        # Diversity: penalize if artist already seen
        artist_name = track.get("artist", "")
        if artist_name in seen_artists:
            context_score -= 0.7
        else:
            seen_artists.add(artist_name)
        
        track["context_score"] = round(context_score, 2)
        if context_score > 0.5:
            filtered_tracks.append(track)
    
    # Sort by enhanced context score (relevance)
    filtered_tracks.sort(key=lambda x: x["context_score"], reverse=True)
    return filtered_tracks

def get_context_fallback_artists(context_type, language_preference=None):
    """Get fallback artists based on context type and language preference"""
    
    # Language-specific fallback artists
    language_fallback_artists = {
        "english": {
            "upbeat": ['Dua Lipa', 'The Weeknd', 'Ariana Grande', 'Taylor Swift', 'Ed Sheeran', 'Bruno Mars', 'Calvin Harris', 'Martin Garrix', 'The Chainsmokers', 'Alan Walker'],
            "study": ['Ludovico Einaudi', 'Max Richter', 'Ólafur Arnalds', 'Nils Frahm', 'Brian Eno', 'Hammock', 'Stars of the Lid', 'Explosions in the Sky', 'This Will Destroy You', 'God Is An Astronaut'],
            "workout": ['Eminem', 'Kanye West', 'Drake', 'Post Malone', 'The Weeknd', 'Dua Lipa', 'Ariana Grande', 'Daft Punk', 'Calvin Harris', 'David Guetta'],
            "relaxing": ['Bon Iver', 'Norah Jones', 'Jack Johnson', 'Kings of Convenience', 'Zero 7', 'Thievery Corporation', 'Bonobo', 'Emancipator', 'Ólafur Arnalds', 'Max Richter'],
            "party": ['Dua Lipa', 'The Weeknd', 'Ariana Grande', 'Drake', 'Post Malone', 'Travis Scott', 'Cardi B', 'Megan Thee Stallion', 'Doja Cat', 'Lil Nas X'],
            "romantic": ['Ed Sheeran', 'John Legend', 'Adele', 'Sam Smith', 'Lewis Capaldi', 'James Arthur', 'Rag\'n\'Bone Man', 'Tom Odell', 'Passenger', 'Birdy']
        },
        "hindi": {
            "upbeat": ['Pritam', 'A.R. Rahman', 'Sachin-Jigar', 'Vishal-Shekhar', 'Shankar-Ehsaan-Loy', 'Amit Trivedi', 'Anirudh Ravichander', 'Badshah', 'Neha Kakkar', 'Arijit Singh'],
            "study": ['A.R. Rahman', 'Amit Trivedi', 'Vishal Bhardwaj', 'Shantanu Moitra', 'Pritam', 'Sachin-Jigar', 'Vishal-Shekhar', 'Shankar-Ehsaan-Loy', 'Anirudh Ravichander', 'Ritviz'],
            "workout": ['Badshah', 'Neha Kakkar', 'Arijit Singh', 'Pritam', 'A.R. Rahman', 'Sachin-Jigar', 'Vishal-Shekhar', 'Shankar-Ehsaan-Loy', 'Amit Trivedi', 'Anirudh Ravichander'],
            "relaxing": ['A.R. Rahman', 'Amit Trivedi', 'Vishal Bhardwaj', 'Shantanu Moitra', 'Pritam', 'Sachin-Jigar', 'Vishal-Shekhar', 'Shankar-Ehsaan-Loy', 'Anirudh Ravichander', 'Ritviz'],
            "party": ['Badshah', 'Neha Kakkar', 'Arijit Singh', 'Pritam', 'A.R. Rahman', 'Sachin-Jigar', 'Vishal-Shekhar', 'Shankar-Ehsaan-Loy', 'Amit Trivedi', 'Anirudh Ravichander'],
            "romantic": ['Arijit Singh', 'Atif Aslam', 'Pritam', 'A.R. Rahman', 'Sachin-Jigar', 'Vishal-Shekhar', 'Shankar-Ehsaan-Loy', 'Amit Trivedi', 'Anirudh Ravichander', 'Ritviz']
        }
    }
    
    # If language preference is specified, use language-specific artists
    if language_preference and language_preference.get('primary_language') in language_fallback_artists:
        language = language_preference['primary_language']
        if context_type in language_fallback_artists[language]:
            print(f"[FALLBACK] Using {language}-specific fallback artists for {context_type}")
            return language_fallback_artists[language][context_type]
    
    # Default fallback artists (mostly English)
    fallback_artists = {
        "soccer": [
            'Eminem', 'Imagine Dragons', 'OneRepublic', 'Fall Out Boy', 
            'Linkin Park', 'The Killers', 'Foo Fighters', 'Queen', 
            'AC/DC', 'Survivor', 'Eye of the Tiger', 'We Will Rock You'
        ],
        "workout": [
            'Eminem', 'Kanye West', 'Drake', 'Post Malone', 
            'The Weeknd', 'Dua Lipa', 'Ariana Grande', 'Daft Punk',
            'Calvin Harris', 'David Guetta'
        ],
        "melancholic": [
            'Bon Iver', 'Iron & Wine', 'Damien Rice', 'Elliott Smith', 
            'John Mayer', 'Jack Johnson', 'Mumford & Sons', 'Simon & Garfunkel',
            'The National', 'Radiohead'
        ],
        "relaxing": [
            'Bon Iver', 'Norah Jones', 'Jack Johnson', 'Kings of Convenience',
            'Zero 7', 'Thievery Corporation', 'Bonobo', 'Emancipator',
            'Ólafur Arnalds', 'Max Richter'
        ],
        "upbeat": [
            'Dua Lipa', 'The Weeknd', 'Ariana Grande', 'Taylor Swift',
            'Ed Sheeran', 'Bruno Mars', 'Daft Punk', 'Calvin Harris',
            'Pharrell Williams', 'Justin Timberlake'
        ],
        "driving": [
            'The Killers', 'Foo Fighters', 'Red Hot Chili Peppers', 'Green Day',
            'Blink-182', 'Sum 41', 'Good Charlotte', 'Simple Plan',
            'Fall Out Boy', 'Panic! At The Disco'
        ],
        "cooking": [
            'Norah Jones', 'Jack Johnson', 'John Mayer', 'Jason Mraz',
            'Colbie Caillat', 'Sara Bareilles', 'Ingrid Michaelson',
            'Regina Spektor', 'Feist', 'The Weepies'
        ],
        "studying": [
            'Ludovico Einaudi', 'Max Richter', 'Ólafur Arnalds', 'Nils Frahm',
            'Brian Eno', 'Hammock', 'Stars of the Lid', 'Explosions in the Sky',
            'This Will Destroy You', 'God Is An Astronaut'
        ],
        "party": [
            'Dua Lipa', 'The Weeknd', 'Ariana Grande', 'Drake',
            'Post Malone', 'Travis Scott', 'Cardi B', 'Megan Thee Stallion',
            'Doja Cat', 'Lil Nas X'
        ],
        "romantic": [
            'Ed Sheeran', 'John Legend', 'Adele', 'Sam Smith',
            'Lewis Capaldi', 'James Arthur', 'Rag\'n\'Bone Man',
            'Tom Odell', 'Passenger', 'Birdy'
        ],
        "morning": [
            'Coldplay', 'U2', 'The Fray', 'OneRepublic',
            'Imagine Dragons', 'The Script', 'Snow Patrol', 'Keane',
            'Travis', 'Embrace'
        ],
        "evening": [
            'The xx', 'London Grammar', 'Halsey', 'Lorde',
            'Billie Eilish', 'Clairo', 'Phoebe Bridgers', 'Lucy Dacus',
            'Julien Baker', 'boygenius'
        ],
        "night": [
            'The Weeknd', 'Drake', 'Post Malone', 'Travis Scott',
            'Lil Uzi Vert', 'Playboi Carti', 'Young Thug', 'Future',
            'Migos', '21 Savage'
        ],
        "rainy": [
            'Bon Iver', 'Iron & Wine', 'Damien Rice', 'Elliott Smith',
            'Nick Drake', 'Jeff Buckley', 'Eva Cassidy', 'Nina Simone',
            'Billie Holiday', 'Ella Fitzgerald'
        ],
        "sunny": [
            'Vampire Weekend', 'Phoenix', 'MGMT', 'Tame Impala',
            'Foster The People', 'Passion Pit', 'Two Door Cinema Club',
            'The Wombats', 'The Kooks', 'Arctic Monkeys'
        ]
    }
    return fallback_artists.get(context_type, fallback_artists["upbeat"])

# --- Spotify Playlist Creation ---
# create_spotify_playlist function moved to spotify_client.py to avoid duplication

def add_tracks_to_playlist(playlist_id, track_uris, access_token):
    """Add tracks to an existing playlist"""
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "uris": track_uris
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error adding tracks to playlist: {e}")
        return False

# get_spotify_track_uri function moved to spotify_client.py to avoid duplication

@app.route('/create-playlist', methods=['POST', 'OPTIONS'])
def create_playlist():
    """Create a new Spotify playlist with selected tracks"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        track_urls = data.get('track_uris', [])  # These are actually URLs, not URIs
        spotify_token = data.get('spotify_token')
        
        if not name or not spotify_token:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Get user ID
        user_id = get_spotify_user_id(spotify_token)
        if not user_id:
            return jsonify({'success': False, 'error': 'Invalid Spotify token'}), 401
        
        # Create playlist
        playlist_id = create_spotify_playlist(user_id, name, description, spotify_token)
        if not playlist_id:
            return jsonify({'success': False, 'error': 'Failed to create playlist. Please ensure you have granted playlist creation permissions.'}), 500
        
        # Convert track URLs to URIs
        spotify_track_uris = []
        for track_url in track_urls:
            track_uri = get_spotify_track_uri(track_url, spotify_token)
            if track_uri:
                spotify_track_uris.append(track_uri)
            else:
                print(f"Could not extract track URI from: {track_url}")
        
        # Add tracks to playlist
        if spotify_track_uris:
            success = add_tracks_to_playlist(playlist_id, spotify_track_uris, spotify_token)
            if not success:
                return jsonify({'success': False, 'error': 'Failed to add tracks to playlist'}), 500
        
        return jsonify({
            'success': True, 
            'playlist_id': playlist_id,
            'message': f'Playlist "{name}" created successfully'
        })
        
    except Exception as e:
        print(f"Error in create_playlist: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/spotify-profile', methods=['POST', 'OPTIONS'])
def spotify_profile():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    try:
        data = request.get_json()
        spotify_token = data.get('spotify_token')
        if not spotify_token:
            return jsonify({'error': 'Missing spotify_token'}), 400
        url = 'https://api.spotify.com/v1/me'
        headers = {'Authorization': f'Bearer {spotify_token}'}
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print("Spotify error response:", resp.text)  # Log the full error
            return jsonify({'error': 'Failed to fetch profile', 'status': resp.status_code, 'spotify_error': resp.text}), resp.status_code
        profile = resp.json()
        # Only return relevant fields
        return jsonify({
            'id': profile.get('id'),
            'display_name': profile.get('display_name'),
            'images': profile.get('images', []),
            'email': profile.get('email'),
            'country': profile.get('country'),
            'product': profile.get('product'),
            'uri': profile.get('uri'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('test_frontend.html')

@app.route('/spotify-auth-url', methods=['GET'])
def spotify_auth_url():
    client_id = request.args.get('client_id', DEFAULT_CLIENT_ID)
    redirect_uri = request.args.get('redirect_uri', DEFAULT_REDIRECT_URI)
    scope = request.args.get('scope', DEFAULT_SCOPE)
    
    # Add parameters to force re-authentication
    import secrets
    unique_state = secrets.token_urlsafe(32)
    
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": unique_state,
        "show_dialog": "true",  # Force Spotify to show the authorization dialog
    }
    url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)
    return jsonify({"auth_url": url})

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if code:
        return f'''<html><head><meta http-equiv="refresh" content="0; url=/?code={code}" /></head><body>
        <p>Redirecting... If you are not redirected, <a href="/?code={code}">click here</a>.</p></body></html>'''
    return "<h3>No code found in URL. Please try connecting Spotify again.</h3>"

@app.route('/exchange-token', methods=['POST', 'OPTIONS'])
def exchange_token():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    code = data.get('code')
    client_id = data.get('client_id', DEFAULT_CLIENT_ID)
    client_secret = data.get('client_secret', DEFAULT_CLIENT_SECRET)
    redirect_uri = data.get('redirect_uri', DEFAULT_REDIRECT_URI)
    try:
        token_response = exchange_code_for_token(code, client_id, client_secret, redirect_uri)
        if token_response and 'access_token' in token_response:
            # Store tokens for future refresh
            user_id = get_spotify_user_id(token_response['access_token'])
            if user_id:
                store_user_tokens(user_id, token_response['access_token'], 
                                token_response.get('refresh_token'), 
                                token_response.get('expires_in', 3600))
            
            return jsonify({
                'access_token': token_response['access_token'],
                'expires_in': token_response.get('expires_in', 3600)
            })
        else:
            return jsonify({'error': 'Failed to exchange code for token'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/refresh-token', methods=['POST', 'OPTIONS'])
def refresh_token():
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    user_id = data.get('user_id')
    client_id = data.get('client_id', DEFAULT_CLIENT_ID)
    client_secret = data.get('client_secret', DEFAULT_CLIENT_SECRET)
    
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    
    try:
        # In a real implementation, you'd retrieve the refresh token from database
        # For now, we'll return an error indicating manual re-auth is needed
        return jsonify({
            'error': 'Token refresh not implemented. Please reconnect your Spotify account.',
            'requires_reauth': True
        }), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/logout', methods=['POST', 'OPTIONS'])
def logout():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    access_token = data.get('access_token')
    client_id = data.get('client_id', DEFAULT_CLIENT_ID)
    client_secret = data.get('client_secret', DEFAULT_CLIENT_SECRET)
    
    if not access_token:
        return jsonify({'error': 'Missing access_token'}), 400
    
    try:
        # Check token validity and force re-auth
        revoke_spotify_token(access_token, client_id, client_secret)
        
        # Generate a unique state parameter to force new OAuth flow
        unique_state = force_spotify_reauth()
        
        return jsonify({
            'success': True, 
            'message': 'Successfully logged out',
            'force_reauth': True,
            'unique_state': unique_state
        })
    except Exception as e:
        # Even if revocation fails, we still return success for logout
        return jsonify({
            'success': True, 
            'message': 'Logged out (token revocation may have failed)',
            'force_reauth': True
        })

@app.route('/musicrecommandation', methods=['POST'])
def music_recommendation():
    try:
        # Clean up expired cache entries periodically
        cleanup_expired_cache()
        
        data = request.get_json()
        user_context = data.get('user_context', '')
        spotify_token = data.get('spotify_token', '')
        gemini_api_key = data.get('gemini_api_key', DEFAULT_GEMINI_API_KEY)
        qloo_api_key = data.get('qloo_api_key', DEFAULT_QLOO_API_KEY)
        client_id = data.get('client_id', DEFAULT_CLIENT_ID)
        client_secret = data.get('client_secret', DEFAULT_CLIENT_SECRET)
        redirect_uri = data.get('redirect_uri', DEFAULT_REDIRECT_URI)
        
        # Input validation
        if not user_context or not spotify_token:
            return jsonify({'error': 'Missing required parameters: user_context and spotify_token'}), 400
        
        if not gemini_api_key or not qloo_api_key:
            return jsonify({'error': 'Missing API keys: gemini_api_key and qloo_api_key'}), 400
        
        # Get user ID first for caching
        access_token = spotify_token
        user_id = get_spotify_user_id(access_token)
        if not user_id:
            return jsonify({'error': 'Could not fetch Spotify user id - invalid token or network error'}), 400
        
        # Check cache first
        cached_result = get_cached_recommendation(user_id, user_context, "music")
        if cached_result:
            # Add cache indicator to response
            cached_result['from_cache'] = True
            cached_result['cache_timestamp'] = datetime.now().isoformat()
            return jsonify(cached_result)
        
        # Get user's country and convert to location
        try:
            user_profile = get_spotify_user_profile(access_token)
            user_country = user_profile.get('country', 'US') if user_profile else 'US'
            location = get_location_from_country(user_country)
            print(f"[LOCATION] User country: {user_country} -> Location: {location}")
        except Exception as e:
            print(f"Error getting user location: {e}")
            location = "New York"  # Default fallback
            user_country = "US"
        
        # ENHANCED: Use enhanced context detection with mood and language preference
        enhanced_context = enhance_context_detection_with_mood_and_language(user_context, gemini_api_key)
        context_type = enhanced_context['context_type']
        language_preference = enhanced_context['language_preference']
        mood_preference = enhanced_context['mood_preference']
        
        print(f"[ENHANCED CONTEXT] Context: {context_type}, Mood: {mood_preference['primary_mood']}, Language: {language_preference['primary_language']}")
        print(f"[ENHANCED CONTEXT] Activity: {mood_preference['activity_type']}, Energy: {mood_preference['energy_level']}")
        
        # ENHANCED: Use enhanced Gemini for better tag generation
        enhanced_tags = call_gemini_for_enhanced_tags(user_context, gemini_api_key, user_country, location)
        if not enhanced_tags:
            print("Warning: No enhanced tags generated from Gemini, using fallback tags")
            enhanced_tags = ["upbeat", "energetic"]  # Fallback tags
        
        print(f"[ENHANCED] Generated {len(enhanced_tags)} enhanced tags: {enhanced_tags}")
        
        # ENHANCED: Generate cultural context
        cultural_context = generate_cultural_context(user_context, user_country, location, gemini_api_key)
        
        # ENHANCED: Get user Spotify data with more artists/tracks - OPTIMIZED
        try:
            # Fetch user data in parallel for faster response
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit all data fetching tasks in parallel with reduced workers to avoid rate limiting
                user_artists_future = executor.submit(get_spotify_top_artists, access_token, 10, "long_term")
                user_tracks_future = executor.submit(get_spotify_top_tracks, access_token, 10, "long_term")
                user_profile_future = executor.submit(get_spotify_user_profile, access_token)
                
                # Get results
                user_artists = user_artists_future.result()
                user_tracks = user_tracks_future.result()
                user_profile = user_profile_future.result()
                
            print(f"Got {len(user_artists)} user artists and {len(user_tracks)} user tracks (long_term) - PARALLEL")
        except Exception as e:
            print(f"Error fetching user Spotify data: {e}")
            user_artists = []
            user_tracks = []
            user_profile = None
        
        # ENHANCED: Filter and rank tags to get 5 most relevant tags for music
        filtered_tags = filter_and_rank_tags_for_music(enhanced_tags, user_context, user_country, location, gemini_api_key)
        print(f"[TAG FILTERING] Filtered from {len(enhanced_tags)} to {len(filtered_tags)} most relevant tags")
        
        # ENHANCED: Get Qloo tag IDs for filtered tags (exactly 5 tags)
        tag_ids = get_qloo_tag_ids(filtered_tags, qloo_api_key)
        if not tag_ids:
            print("Warning: No Qloo tag IDs found, using fallback approach")
        else:
            print(f"Found {len(tag_ids)} Qloo tag IDs from {len(filtered_tags)} filtered tags")
        
        # ENHANCED: Use enhanced Qloo client for better recommendations - OPTIMIZED
        qloo_client = EnhancedQlooClient(qloo_api_key)
        
        # Get enhanced recommendations with cultural intelligence - OPTIMIZED
        enhanced_recommendations = qloo_client.get_enhanced_recommendations(
            tag_ids=tag_ids,
            user_artists=user_artists[:8],  # Reduced from full list to 8 for speed
            user_tracks=user_tracks[:8],    # Reduced from full list to 8 for speed
            location=location,
            location_radius=5000000,
            cultural_context=cultural_context,
            limit=15  # Reduced from 25 to 15 for speed
        )
        
        # Get cultural insights (with error handling) - OPTIMIZED
        try:
            cultural_insights = qloo_client.get_cultural_insights(location, "music")
        except Exception as e:
            print(f"[WARNING] Cultural insights failed: {e}")
            cultural_insights = []
        
        # Extract artist names from enhanced recommendations - OPTIMIZED for speed
        qloo_reco_artists = [artist["name"] for artist in enhanced_recommendations[:15]]  # Reduced from 20 to 15 for speed
        
        print(f"[ENHANCED] Got {len(qloo_reco_artists)} enhanced Qloo recommended artists (OPTIMIZED)")
        
        # FAST LANGUAGE FILTERING: Use known artist lists for speed
        if language_preference and language_preference.get('primary_language') != 'any':
            print(f"[FAST LANGUAGE FILTER] Applying quick filter to {len(qloo_reco_artists)} artists")
            primary_language = language_preference['primary_language']
            
            # Known artist lists for fast filtering
            known_english_artists = {
                'martin garrix', 'the chainsmokers', 'alan walker', 'marshmello', 'dj snake', 
                'major lazer', 'calvin harris', 'david guetta', 'avicii', 'skrillex',
                'zedd', 'kygo', 'the weeknd', 'ed sheeran', 'taylor swift', 'justin bieber',
                'ariana grande', 'post malone', 'dua lipa', 'billie eilish', 'harry styles',
                'coldplay', 'imagine dragons', 'maroon 5', 'one republic', 'twenty one pilots',
                'pali', 'emiway bantai', 'kana-boon', 'weaver', 'uverworld', 'asian kung-fu generation',
                'amazarashi', 'supercell', 'hello sleepwalkers', 'scandal', 'aqua timez',
                'unison square garden', 'l\'arc-en-ciel', 'tk from ling tosite sigure'
            }
            
            known_hindi_artists = {
                'pritam', 'atif aslam', 'a.r. rahman', 'anuv jain', 'ritviz', 'arijit singh',
                'sachin-jigar', 'mohit lalwani', 'shankar-ehsaan-loy', 'mohit chauhan',
                'neha kakkar', 'badshah', 'karan aujla', 'amit trivedi', 'vishal-shekhar',
                'jatin-lalit', 'kailash kher', 'benny dayal', 'sunidhi chauhan', 'shreya ghoshal',
                'armaan malik', 'harrdy sandhu', 'shaan', 'vishal dadlani', 'shankar mahadevan',
                'tony kakkar', 'vishal mishra', 'shekhar ravjiani', 'palak muchhal', 'monali thakur',
                'anushka manchanda', 'jonita gandhi', 'shalmali kholgade', 'lata mangeshkar'
            }
            
            # Fast filtering using known lists
            filtered_artists = []
            for artist in qloo_reco_artists:
                artist_lower = artist.lower()
                if primary_language == 'english':
                    if artist_lower in known_english_artists:
                        filtered_artists.append(artist)
                        print(f"[FAST FILTER] ✓ {artist} - Known English artist")
                    elif artist_lower not in known_hindi_artists:
                        # If not known Hindi, include it (conservative approach)
                        filtered_artists.append(artist)
                        print(f"[FAST FILTER] ? {artist} - Unknown, including for English")
                elif primary_language == 'hindi':
                    if artist_lower in known_hindi_artists:
                        filtered_artists.append(artist)
                        print(f"[FAST FILTER] ✓ {artist} - Known Hindi artist")
                    elif artist_lower not in known_english_artists:
                        # If not known English, include it (conservative approach)
                        filtered_artists.append(artist)
                        print(f"[FAST FILTER] ? {artist} - Unknown, including for Hindi")
            
            if filtered_artists:
                qloo_reco_artists = filtered_artists
                print(f"[FAST LANGUAGE FILTER] Filtered to {len(qloo_reco_artists)} artists in 0.1s")
            else:
                print(f"[FAST LANGUAGE FILTER] No artists matched, using fallback")
                qloo_reco_artists = get_context_fallback_artists("upbeat", language_preference)
        
        # Use context-appropriate fallback artists if Qloo doesn't return enough
        if len(qloo_reco_artists) < 10:
            fallback_artists = get_context_fallback_artists("upbeat", language_preference)  # Default to upbeat
            # Fallback artists are already language-filtered by the function
            qloo_reco_artists.extend(fallback_artists)
            print(f"Added {len(fallback_artists)} fallback artists")
        
        qloo_reco_artists = list(set(qloo_reco_artists))  # Remove duplicates
        print(f"Final artist list has {len(qloo_reco_artists)} unique language-appropriate artists")
        
        playlist = []
        # ENHANCED: Detect and validate context type from user input
        detected_context = detect_context_type_llm(user_context, gemini_api_key)
        print(f"[CONTEXT] Detected context type: {detected_context}")
        
        # Validate the detected context
        validated_context = validate_context_detection(user_context, detected_context, gemini_api_key)
        if validated_context != detected_context:
            detected_context = validated_context
            print(f"[CONTEXT] Context corrected to: {detected_context}")
        
        # ENHANCED: If user data is sparse, use trending/contextual tracks as fallback
        if not user_artists and not user_tracks:
            print("[INFO] Using trending/contextual fallback tracks (no user Spotify data available)")
            trending_tracks = get_trending_tracks_for_context(detected_context, access_token, limit=15)
            playlist.extend(trending_tracks)
            print(f"[TRENDING] Added {len(trending_tracks)} trending tracks")
        else:
            print("[INFO] Using enhanced Qloo-based recommendations (user Spotify data found)")
            seen_tracks = set()  # Track duplicate prevention
            seen_artists = set()  # Artist diversity tracking
            
            # Shuffle artists to prevent repetition patterns
            import random
            shuffled_artists = qloo_reco_artists[:15]  # Reduced from 25 to 15 artists for speed
            random.shuffle(shuffled_artists)
            
            # ENHANCED: Get enhanced user preferences for personalization (FAST MODE - no mood filtering)
            user_preferences = get_enhanced_user_preferences(access_token, detected_context, language_preference, None)

            # INCREASED: Process more artists for better variety
            for artist_name in shuffled_artists[:15]:  # Increased from 10 to 15 artists
                if len(playlist) >= 35:  # Increased from 20 to 35 for more tracks
                    break
                
                artist_id = get_spotify_artist_id(artist_name, access_token)
                if artist_id:
                    artist_tracks = get_artist_tracks_smart(artist_id, access_token, limit=8)  # Reduced from 20 to 8
                    if artist_tracks:
                        # Skip AI filtering for speed - use direct selection
                        context_tracks = artist_tracks[:6]  # Reduced from 12 to 6
                        print(f"[ARTIST] {artist_name}: got {len(artist_tracks)} tracks, {len(context_tracks)} selected (FAST MODE)")
                        
                        tracks_added = 0
                        for track in context_tracks:
                            if tracks_added >= 3:  # Increased from 2 to 3 tracks per artist
                                break
                            if len(playlist) >= 40:  # Increased limit from 30 to 40
                                break
                            
                            # Add personalization score based on user preferences
                            personalization_score = 0
                            if user_preferences and user_preferences.get('favorite_artists'):
                                if artist_name in user_preferences.get('favorite_artists', []):
                                    personalization_score = 2.0
                                    print(f"[PERSONALIZATION] {artist_name} is in user's top 10 favorites - Score: {personalization_score}")
                            
                            # FAST MODE: Skip expensive language checks - already filtered at artist level
                            
                            # Get track genres and emotional context analysis
                            track_genres = []
                            try:
                                if track.get('id'):
                                    # Get audio features for the track
                                    audio_features = get_audio_features([track['id']], access_token)
                                    if audio_features and len(audio_features) > 0:
                                        features = audio_features[0]
                                        # Analyze emotional context from audio features
                                        emotional_context = analyze_track_emotional_context(features, track.get('name', ''), artist_name)
                                        track["emotional_context"] = emotional_context
                                        track["audio_features"] = features
                                        
                                        # Get artist genres for additional context
                                        artist_genres = get_spotify_artist_genres(artist_name, access_token)
                                        if artist_genres:
                                            track["artist_genres"] = artist_genres
                                            track["primary_genre"] = artist_genres[0] if artist_genres else "unknown"
                                        
                                        print(f"[EMOTIONAL CONTEXT] {track.get('name', 'Unknown')} - {emotional_context} | Genres: {track.get('primary_genre', 'unknown')}")
                            except Exception as e:
                                print(f"[GENRE ANALYSIS] Error analyzing track {track.get('name', 'Unknown')}: {e}")
                            
                            track["personalization_score"] = personalization_score
                            track["context_score"] = 1.0 + personalization_score
                            
                            playlist.append(track)
                            tracks_added += 1
                        
                        print(f"[ARTIST] {artist_name}: added {tracks_added} tracks")
        
        # Ensure we have tracks - fallback if playlist is empty
        if len(playlist) == 0:
            print(f"[WARNING] No tracks found, using {detected_context} fallback")
            try:
                playlist = get_trending_tracks_for_context(detected_context, access_token, limit=15)
                print(f"[FALLBACK] Got {len(playlist)} trending tracks")
            except Exception as e:
                print(f"[ERROR] Fallback also failed: {e}")
                # Final fallback - create context-appropriate tracks
                if detected_context in ["melancholic", "sad", "heartbreak"]:
                    playlist = [
                        {
                            "name": "Someone Like You",
                            "artist": "Adele",
                            "url": "https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV",
                            "context_score": 1.0,
                            "album_name": "21",
                            "album_art_url": "https://i.scdn.co/image/ab67616d0000b273c8c4bb14fd21f320a8e2f8fd",
                            "release_year": "2011",
                            "preview_url": None
                        },
                        {
                            "name": "All of Me",
                            "artist": "John Legend",
                            "url": "https://open.spotify.com/track/3U4isOIWM3VvDubwSI3y7a",
                            "context_score": 1.0,
                            "album_name": "Love in the Future",
                            "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                            "release_year": "2013",
                            "preview_url": None
                        }
                    ]
                else:
                    playlist = [
                        {
                            "name": "Party Rock Anthem",
                            "artist": "LMFAO",
                            "url": "https://open.spotify.com/track/0IkKz2J93C94Ei4BvDop7P",
                            "context_score": 1.0,
                            "album_name": "Sorry for Party Rocking",
                            "album_art_url": "https://i.scdn.co/image/ab67616d0000b273c8c4bb14fd21f320a8e2f8fd",
                            "release_year": "2011",
                            "preview_url": None
                        },
                        {
                            "name": "Uptown Funk",
                            "artist": "Mark Ronson ft. Bruno Mars",
                            "url": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS",
                            "context_score": 1.0,
                            "album_name": "Uptown Special",
                            "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                            "release_year": "2014",
                            "preview_url": None
                        }
                    ]
        
        # Deduplicate tracks to prevent repeats
        unique_playlist = []
        seen_track_ids = set()
        seen_track_names = set()
        
        for track in playlist:
            # Try to get track ID first (most reliable)
            track_id = track.get('id') or track.get('spotify_id')
            track_name = track.get('name', '').lower().strip()
            track_artist = track.get('artist', '').lower().strip()
            
            # Create a unique identifier
            if track_id:
                unique_id = track_id
            else:
                # Fallback to name + artist combination
                unique_id = f"{track_name}_{track_artist}"
            
            # Check if we've seen this track before
            if unique_id not in seen_track_ids and track_name not in seen_track_names:
                unique_playlist.append(track)
                seen_track_ids.add(unique_id)
                seen_track_names.add(track_name)
            else:
                print(f"[DEDUPLICATION] Removed duplicate track: {track_name} by {track_artist}")
        
        playlist = unique_playlist
        print(f"[DEBUG] Final playlist length: {len(playlist)} (after deduplication)")
        if len(playlist) == 0:
            print("[CRITICAL ERROR] Playlist is still empty after all fallbacks!")
            print("[FALLBACK] Using hardcoded popular tracks as last resort...")
            
            # Hardcoded popular tracks as ultimate fallback
            fallback_tracks = [
                {
                    "id": "4iV5W9uYEdYUVa79Axb7Rh",
                    "name": "Shape of You",
                    "artist": "Ed Sheeran",
                    "url": "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh",
                    "context_score": 1.0,
                    "album_name": "÷ (Divide)",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                    "release_year": "2017",
                    "preview_url": None,
                    "type": "fallback"
                },
                {
                    "id": "32OlwWuMpZ6b0aN2RZOeMS",
                    "name": "Uptown Funk",
                    "artist": "Mark Ronson ft. Bruno Mars",
                    "url": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS",
                    "context_score": 1.0,
                    "album_name": "Uptown Special",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                    "release_year": "2014",
                    "preview_url": None,
                    "type": "fallback"
                },
                {
                    "id": "0V3wPSX9ygBnCm8psKIegu",
                    "name": "Blinding Lights",
                    "artist": "The Weeknd",
                    "url": "https://open.spotify.com/track/0V3wPSX9ygBnCm8psKIegu",
                    "context_score": 1.0,
                    "album_name": "After Hours",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                    "release_year": "2020",
                    "preview_url": None,
                    "type": "fallback"
                },
                {
                    "id": "5QO79kh1waicV47BqGRL3g",
                    "name": "Bad Guy",
                    "artist": "Billie Eilish",
                    "url": "https://open.spotify.com/track/5QO79kh1waicV47BqGRL3g",
                    "context_score": 1.0,
                    "album_name": "WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                    "release_year": "2019",
                    "preview_url": None,
                    "type": "fallback"
                },
                {
                    "id": "3CRDbSIZ4r5MsZ0YwxuEkn",
                    "name": "Stressed Out",
                    "artist": "Twenty One Pilots",
                    "url": "https://open.spotify.com/track/3CRDbSIZ4r5MsZ0YwxuEkn",
                    "context_score": 1.0,
                    "album_name": "Blurryface",
                    "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                    "release_year": "2015",
                    "preview_url": None,
                    "type": "fallback"
                }
            ]
            
            # Add context-specific tracks for breakup
            if detected_context == "breakup" or "breakup" in user_context.lower():
                breakup_tracks = [
                    {
                        "id": "4h9wh7iOZ0GGq8iJuPEVCf",
                        "name": "Someone Like You",
                        "artist": "Adele",
                        "url": "https://open.spotify.com/track/4h9wh7iOZ0GGq8iJuPEVCf",
                        "context_score": 1.5,
                        "album_name": "21",
                        "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                        "release_year": "2011",
                        "preview_url": None,
                        "type": "fallback_breakup"
                    },
                    {
                        "id": "6Qcn6T7lPZbqXg5vzLpCDJ",
                        "name": "All of Me",
                        "artist": "John Legend",
                        "url": "https://open.spotify.com/track/6Qcn6T7lPZbqXg5vzLpCDJ",
                        "context_score": 1.3,
                        "album_name": "Love in the Future",
                        "album_art_url": "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
                        "release_year": "2013",
                        "preview_url": None,
                        "type": "fallback_breakup"
                    }
                ]
                fallback_tracks.extend(breakup_tracks)
            
            playlist = fallback_tracks
            print(f"[FALLBACK] Added {len(playlist)} hardcoded fallback tracks")
        
        # Remove duplicate tracks and apply AI-powered relevance scoring
        seen_tracks = set()
        unique_playlist = []
        for track in playlist:
            # Create a unique identifier using name and artist
            track_key = f"{track.get('name', '')}_{track.get('artist', '')}"
            if track_key not in seen_tracks:
                unique_playlist.append(track)
                seen_tracks.add(track_key)
        
        # Apply AI-powered relevance scoring to sort tracks by relevance
        if unique_playlist:
            # Get enhanced user preferences including playlist analysis with mood and language filtering
            user_preferences = get_enhanced_user_preferences(access_token, detected_context, None, None)
            user_genres = []
            for artist in user_artists[:5]:  # Reduced from 10 to 5 for speed
                genres = get_spotify_artist_genres(artist, access_token)
                user_genres.extend(genres)
            user_genres = list(set(user_genres))  # Remove duplicates
            
            # OPTIMIZED: Use AI sorting with reduced entities for speed
            try:
                # Limit to top 15 tracks for AI sorting to improve speed
                tracks_for_ai = unique_playlist[:25]  # Increased from 15 to 25 tracks for AI sorting
                ai_sorted_playlist = ai_sort_by_relevance(
                    entities=tracks_for_ai,
                    user_context=user_context,
                    user_artists=user_artists,
                    user_genres=user_genres,
                    context_type=detected_context,
                    user_country=user_country,
                    location=location,
                    user_preferences=user_preferences,
                    gemini_api_key=gemini_api_key
                )
                
                # Combine AI sorted tracks with remaining tracks
                remaining_tracks = unique_playlist[25:]  # Adjusted to match the new limit
                combined_playlist = ai_sorted_playlist + remaining_tracks
                
                # Deduplicate the combined playlist
                final_playlist = []
                seen_track_ids = set()
                seen_track_names = set()
                
                for track in combined_playlist:
                    # Try to get track ID first (most reliable)
                    track_id = track.get('id') or track.get('spotify_id')
                    track_name = track.get('name', '').lower().strip()
                    track_artist = track.get('artist', '').lower().strip()
                    
                    # Create a unique identifier
                    if track_id:
                        unique_id = track_id
                    else:
                        # Fallback to name + artist combination
                        unique_id = f"{track_name}_{track_artist}"
                    
                    # Check if we've seen this track before
                    if unique_id not in seen_track_ids and track_name not in seen_track_names:
                        final_playlist.append(track)
                        seen_track_ids.add(unique_id)
                        seen_track_names.add(track_name)
                    else:
                        print(f"[AI DEDUPLICATION] Removed duplicate track: {track_name} by {track_artist}")
                
                playlist = final_playlist
                print(f"[AI SORTING] Sorted {len(tracks_for_ai)} tracks with AI, added {len(remaining_tracks)} remaining tracks, final length: {len(playlist)}")
                
            except Exception as e:
                print(f"[AI SORTING] Error in AI sorting, using original order: {e}")
                playlist = unique_playlist
        else:
            playlist = unique_playlist  # Show all tracks, no limit
        
        # --- ENHANCED: Add display_tags and display_qloo_artists for simple rendering ---
        display_tags = ', '.join(enhanced_tags)
        display_qloo_artists = ', '.join(qloo_reco_artists[:15])

        # --- ENHANCED: Add Qloo power showcase information ---
        qloo_power_showcase = {
            "enhanced_system": True,
            "cultural_intelligence": bool(cultural_context),
            "location_awareness": bool(location),
            "multi_strategy_recommendations": True,
            "cross_domain_analysis": True,
            "url_encoding_fixed": True,
            "total_recommendations": len(enhanced_recommendations),
                        "cultural_tags_count": len([tag for tag in enhanced_tags if any(cultural_word in tag.lower() for cultural_word in ["latin", "k-pop", "afrobeats", "jazz", "blues", "folk", "world", "bollywood", "hindi", "indian", "ghazal", "sufi", "marathi", "punjabi", "bengali", "tamil", "telugu", "malayalam", "kannada", "urdu", "bhangra", "qawwali", "classical", "carnatic", "hindustani"])]),
            "location_based_count": len([artist for artist in enhanced_recommendations if artist.get("cultural_relevance", 0) > 0.3 or any(indian_word in artist.get("name", "").lower() for indian_word in ["singh", "kumar", "sharma", "patel", "khan", "ali", "ahmed", "reddy", "naidu", "iyer", "menon", "pillai", "nair", "gandhi", "nehru", "bose", "chakraborty", "mukherjee", "banerjee", "das", "roy", "sarkar", "malhotra", "kapoor", "chopra", "bhatt", "verma", "yadav", "gupta", "jain", "mehta", "shah", "desai", "joshi", "tripathi", "mishra", "tiwari", "pandey", "chauhan", "thakur", "rajput", "yadav", "kumar", "singh", "khan", "ali", "ahmed", "reddy", "naidu", "iyer", "menon", "pillai", "nair"])]),
        }

        enhanced_features = [
            "Cultural Intelligence",
            "Location-Aware Recommendations", 
            "Multi-Strategy Analysis",
            "Affinity Scoring",
            "Cross-Domain Insights",
            "Enhanced Gemini Integration",
            "AI-Powered Relevance Scoring"
        ]

        # --- ENHANCED: Add debug info to response ---
        debug_info = {
            "enhanced_gemini_tags": enhanced_tags,
            "cultural_context": cultural_context,
            "qloo_tag_ids": tag_ids,
            "spotify_user_artists": user_artists,
            "spotify_user_tracks": user_tracks,
            "qloo_recommended_artists": qloo_reco_artists,
            "playlist_length": len(playlist),
            "context_type": "enhanced",
            "location_used": location,
            "user_country": user_country
        }

        # --- No DB save ---

        # ENHANCED: Prepare response data with Qloo power showcase and AI scoring info
        response_data = {
            "playlist": playlist, 
            "tags": enhanced_tags, 
            "qloo_artists": qloo_reco_artists[:15],  # Limit returned artists
            "context_type": "enhanced",
            "debug": debug_info,
            "display_tags": display_tags,
            "display_qloo_artists": display_qloo_artists,
            "access_token": access_token,
            "from_cache": False,
            "generated_timestamp": datetime.now().isoformat(),
            "location_used": location,
            "user_country": user_country,
            # ENHANCED: Qloo power showcase information
            "qloo_power_showcase": qloo_power_showcase,
            "enhanced_features": enhanced_features,
            "cultural_insights": cultural_insights[:5],  # Top 5 cultural insights
            # AI scoring information
            "ai_scoring_info": {
                "ai_scoring_enabled": True,
                "tracks_ai_scored": len([t for t in playlist if t.get('ai_scored', False)]),
                "total_tracks": len(playlist),
                "scoring_method": "AI-Enhanced Relevance Scoring",
                "ai_score_weight": 0.7,
                "traditional_score_weight": 0.3
            }
        }
        
        # Cache the result
        cache_recommendation(user_id, user_context, response_data, "music")
        
        # At the end, return the access_token to the frontend as well
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in music recommendation: {e}")
        return jsonify({'error': 'Failed to generate music recommendations'}), 500

# get_spotify_playlist_by_id function moved to spotify_client.py to avoid duplication

# Helper: Get user's recently played tracks
# get_spotify_recently_played function moved to spotify_client.py to avoid duplication

# Helper: Get Qloo artist info and generate tags using Gemini
def get_artist_tags_with_gemini(artist_name, qloo_api_key, gemini_api_key):
    # First get artist info from Qloo
    url = "https://hackathon.api.qloo.com/v2/search"
    headers = {
        "X-API-KEY": qloo_api_key,
        "Content-Type": "application/json"
    }
    params = {
        "query": artist_name,
        "type": "artist",
        "limit": 1
    }
    
    artist_info = ""
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", {}).get("entities", [])
        if results:
            artist = results[0]
            artist_info = f"Artist: {artist.get('name', '')}, Description: {artist.get('description', '')}, Genres: {', '.join(artist.get('genres', []))}"
    except Exception as e:
        print(f"Error getting Qloo artist info for {artist_name}: {e}")
        artist_info = f"Artist: {artist_name}"
    
    # Use Gemini to generate relevant tags based on artist info
    prompt = f"""
Based on this artist information: "{artist_info}"
Generate relevant tags that would be useful for cross-domain recommendations (movies, books, restaurants, etc.).
Focus on mood, themes, cultural elements, and characteristics that could translate across different domains.
Respond with a concise, comma-separated list of descriptive tags.
"""
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        tag_str = data['candidates'][0]['content']['parts'][0]['text']
        tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        return tags
    except Exception as e:
        print(f"Error calling Gemini for artist tags: {e}")
        return []

# Helper: Get domain-specific tag selection using Gemini
def select_domain_tags_with_gemini(all_tags, target_domain, user_country, gemini_api_key):
    prompt = f"""
From this list of tags: {', '.join(all_tags)}
Select the most relevant tags for {target_domain} recommendations for a user in {user_country}.
Consider cultural preferences, local trends, and domain-specific relevance.
Respond with a comma-separated list of the 5-8 most relevant tags only.
"""
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        tag_str = data['candidates'][0]['content']['parts'][0]['text']
        selected_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        return selected_tags
    except Exception as e:
        print(f"Error selecting domain tags with Gemini: {e}")
        return all_tags[:5]  # Fallback to first 5 tags

def get_fallback_genres_for_artist(artist_name):
    """Get fallback genres based on artist name patterns when Spotify API fails"""
    artist_name_lower = artist_name.lower()
    
    # Common genre mappings based on artist name patterns
    genre_patterns = {
        # Electronic/Dance
        'alan walker': ['electronic', 'dance', 'edm', 'pop'],
        'marshmello': ['electronic', 'dance', 'edm', 'pop'],
        'skrillex': ['electronic', 'dubstep', 'edm'],
        'calvin harris': ['electronic', 'dance', 'pop'],
        'david guetta': ['electronic', 'dance', 'edm'],
        
        # Indian Artists
        'arijit singh': ['bollywood', 'indian pop', 'film music'],
        'amit trivedi': ['bollywood', 'indian pop', 'film music'],
        'pritam': ['bollywood', 'indian pop', 'film music'],
        'shreya ghoshal': ['bollywood', 'indian pop', 'film music'],
        'neha kakkar': ['bollywood', 'indian pop', 'film music'],
        'divine': ['hip hop', 'rap', 'indian hip hop'],
        'karsh kale': ['electronic', 'indian classical', 'fusion'],
        'ritviz': ['indian pop', 'electronic', 'fusion'],
        'pali': ['indian pop', 'bollywood'],
        'sachin-jigar': ['bollywood', 'indian pop', 'film music'],
        'mitraz': ['indian pop', 'bollywood'],
        'benny dayal': ['bollywood', 'indian pop', 'film music'],
        
        # Pop Artists
        'taylor swift': ['pop', 'country pop', 'pop rock'],
        'ed sheeran': ['pop', 'folk pop', 'acoustic'],
        'adele': ['pop', 'soul', 'r&b'],
        'beyonce': ['pop', 'r&b', 'hip hop'],
        'justin bieber': ['pop', 'r&b', 'dance pop'],
        
        # Rock Artists
        'queen': ['rock', 'classic rock', 'hard rock'],
        'led zeppelin': ['rock', 'hard rock', 'blues rock'],
        'pink floyd': ['rock', 'progressive rock', 'psychedelic rock'],
        'the beatles': ['rock', 'pop rock', 'classic rock'],
        
        # Hip Hop/Rap
        'eminem': ['hip hop', 'rap', 'hardcore hip hop'],
        'drake': ['hip hop', 'rap', 'r&b'],
        'kendrick lamar': ['hip hop', 'rap', 'conscious hip hop'],
        'post malone': ['hip hop', 'rap', 'pop rap'],
    }
    
    # Check for exact matches first
    for pattern, genres in genre_patterns.items():
        if pattern in artist_name_lower:
            return genres
    
    # Check for partial matches
    for pattern, genres in genre_patterns.items():
        if any(word in artist_name_lower for word in pattern.split()):
            return genres
    
    # Default genres based on common patterns
    if any(word in artist_name_lower for word in ['dj', 'electronic', 'edm']):
        return ['electronic', 'dance', 'edm']
    elif any(word in artist_name_lower for word in ['rock', 'metal']):
        return ['rock', 'alternative rock']
    elif any(word in artist_name_lower for word in ['rap', 'hip hop']):
        return ['hip hop', 'rap']
    elif any(word in artist_name_lower for word in ['jazz', 'blues']):
        return ['jazz', 'blues']
    elif any(word in artist_name_lower for word in ['classical', 'orchestra']):
        return ['classical', 'orchestral']
    else:
        # Default to pop for unknown artists
        return ['pop', 'contemporary']

# Helper: Get user's country from Spotify profile
def get_user_country(access_token):
    url = "https://api.spotify.com/v1/me"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("country", "US")  # Default to US if not found
    except Exception as e:
        print(f"Error fetching user country: {e}")
        return "US"

def get_location_from_country(country_code):
    """
    Convert country code to location name for Qloo API
    Maps country codes to major cities/regions for better recommendations
    """
    country_to_location = {
        # India - Major cities for diverse recommendations
        "IN": "Mumbai",  # Bollywood capital
        "US": "New York",  # US music hub
        "GB": "London",  # UK music scene
        "CA": "Toronto",  # Canadian music
        "AU": "Sydney",  # Australian music
        "DE": "Berlin",  # German electronic scene
        "FR": "Paris",  # French music
        "JP": "Tokyo",  # Japanese music
        "KR": "Seoul",  # Korean music
        "BR": "Rio de Janeiro",  # Brazilian music
        "MX": "Mexico City",  # Mexican music
        "ES": "Madrid",  # Spanish music
        "IT": "Milan",  # Italian music
        "NL": "Amsterdam",  # Dutch music
        "SE": "Stockholm",  # Swedish music
        "NO": "Oslo",  # Norwegian music
        "DK": "Copenhagen",  # Danish music
        "FI": "Helsinki",  # Finnish music
        "CH": "Zurich",  # Swiss music
        "AT": "Vienna",  # Austrian music
        "BE": "Brussels",  # Belgian music
        "PT": "Lisbon",  # Portuguese music
        "IE": "Dublin",  # Irish music
        "PL": "Warsaw",  # Polish music
        "CZ": "Prague",  # Czech music
        "HU": "Budapest",  # Hungarian music
        "RO": "Bucharest",  # Romanian music
        "BG": "Sofia",  # Bulgarian music
        "HR": "Zagreb",  # Croatian music
        "SI": "Ljubljana",  # Slovenian music
        "SK": "Bratislava",  # Slovak music
        "LT": "Vilnius",  # Lithuanian music
        "LV": "Riga",  # Latvian music
        "EE": "Tallinn",  # Estonian music
        "GR": "Athens",  # Greek music
        "CY": "Nicosia",  # Cypriot music
        "MT": "Valletta",  # Maltese music
        "LU": "Luxembourg",  # Luxembourg music
        "IS": "Reykjavik",  # Icelandic music
        "TR": "Istanbul",  # Turkish music
        "IL": "Tel Aviv",  # Israeli music
        "AE": "Dubai",  # UAE music
        "SA": "Riyadh",  # Saudi music
        "EG": "Cairo",  # Egyptian music
        "ZA": "Johannesburg",  # South African music
        "NG": "Lagos",  # Nigerian music
        "KE": "Nairobi",  # Kenyan music
        "GH": "Accra",  # Ghanaian music
        "UG": "Kampala",  # Ugandan music
        "TZ": "Dar es Salaam",  # Tanzanian music
        "ET": "Addis Ababa",  # Ethiopian music
        "RW": "Kigali",  # Rwandan music
        "BI": "Bujumbura",  # Burundian music
        "DJ": "Djibouti",  # Djiboutian music
        "SO": "Mogadishu",  # Somali music
        "ER": "Asmara",  # Eritrean music
        "SD": "Khartoum",  # Sudanese music
        "SS": "Juba",  # South Sudanese music
        "CF": "Bangui",  # Central African music
        "TD": "N'Djamena",  # Chadian music
        "CM": "Yaounde",  # Cameroonian music
        "GQ": "Malabo",  # Equatorial Guinean music
        "GA": "Libreville",  # Gabonese music
        "CG": "Brazzaville",  # Congolese music
        "CD": "Kinshasa",  # DR Congolese music
        "AO": "Luanda",  # Angolan music
        "ZM": "Lusaka",  # Zambian music
        "ZW": "Harare",  # Zimbabwean music
        "BW": "Gaborone",  # Botswanan music
        "NA": "Windhoek",  # Namibian music
        "SZ": "Mbabane",  # Eswatini music
        "LS": "Maseru",  # Lesotho music
        "MG": "Antananarivo",  # Malagasy music
        "MU": "Port Louis",  # Mauritian music
        "SC": "Victoria",  # Seychellois music
        "KM": "Moroni",  # Comorian music
        "MV": "Male",  # Maldivian music
        "LK": "Colombo",  # Sri Lankan music
        "BD": "Dhaka",  # Bangladeshi music
        "NP": "Kathmandu",  # Nepali music
        "BT": "Thimphu",  # Bhutanese music
        "MM": "Yangon",  # Myanmar music
        "TH": "Bangkok",  # Thai music
        "LA": "Vientiane",  # Lao music
        "KH": "Phnom Penh",  # Cambodian music
        "VN": "Hanoi",  # Vietnamese music
        "MY": "Kuala Lumpur",  # Malaysian music
        "SG": "Singapore",  # Singaporean music
        "ID": "Jakarta",  # Indonesian music
        "PH": "Manila",  # Filipino music
        "TW": "Taipei",  # Taiwanese music
        "HK": "Hong Kong",  # Hong Kong music
        "MO": "Macau",  # Macanese music
        "CN": "Beijing",  # Chinese music
        "MN": "Ulaanbaatar",  # Mongolian music
        "KZ": "Almaty",  # Kazakh music
        "KG": "Bishkek",  # Kyrgyz music
        "TJ": "Dushanbe",  # Tajik music
        "UZ": "Tashkent",  # Uzbek music
        "TM": "Ashgabat",  # Turkmen music
        "AF": "Kabul",  # Afghan music
        "PK": "Islamabad",  # Pakistani music
        "IR": "Tehran",  # Iranian music
        "IQ": "Baghdad",  # Iraqi music
        "SY": "Damascus",  # Syrian music
        "LB": "Beirut",  # Lebanese music
        "JO": "Amman",  # Jordanian music
        "PS": "Ramallah",  # Palestinian music
        "KW": "Kuwait City",  # Kuwaiti music
        "BH": "Manama",  # Bahraini music
        "QA": "Doha",  # Qatari music
        "OM": "Muscat",  # Omani music
        "YE": "Sanaa",  # Yemeni music
    }
    
    return country_to_location.get(country_code.upper(), "New York")  # Default fallback

# Global progress tracking for cross-domain recommendations
crossdomain_progress = {}

@app.route('/crossdomain-progress/<user_id>', methods=['GET'])
def get_crossdomain_progress(user_id):
    """Get progress for cross-domain recommendations"""
    progress = crossdomain_progress.get(user_id, {
        'current_step': 0,
        'total_steps': 25,  # 5 artists × 5 domains
        'current_artist': '',
        'current_domain': '',
        'percentage': 0,
        'status': 'idle'
    })
    print(f"[PROGRESS] Returning progress for user {user_id}: {progress}")
    return jsonify(progress)

@app.route('/crossdomain-recommendations', methods=['POST', 'OPTIONS'])
def crossdomain_recommendations():
    """Enhanced cross-domain recommendations using the unified function"""
    
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    try:
        # Clean up expired cache entries periodically
        cleanup_expired_cache()
        
        # Validate request data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        spotify_token = data.get('spotify_token', '')
        user_context = data.get('user_context', '')
        music_artists = data.get('music_artists', [])
        top_scored_artists = data.get('top_scored_artists', [])
        user_tags = data.get('user_tags', [])
        tracks = data.get('tracks', [])
        qloo_api_key = data.get('qloo_api_key', DEFAULT_QLOO_API_KEY)
        gemini_api_key = data.get('gemini_api_key', DEFAULT_GEMINI_API_KEY)
        recommendations_per_domain = data.get('limit', 10)  # Default to 10, but allow override
        
        # Validate required parameters
        if not spotify_token:
            print("[CROSSDOMAIN] No Spotify token provided - will use fallback artists")
        
        print(f"[CROSSDOMAIN] Starting crossdomain recommendations with token length: {len(spotify_token) if spotify_token else 0}")
        print(f"[CROSSDOMAIN] User context: {user_context}")
        print(f"[CROSSDOMAIN] User tags: {user_tags}")
        print(f"[CROSSDOMAIN] Music artists: {music_artists}")
        print(f"[CROSSDOMAIN] Top scored artists: {top_scored_artists}")
        print(f"[CROSSDOMAIN] Tracks count: {len(tracks) if tracks else 0}")
        
        # Use the unified function
        response_data = generate_cross_domain_recommendations_unified(
            spotify_token=spotify_token,
            qloo_api_key=qloo_api_key,
            gemini_api_key=gemini_api_key,
            user_context=user_context,
            music_artists=music_artists,
            top_scored_artists=top_scored_artists,
            user_tags=user_tags,
            tracks=tracks,
            recommendations_per_domain=recommendations_per_domain
        )
        
        # Cache the result
        try:
            if spotify_token:
                user_profile = get_spotify_user_profile(spotify_token)
                user_id = user_profile.get('id') if user_profile else 'anonymous_user'
            else:
                user_id = 'anonymous_user'
            
            cache_context = f"crossdomain_top{recommendations_per_domain}"
            cache_recommendation(user_id, cache_context, response_data, "crossdomain")
        except Exception as e:
            print(f"Could not cache result: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in crossdomain recommendations: {e}")
        return jsonify({'error': f'Failed to generate crossdomain recommendations: {str(e)}'}), 500

# Helper functions for the enhanced cross-domain system
def get_dynamic_tags_for_domain_enhanced(domain: str, api_key: str, user_country: str = None, location: str = None, limit: int = 15, artist_name: str = None, artist_genres: list = None, gemini_api_key: str = None):
    """Enhanced version of get_dynamic_tags_for_domain with dynamic cultural context generation using Gemini"""
    domain_tag_types = {
        "music artist": ["genre:music", "mood:music"],
        "book": ["genre:media"],
        "movie": ["genre:media"],
        "podcast": ["genre:media"],
        "TV show": ["genre:media"],
    }
    
    tag_types = domain_tag_types.get(domain, [])
    if not tag_types:
        return []
        
    url = "https://hackathon.api.qloo.com/v2/tags"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    # Lower popularity threshold to get more diverse and relevant results
    # This allows for more niche and culturally relevant content
    min_popularity = 0.3 if user_country == "IN" else 0.4
    
    params = {
        "filter.tag.types": ",".join(tag_types),
        "filter.popularity.min": min_popularity,
        "take": limit * 3  # Get more tags to filter for relevance
    }
    
    # Add location-based filtering
    if location:
        params["signal.location.query"] = location
        print(f"[QLOO] Using location-based tag discovery for {domain} in {location}")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        all_tags = [
            {"name": tag["name"], "id": tag["id"]}
            for tag in data.get("results", {}).get("tags", [])
        ]
        
        # Generate cultural keywords dynamically using Gemini if available
        cultural_keywords = []
        if gemini_api_key and (user_country or location or artist_name or artist_genres):
            print(f"[GEMINI] Generating cultural keywords for {domain} - Artist: {artist_name}, Genres: {artist_genres}, Location: {location}")
            cultural_keywords = generate_cultural_keywords_with_gemini(
                user_country, location, artist_name, artist_genres, domain, gemini_api_key
            )
            print(f"[GEMINI] Generated {len(cultural_keywords)} cultural keywords for {domain}: {cultural_keywords}")
        else:
            print(f"[GEMINI] Skipping cultural keyword generation - missing API key or context data")
        
        # If Gemini keywords are available, prioritize them
        if cultural_keywords:
            prioritized_tags = []
            other_tags = []
            
            for tag in all_tags:
                tag_name_lower = tag["name"].lower()
                if any(keyword.lower() in tag_name_lower for keyword in cultural_keywords):
                    prioritized_tags.append(tag)
                else:
                    other_tags.append(tag)
            
            # Return prioritized tags first, then others to reach the limit
            return (prioritized_tags + other_tags)[:limit]
        
        return all_tags[:limit]
    except Exception as e:
        print(f"Error fetching tags for domain '{domain}': {e}")
        return []

@retry_on_gemini_rate_limit()
@gemini_rate_limit()
def generate_cultural_keywords_with_gemini(user_country: str = None, location: str = None, artist_name: str = None, artist_genres: list = None, domain: str = None, gemini_api_key: str = None):
    """Generate culturally relevant keywords using Gemini based on location, artist, and domain"""
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[GEMINI] No valid API key provided, returning fallback cultural keywords")
        # Return fallback cultural keywords based on location
        if location and 'mumbai' in location.lower():
            return ['bollywood', 'indian', 'hindi', 'desi', 'mumbai']
        elif user_country and user_country.lower() == 'in':
            return ['india', 'bollywood', 'indian', 'desi']
        else:
            return ['international', 'global', 'pop', 'contemporary']
    
    # Check cache first
    cache_key = get_gemini_cache_key(user_country, location, artist_name, artist_genres, domain, gemini_api_key)
    cached_response = get_cached_gemini_response(cache_key)
    if cached_response:
        return cached_response
    
    # Build context for Gemini
    context_parts = []
    if user_country:
        context_parts.append(f"user country: {user_country}")
    if location:
        context_parts.append(f"location: {location}")
    if artist_name:
        context_parts.append(f"artist: {artist_name}")
    if artist_genres:
        context_parts.append(f"artist genres: {', '.join(artist_genres)}")
    if domain:
        context_parts.append(f"target domain: {domain}")
    
    context = ", ".join(context_parts)
    
    prompt = f"""
    Based on this context: {context}

    Generate 10-15 culturally relevant keywords that would be useful for finding {domain} recommendations.
    Consider:
    1. Local cultural elements, languages, and traditions
    2. Regional music styles and preferences  
    3. Artist's genre and style characteristics
    4. Cultural themes that translate across domains
    5. Popular local trends and preferences

    Return only a comma-separated list of keywords, no explanations.
    Focus on specific, actionable terms that would appear in tag names.
    Examples: bollywood, indie, classical, romance, action, comedy, drama, etc.
    """
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        keywords_str = data['candidates'][0]['content']['parts'][0]['text']
        keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        
        # Cache the response
        cache_gemini_response(cache_key, keywords)
        
        return keywords
    except Exception as e:
        print(f"Error generating cultural keywords with Gemini: {e}")
        return []

@retry_on_gemini_rate_limit()
@gemini_rate_limit()
def call_gemini_for_tag_enhanced(user_context: str, tag_list, gemini_api_key: str, user_country: str = None):
    """Enhanced Gemini tag selection with better error handling and country context"""
    if not tag_list:
        return None
    
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[GEMINI] No valid API key provided, using fallback tag selection")
        # Return the first tag as fallback
        return tag_list[0] if tag_list else None
        
    # Check cache first
    cache_key = get_gemini_cache_key(user_context, tag_list, gemini_api_key, user_country)
    cached_response = get_cached_gemini_response(cache_key)
    if cached_response:
        return cached_response
        
    tag_list_str = ', '.join(tag_list)
    country_context = f" The user is from {user_country}." if user_country else ""
    
    # Extract domain from context for better guidance
    domain_guidance = ""
    if "Domain: movie" in user_context:
        domain_guidance = " Choose a movie genre tag like action, drama, comedy, thriller, romance, sci-fi, horror, adventure, mystery, or documentary."
    elif "Domain: book" in user_context:
        domain_guidance = " Choose a book genre tag like fiction, romance, mystery, biography, self-help, fantasy, sci-fi, thriller, historical, or young-adult."
    elif "Domain: podcast" in user_context:
        domain_guidance = " Choose a podcast category tag like comedy, news, true-crime, business, health, education, technology, sports, politics, or entertainment."
    elif "Domain: TV show" in user_context:
        domain_guidance = " Choose a TV show genre tag like drama, comedy, reality, documentary, action, thriller, sci-fi, horror, adventure, or mystery."
    elif "Domain: music artist" in user_context:
        domain_guidance = " Choose a music genre tag like pop, rock, hip-hop, jazz, classical, electronic, country, r&b, indie, or alternative."
    
    prompt = (
        f"Given the context: {user_context}{country_context}, pick the best tag from this list: [{tag_list_str}]. "
        f"Consider cultural preferences and regional tastes when selecting the tag.{domain_guidance} "
        f"Only return the tag name, nothing else."
    )
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        tag_str = data['candidates'][0]['content']['parts'][0]['text']
        tag = tag_str.split(",")[0].strip()
        
        # Cache the response
        cache_gemini_response(cache_key, tag)
        
        return tag
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return None

@retry_on_gemini_rate_limit()
@gemini_rate_limit()
def get_domain_appropriate_tags(artist_genres, domain, qloo_api_key, gemini_api_key, user_country=None, location=None):
    """Get domain-appropriate tags based on artist genres with rate limiting"""
    try:
        # Check if we have a valid API key
        if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
            print(f"[GEMINI] No valid API key provided, using fallback domain tags")
            # Return fallback tags based on domain
            fallback_tags = {
                'movie': ['action', 'drama', 'comedy', 'romance'],
                'book': ['fiction', 'romance', 'mystery', 'biography'],
                'podcast': ['comedy', 'news', 'business', 'health'],
                'tv_show': ['drama', 'comedy', 'reality', 'documentary'],
                'artist': ['pop', 'rock', 'electronic', 'jazz']
            }
            return [], fallback_tags.get(domain, ['general'])
        
        # Check cache first
        cache_key = get_gemini_cache_key(artist_genres, domain, qloo_api_key, gemini_api_key, user_country, location)
        cached_response = get_cached_gemini_response(cache_key)
        if cached_response:
            # Return cached tag IDs and tags
            tag_ids, tags = cached_response
            return tag_ids, tags
        
        # Generate domain-specific tags using Gemini
        prompt = f"""
        Based on these music genres: {', '.join(artist_genres)}
        Generate 5-8 tags for {domain} recommendations that would appeal to someone who listens to this music.
        
        Consider:
        1. Emotional connections between music and {domain}
        2. Cultural preferences for {user_country if user_country else 'global'} audience
        3. Popular {domain} categories that align with the music taste
        
        Return only a comma-separated list of tag names, no explanations.
        """
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': gemini_api_key
        }
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if 'candidates' not in data or not data['candidates']:
            return [], []
            
        tags_str = data['candidates'][0]['content']['parts'][0]['text']
        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        
        # Convert tags to Qloo tag IDs
        tag_ids = []
        for tag in tags:
            tag_id = get_qloo_tag_id(tag, qloo_api_key)
            if tag_id:
                tag_ids.append(tag_id)
        
        # Cache the response
        cache_gemini_response(cache_key, (tag_ids, tags))
        
        return tag_ids, tags
        
    except Exception as e:
        print(f"Error getting domain appropriate tags: {e}")
        return [], []

def find_working_tag(gemini_tag, tag_names, tag_ids, qloo_client, etype, artist_ids):
    """Find a working tag using the robust fallback system with domain-specific improvements"""
    selected_tag = None
    selected_tag_id = None
    
    # Domain-specific fallback tags that are known to work
    domain_fallbacks = {
        "urn:entity:artist": ["pop", "rock", "hip-hop", "jazz", "classical", "electronic", "country", "r&b", "indie", "alternative"],
        "urn:entity:movie": ["action", "drama", "comedy", "thriller", "romance", "sci-fi", "horror", "adventure", "mystery", "documentary"],
        "urn:entity:book": ["fiction", "romance", "mystery", "biography", "self-help", "fantasy", "sci-fi", "thriller", "historical", "young-adult"],
        "urn:entity:podcast": ["comedy", "news", "true-crime", "business", "health", "education", "technology", "sports", "politics", "entertainment"],
        "urn:entity:tv_show": ["drama", "comedy", "reality", "documentary", "action", "thriller", "sci-fi", "horror", "adventure", "mystery"]
    }
    
    # Try exact match first
    if gemini_tag and gemini_tag in tag_ids:
        selected_tag = gemini_tag
        selected_tag_id = tag_ids[gemini_tag]
        print(f"Using Gemini's exact tag: {selected_tag}")
    else:
        # Try partial match
        if gemini_tag:
            gemini_lower = gemini_tag.lower()
            for tag_name in tag_names:
                if gemini_lower in tag_name.lower() or tag_name.lower() in gemini_lower:
                    selected_tag = tag_name
                    selected_tag_id = tag_ids[tag_name]
                    print(f"Using partial match tag: {selected_tag}")
                    break
        
        # If still no match, test each tag until one works
        if not selected_tag:
            print("Testing available tags for working recommendations...")
            for tag_name in tag_names:
                test_tag_id = tag_ids[tag_name]
                test_rec = qloo_client.get_recommendations(etype, test_tag_id, artist_ids, take=1)
                if test_rec and test_rec.get("results", {}).get("entities", []):
                    selected_tag = tag_name
                    selected_tag_id = test_tag_id
                    print(f"Found working tag: {selected_tag}")
                    break
        
        # If still no working tag found, try domain-specific fallbacks
        if not selected_tag:
            print("No working tags found in available tags, trying domain-specific fallbacks...")
            fallback_tags = domain_fallbacks.get(etype, [])
            
            for fallback_tag in fallback_tags:
                try:
                    # Try to get the tag ID from Qloo
                    fallback_tag_id = get_qloo_tag_id(fallback_tag, qloo_client.api_key)
                    if fallback_tag_id:
                        test_rec = qloo_client.get_recommendations(etype, fallback_tag_id, artist_ids, take=1)
                        if test_rec and test_rec.get("results", {}).get("entities", []):
                            selected_tag = fallback_tag
                            selected_tag_id = fallback_tag_id
                            print(f"Found working fallback tag: {selected_tag}")
                            break
                except Exception as e:
                    print(f"Could not test fallback tag {fallback_tag}: {e}")
                    continue
    
    return selected_tag, selected_tag_id



@app.route('/gemini-cache', methods=['GET', 'POST', 'DELETE'])
def gemini_cache_management():
    """Manage Gemini cache entries"""
    if request.method == 'GET':
        # For GET requests, get API key from query params
        gemini_api_key = request.args.get('gemini_api_key', DEFAULT_GEMINI_API_KEY)
    else:
        # For POST/DELETE requests, get from JSON body
        data = request.get_json() or {}
        gemini_api_key = data.get('gemini_api_key', DEFAULT_GEMINI_API_KEY)
    
    if request.method == 'GET':
        # List all cached contents
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': gemini_api_key
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                cache_data = response.json()
                return jsonify({
                    'success': True,
                    'cached_contents': cache_data.get('cachedContents', []),
                    'total_count': len(cache_data.get('cachedContents', [])),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to fetch cache: {response.status_code}',
                    'timestamp': datetime.now().isoformat()
                }), response.status_code
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    elif request.method == 'DELETE':
        # Delete specific cache entry
        cache_name = data.get('cache_name')
        if not cache_name:
            return jsonify({
                'success': False,
                'error': 'cache_name is required for deletion'
            }), 400
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{cache_name}"
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': gemini_api_key
            }
            
            response = requests.delete(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return jsonify({
                    'success': True,
                    'message': f'Cache {cache_name} deleted successfully',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to delete cache: {response.status_code}',
                    'timestamp': datetime.now().isoformat()
                }), response.status_code
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    else:  # POST - Create new cache entry
        prompt = data.get('prompt')
        target_domain = data.get('target_domain', 'general')
        
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'prompt is required for cache creation'
            }), 400
        
        try:
            cache_tags = use_gemini_cached_content(prompt, target_domain, gemini_api_key)
            if cache_tags:
                return jsonify({
                    'success': True,
                    'message': 'Cache created and tags generated successfully',
                    'tags': cache_tags,
                    'count': len(cache_tags),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create cache or generate tags',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

@app.route('/performance', methods=['GET'])
def performance_metrics():
    """Get performance metrics and cache statistics"""
    try:
        # Calculate cache statistics
        total_entries = len(recommendation_cache)
        expired_entries = 0
        valid_entries = 0
        
        for entry in recommendation_cache.values():
            if is_cache_valid(entry):
                valid_entries += 1
            else:
                expired_entries += 1
        
        # Calculate cache hit rate (simplified)
        cache_hit_rate = valid_entries / max(total_entries, 1) * 100
        
        # Get Gemini cache statistics
        gemini_cache_count = 0
        try:
            gemini_url = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
            gemini_headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': DEFAULT_GEMINI_API_KEY
            }
            gemini_response = requests.get(gemini_url, headers=gemini_headers, timeout=5)
            if gemini_response.status_code == 200:
                gemini_data = gemini_response.json()
                gemini_cache_count = len(gemini_data.get('cachedContents', []))
        except:
            pass
        
        metrics = {
            'cache': {
                'total_entries': total_entries,
                'valid_entries': valid_entries,
                'expired_entries': expired_entries,
                'hit_rate_percentage': round(cache_hit_rate, 2)
            },
            'gemini_cache': {
                'total_entries': gemini_cache_count,
                'status': 'active' if gemini_cache_count > 0 else 'inactive'
            },
            'system': {
                'uptime': 'active',
                'memory_usage': 'monitored',
                'api_endpoints': 10
            },
            'features': {
                'dynamic_tag_generation': 'enabled',
                'cultural_context': 'enabled',
                'cross_domain_recommendations': 'enabled',
                'caching': 'enabled',
                'gemini_caching': 'enabled',
                'rate_limiting': 'enabled'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test-tag-filtering', methods=['POST'])
def test_tag_filtering():
    """Test endpoint to debug tag filtering and Qloo API responses"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        domain = data.get('domain', 'movie')
        tag_name = data.get('tag_name', 'action')
        qloo_api_key = data.get('qloo_api_key', DEFAULT_QLOO_API_KEY)
        
        # Map domain to entity type
        domain_to_etype = {
            "movie": "urn:entity:movie",
            "book": "urn:entity:book", 
            "podcast": "urn:entity:podcast",
            "TV show": "urn:entity:tv_show",
            "music artist": "urn:entity:artist"
        }
        
        etype = domain_to_etype.get(domain, "urn:entity:movie")
        
        # Get tag ID
        tag_id = get_qloo_tag_id(tag_name, qloo_api_key)
        if not tag_id:
            return jsonify({'error': f'Tag "{tag_name}" not found'}), 404
        
        # Test Qloo API call
        qloo_client = QlooAPIClient(qloo_api_key)
        result = qloo_client.get_recommendations(etype, tag_id, take=10)
        
        entities = result.get("results", {}).get("entities", []) if result else []
        
        return jsonify({
            'domain': domain,
            'tag_name': tag_name,
            'tag_id': tag_id,
            'entity_type': etype,
            'entities_found': len(entities),
            'entities': entities[:5],  # Return first 5 for debugging
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    """Test endpoint for tag filtering functionality"""
    try:
        data = request.get_json()
        test_tags = data.get('tags', ['sad', 'melancholic', 'emotional', 'romantic', 'upbeat', 'energetic', 'calm', 'bollywood', 'hindi pop', 'pop', 'rock', 'electronic'])
        user_context = data.get('user_context', 'I am feeling sad and lonely')
        user_country = data.get('user_country', 'IN')
        location = data.get('location', 'Mumbai')
        gemini_api_key = data.get('gemini_api_key', DEFAULT_GEMINI_API_KEY)
        
        filtered_tags = filter_and_rank_tags_for_music(test_tags, user_context, user_country, location, gemini_api_key)
        
        return jsonify({
            'original_tags': test_tags,
            'filtered_tags': filtered_tags,
            'user_context': user_context,
            'user_country': user_country,
            'location': location,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/search-playlists', methods=['POST', 'OPTIONS'])
def search_playlists():
    """Search for Spotify playlists"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.get_json()
        access_token = data.get('access_token')
        query = data.get('query', '')
        limit = data.get('limit', 20)
        market = data.get('market')
        
        if not access_token:
            return jsonify({'error': 'Access token is required'}), 400
            
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        # Search for playlists
        playlists = search_spotify_playlists(access_token, query, limit, market)
        
        return jsonify({
            'success': True,
            'playlists': playlists,
            'count': len(playlists)
        })
        
    except Exception as e:
        print(f"Error in search_playlists: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-playlist-by-id', methods=['POST', 'OPTIONS'])
def get_playlist_by_id():
    """Get a specific Spotify playlist by ID"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.get_json()
        access_token = data.get('access_token')
        playlist_id = data.get('playlist_id')
        market = data.get('market')
        
        if not access_token:
            return jsonify({'error': 'Access token is required'}), 400
            
        if not playlist_id:
            return jsonify({'error': 'Playlist ID is required'}), 400
        
        # Get playlist by ID
        playlist = get_spotify_playlist_by_id(access_token, playlist_id, market)
        
        if not playlist:
            return jsonify({'error': 'Playlist not found or access denied'}), 404
        
        return jsonify({
            'success': True,
            'playlist': playlist
        })
        
    except Exception as e:
        print(f"Error in get_playlist_by_id: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if cache is working
        cache_status = len(recommendation_cache)
        
        # Check if we can make basic API calls
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'cache_entries': cache_status,
            'version': '2.0.0',
            'features': {
                'dynamic_tag_generation': True,
                'cultural_context': True,
                'cross_domain_recommendations': True,
                'caching': True,
                'rate_limiting': True
            }
        }
        
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500



@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear cache for a specific user or all cache"""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    
    if user_id:
        clear_user_cache(user_id)
        return jsonify({
            'success': True,
            'message': f'Cache cleared for user {user_id}',
            'remaining_entries': len(recommendation_cache)
        })
    else:
        # Clear all cache
        try:
            # Clear Redis cache
            pattern = f"recommendation:*"
            keys = redis_client.keys(pattern)
            redis_client.delete(*keys) if keys else None
            redis_cleared = len(keys)
        except Exception as e:
            print(f"[CACHE CLEAR] Redis error: {e}")
            redis_cleared = 0
        
        # Clear memory cache
        memory_cleared = len(recommendation_cache)
        recommendation_cache.clear()
        
        total_cleared = redis_cleared + memory_cleared
        
        return jsonify({
            'success': True,
            'message': f'All cache cleared (Redis: {redis_cleared}, Memory: {memory_cleared} entries)',
            'remaining_entries': 0
        })

@app.route('/clear-gemini-cache', methods=['POST'])
def clear_gemini_cache():
    """Clear all Gemini API cache"""
    try:
        # Clear Redis Gemini cache
        pattern = f"gemini:*"
        keys = redis_client.keys(pattern)
        redis_client.delete(*keys) if keys else None
        redis_cleared = len(keys)
        
        # Clear memory Gemini cache
        memory_cleared = len(gemini_cache)
        gemini_cache.clear()
        
        total_cleared = redis_cleared + memory_cleared
        
        print(f"[GEMINI CACHE] All Gemini cache cleared (Redis: {redis_cleared}, Memory: {memory_cleared} entries)")
        return jsonify({
            'success': True,
            'message': f'All Gemini cache cleared (Redis: {redis_cleared}, Memory: {memory_cleared} entries)',
            'remaining_entries': 0
        })
    except Exception as e:
        print(f"[GEMINI CACHE] Error clearing cache: {e}")
        return jsonify({
            'success': False,
            'message': 'Error clearing Gemini cache',
            'error': str(e)
        }), 500

def extract_playlist_id_from_url(spotify_url):
    """Extract playlist ID from various Spotify URL formats"""
    try:
        # Handle different Spotify URL formats
        if 'spotify.com/playlist/' in spotify_url:
            # Format: https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
            parts = spotify_url.split('/playlist/')
            if len(parts) > 1:
                playlist_id = parts[1].split('?')[0].split('&')[0]  # Remove query parameters
                return playlist_id
        elif 'spotify:playlist:' in spotify_url:
            # Format: spotify:playlist:37i9dQZF1DXcBWIGoYBM5M
            parts = spotify_url.split(':playlist:')
            if len(parts) > 1:
                return parts[1]
        return None
    except Exception as e:
        print(f"Error extracting playlist ID from URL: {e}")
        return None

@app.route('/get-playlist-by-url', methods=['POST', 'OPTIONS'])
def get_playlist_by_url():
    """Get a playlist by Spotify URL"""
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    spotify_url = data.get('spotify_url')
    spotify_token = data.get('spotify_token')
    market = data.get('market')
    fields = data.get('fields')
    
    # Input validation
    if not spotify_url or not spotify_token:
        return jsonify({'error': 'Missing required parameters: spotify_url and spotify_token'}), 400
    
    # Extract playlist ID from URL
    playlist_id = extract_playlist_id_from_url(spotify_url)
    if not playlist_id:
        return jsonify({'error': 'Invalid Spotify playlist URL'}), 400
    
    # Call the main get-playlist endpoint
    return get_playlist_internal(playlist_id, spotify_token, market, fields)

def get_playlist_internal(playlist_id, spotify_token, market=None, fields=None):
    """Internal function to get playlist data"""
    # Build the API URL
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
    headers = {"Authorization": f"Bearer {spotify_token}"}
    params = {}
    
    # Add optional parameters
    if market:
        params['market'] = market
    if fields:
        params['fields'] = fields
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        playlist_data = response.json()
        
        # Extract and format the playlist information
        formatted_playlist = {
            'id': playlist_data.get('id'),
            'name': playlist_data.get('name'),
            'description': playlist_data.get('description'),
            'public': playlist_data.get('public'),
            'collaborative': playlist_data.get('collaborative'),
            'uri': playlist_data.get('uri'),
            'href': playlist_data.get('href'),
            'type': playlist_data.get('type'),
            'snapshot_id': playlist_data.get('snapshot_id'),
            'owner': {
                'id': playlist_data.get('owner', {}).get('id'),
                'display_name': playlist_data.get('owner', {}).get('display_name'),
                'type': playlist_data.get('owner', {}).get('type'),
                'uri': playlist_data.get('owner', {}).get('uri'),
                'href': playlist_data.get('owner', {}).get('href')
            },
            'images': playlist_data.get('images', []),
            'external_urls': playlist_data.get('external_urls', {}),
            'tracks': {
                'href': playlist_data.get('tracks', {}).get('href'),
                'total': playlist_data.get('tracks', {}).get('total', 0),
                'limit': playlist_data.get('tracks', {}).get('limit', 0),
                'offset': playlist_data.get('tracks', {}).get('offset', 0),
                'next': playlist_data.get('tracks', {}).get('next'),
                'previous': playlist_data.get('tracks', {}).get('previous'),
                'items': []
            }
        }
        
        # Format track items
        tracks = playlist_data.get('tracks', {}).get('items', [])
        for item in tracks:
            track = item.get('track', {})
            if track and track.get('type') == 'track':  # Only include actual tracks, not episodes
                album = track.get('album', {})
                artists = track.get('artists', [])
                
                formatted_track = {
                    'id': track.get('id'),
                    'name': track.get('name'),
                    'uri': track.get('uri'),
                    'href': track.get('href'),
                    'type': track.get('type'),
                    'popularity': track.get('popularity'),
                    'duration_ms': track.get('duration_ms'),
                    'explicit': track.get('explicit'),
                    'external_urls': track.get('external_urls', {}),
                    'preview_url': track.get('preview_url'),
                    'album': {
                        'id': album.get('id'),
                        'name': album.get('name'),
                        'uri': album.get('uri'),
                        'href': album.get('href'),
                        'type': album.get('type'),
                        'release_date': album.get('release_date'),
                        'release_date_precision': album.get('release_date_precision'),
                        'images': album.get('images', []),
                        'external_urls': album.get('external_urls', {})
                    },
                    'artists': [
                        {
                            'id': artist.get('id'),
                            'name': artist.get('name'),
                            'uri': artist.get('uri'),
                            'href': artist.get('href'),
                            'type': artist.get('type'),
                            'external_urls': artist.get('external_urls', {})
                        }
                        for artist in artists
                    ],
                    'added_at': item.get('added_at'),
                    'added_by': {
                        'id': item.get('added_by', {}).get('id'),
                        'display_name': item.get('added_by', {}).get('display_name'),
                        'type': item.get('added_by', {}).get('type'),
                        'uri': item.get('added_by', {}).get('uri'),
                        'href': item.get('added_by', {}).get('href')
                    } if item.get('added_by') else None
                }
                formatted_playlist['tracks']['items'].append(formatted_track)
        
        return jsonify(formatted_playlist)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return jsonify({'error': 'Playlist not found'}), 404
        elif e.response.status_code == 401:
            return jsonify({'error': 'Invalid or expired access token'}), 401
        elif e.response.status_code == 403:
            return jsonify({'error': 'Insufficient permissions to access this playlist'}), 403
        else:
            return jsonify({'error': f'Spotify API error: {e.response.status_code}'}), e.response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/get-playlist', methods=['POST', 'OPTIONS'])
def get_playlist():
    """Get a playlist owned by a Spotify user"""
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.get_json()
    playlist_id = data.get('playlist_id')
    spotify_token = data.get('spotify_token')
    market = data.get('market')  # Optional ISO 3166-1 alpha-2 country code
    fields = data.get('fields')  # Optional fields filter
    
    # Input validation
    if not playlist_id or not spotify_token:
        return jsonify({'error': 'Missing required parameters: playlist_id and spotify_token'}), 400
    
    return get_playlist_internal(playlist_id, spotify_token, market, fields)

# --- Enhanced Relevance Scoring ---
def calculate_relevance_score(entity, user_artists, user_genres, context_type, user_country, location, user_preferences=None):
    """
    Calculate a relevance score for an entity based on user preferences and context
    Higher scores indicate more relevant recommendations
    """
    try:
        relevance_score = 0.0
        
        # Base score from popularity (but weighted lower)
        popularity = entity.get('popularity', 0)
        if isinstance(popularity, (int, float)):
            relevance_score += popularity * 0.2  # Reduced weight for popularity
        
        # User preference matching
        entity_name = entity.get('name', '')
        if isinstance(entity_name, str):
            entity_name = entity_name.lower()
        else:
            entity_name = str(entity_name).lower()
        
        # Handle tags that might be dictionaries or strings
        entity_tags = []
        for tag in entity.get('tags', []):
            if isinstance(tag, str):
                entity_tags.append(tag.lower())
            elif isinstance(tag, dict) and 'name' in tag:
                entity_tags.append(tag['name'].lower())
            elif isinstance(tag, dict) and 'tag' in tag:
                entity_tags.append(tag['tag'].lower())
        
        entity_properties = entity.get('properties', {})
        
        # Enhanced artist preference matching with priority scores
        if user_preferences and user_preferences.get('artist_priority_scores'):
            artist_priority_scores = user_preferences.get('artist_priority_scores', {})
            favorite_artists = user_preferences.get('favorite_artists', [])
            
            # Check for favorite artists (highest priority)
            for artist in favorite_artists:
                artist_lower = artist.lower()
                if artist_lower in entity_name or entity_name in artist_lower:
                    priority_score = artist_priority_scores.get(artist, 0)
                    relevance_score += 4.0 + (priority_score * 0.2)  # Extra high boost for favorites
                    print(f"[ARTIST MATCH] Favorite artist '{artist}' matched - Score: {4.0 + (priority_score * 0.2):.1f}")
                # Check for similar artists in tags
                if any(artist_lower in tag for tag in entity_tags):
                    priority_score = artist_priority_scores.get(artist, 0)
                    relevance_score += 2.5 + (priority_score * 0.1)
            
            # Check for top artists (high priority)
            for artist in user_artists:
                artist_lower = artist.lower()
                if artist_lower in entity_name or entity_name in artist_lower:
                    priority_score = artist_priority_scores.get(artist, 0)
                    relevance_score += 2.0 + (priority_score * 0.1)  # High boost with priority bonus
                # Check for similar artists in tags
                if any(artist_lower in tag for tag in entity_tags):
                    priority_score = artist_priority_scores.get(artist, 0)
                    relevance_score += 1.5 + (priority_score * 0.05)
        else:
            # Fallback to basic artist matching
            for artist in user_artists:
                artist_lower = artist.lower()
                if artist_lower in entity_name or entity_name in artist_lower:
                    relevance_score += 2.0  # High boost for direct artist match
                # Check for similar artists in tags
                if any(artist_lower in tag for tag in entity_tags):
                    relevance_score += 1.5
        
        # Enhanced genre matching
        for genre in user_genres:
            genre_lower = genre.lower()
            if genre_lower in entity_name or any(genre_lower in tag for tag in entity_tags):
                relevance_score += 1.0  # Boost for genre match
                print(f"[GENRE MATCH] Genre '{genre}' matched - Score: 1.0")
        
        # Enhanced context matching
        context_keywords = get_context_keywords(context_type)
        for keyword in context_keywords:
            if keyword in entity_name or any(keyword in tag for tag in entity_tags):
                relevance_score += 1.5  # Boost for context match
                print(f"[CONTEXT MATCH] Context keyword '{keyword}' matched - Score: 1.5")
        
        # Enhanced cultural relevance
        cultural_keywords = get_cultural_keywords(user_country, location)
        for keyword in cultural_keywords:
            if keyword in entity_name or any(keyword in tag for tag in entity_tags):
                relevance_score += 0.5  # Boost for cultural relevance
                print(f"[CULTURAL MATCH] Cultural keyword '{keyword}' matched - Score: 0.5")
        
        # Check if entity matches user's genre preferences
        for genre in user_genres:
            genre_lower = genre.lower()
            if genre_lower in entity_name or any(genre_lower in tag for tag in entity_tags):
                relevance_score += 1.0
        
        # Context-based scoring
        if context_type:
            context_keywords = get_context_keywords(context_type)
            for keyword in context_keywords:
                if keyword in entity_name or any(keyword in tag for tag in entity_tags):
                    relevance_score += 0.8
        
        # Cultural relevance based on user location/country
        if user_country and location:
            cultural_keywords = get_cultural_keywords(user_country, location)
            for keyword in cultural_keywords:
                if keyword in entity_name or any(keyword in tag for tag in entity_tags):
                    relevance_score += 0.6
        
        # Enhanced user preference matching
        if user_preferences:
            # Check recent listening patterns with priority
            if user_preferences.get('recent_artists'):
                for recent_artist in user_preferences['recent_artists']:
                    if recent_artist.lower() in entity_name:
                        priority_score = user_preferences.get('artist_priority_scores', {}).get(recent_artist, 0)
                        relevance_score += 1.0 + (priority_score * 0.1)  # Bonus for recently listened artists with priority
                        print(f"[RECENT ARTIST] Recent artist '{recent_artist}' matched - Score: {1.0 + (priority_score * 0.1):.1f}")
            
            # Check most played artists (highest priority - actual listening behavior)
            if user_preferences.get('most_played_artists'):
                for most_played_artist in user_preferences['most_played_artists']:
                    if most_played_artist.lower() in entity_name:
                        play_count = user_preferences.get('artist_play_counts', {}).get(most_played_artist, 0)
                        relevance_score += 3.0 + (play_count * 0.2)  # High bonus for most played artists
                        print(f"[MOST PLAYED ARTIST] Most played artist '{most_played_artist}' matched - Score: {3.0 + (play_count * 0.2):.1f}")
                    # Check for similar artists in tags
                    if any(most_played_artist.lower() in tag for tag in entity_tags):
                        play_count = user_preferences.get('artist_play_counts', {}).get(most_played_artist, 0)
                        relevance_score += 2.0 + (play_count * 0.1)
                        print(f"[MOST PLAYED ARTIST TAG] Most played artist '{most_played_artist}' tag matched - Score: {2.0 + (play_count * 0.1):.1f}")
            
            # Check most played songs for direct matches
            if user_preferences.get('most_played_songs'):
                for most_played_song in user_preferences['most_played_songs']:
                    if most_played_song.lower() in entity_name or entity_name.lower() in most_played_song.lower():
                        song_play_count = user_preferences.get('song_play_counts', {}).get(most_played_song, 0)
                        relevance_score += 4.0 + (song_play_count * 0.3)  # Very high bonus for direct song matches
                        print(f"[MOST PLAYED SONG] Most played song '{most_played_song}' matched - Score: {4.0 + (song_play_count * 0.3):.1f}")
            
            # Check mood indicators
            if user_preferences.get('mood_indicators'):
                for mood in user_preferences['mood_indicators']:
                    mood_keywords = get_context_keywords(mood)
                    for keyword in mood_keywords:
                        if keyword in entity_name or any(keyword in tag for tag in entity_tags):
                            relevance_score += 0.5
        
        # Recency bonus (for movies, books, etc.)
        if 'release_year' in entity_properties or 'publication_year' in entity_properties:
            year = entity_properties.get('release_year') or entity_properties.get('publication_year')
            if year and isinstance(year, str) and year.isdigit():
                try:
                    year_int = int(year)
                    current_year = datetime.now().year
                    if year_int >= current_year - 5:  # Recent content gets bonus
                        relevance_score += 0.4
                except (ValueError, TypeError):
                    pass  # Skip if year parsing fails
        
        # Diversity bonus (penalize if too similar to other recommendations)
        # This will be handled in the calling function
        
        return round(relevance_score, 2)
        
    except Exception as e:
        print(f"Error in calculate_relevance_score: {e}")
        return 0.5  # Return default score if calculation fails

def get_context_keywords(context_type):
    """Get relevant keywords for different context types"""
    context_keywords = {
        "soccer": ["sport", "energy", "motivation", "team", "victory", "champion"],
        "workout": ["energy", "motivation", "strength", "power", "intense", "pump"],
        "melancholic": ["sad", "emotional", "deep", "thoughtful", "melancholy", "nostalgic"],
        "relaxing": ["calm", "peaceful", "chill", "ambient", "soothing", "tranquil"],
        "upbeat": ["happy", "energetic", "positive", "uplifting", "cheerful", "vibrant"],
        "driving": ["road", "journey", "adventure", "freedom", "travel", "motion"],
        "cooking": ["home", "comfort", "warm", "cozy", "family", "nourishing"],
        "studying": ["focus", "concentration", "intellectual", "academic", "learning", "mindful"],
        "party": ["celebration", "fun", "dance", "social", "exciting", "festive"],
        "romantic": ["love", "romance", "intimate", "passionate", "tender", "sweet"],
        "morning": ["fresh", "new", "awakening", "bright", "optimistic", "start"],
        "evening": ["winding", "reflection", "calm", "transition", "peaceful", "quiet"],
        "night": ["dark", "mysterious", "intimate", "deep", "contemplative", "dreamy"],
        "rainy": ["melancholy", "cozy", "reflective", "peaceful", "nostalgic", "comforting"],
        "sunny": ["bright", "energetic", "positive", "warm", "outdoor", "vibrant"]
    }
    return context_keywords.get(context_type, [])

def get_cultural_keywords(user_country, location):
    """Get culturally relevant keywords based on user location"""
    cultural_keywords = {
        "IN": ["indian", "bollywood", "desi", "hindi", "punjabi", "tamil", "telugu", "bengali"],
        "US": ["american", "hollywood", "western", "english", "pop", "rock", "hip hop"],
        "GB": ["british", "uk", "english", "indie", "alternative", "folk"],
        "CA": ["canadian", "north american", "indie", "folk", "alternative"],
        "AU": ["australian", "indie", "alternative", "folk", "rock"],
        "DE": ["german", "european", "electronic", "techno", "industrial"],
        "FR": ["french", "european", "chanson", "electronic", "indie"],
        "JP": ["japanese", "asian", "j-pop", "j-rock", "anime", "manga"],
        "KR": ["korean", "asian", "k-pop", "k-drama", "korean wave"],
        "BR": ["brazilian", "latin", "samba", "bossa nova", "tropical"],
        "MX": ["mexican", "latin", "spanish", "mariachi", "ranchera"],
        "ES": ["spanish", "european", "flamenco", "latin", "mediterranean"]
    }
    
    # Get country-specific keywords
    country_keywords = cultural_keywords.get(user_country, [])
    
    # Add location-specific keywords
    location_keywords = []
    if "mumbai" in location.lower():
        location_keywords.extend(["bollywood", "marathi", "mumbai", "bombay"])
    elif "new york" in location.lower():
        location_keywords.extend(["new york", "nyc", "american", "urban"])
    elif "london" in location.lower():
        location_keywords.extend(["london", "british", "uk", "english"])
    
    return country_keywords + location_keywords

def sort_by_relevance(entities, user_artists, user_genres, context_type, user_country, location, user_preferences=None):
    """
    Sort entities by relevance score instead of popularity
    """
    if not entities:
        return entities
    
    # Calculate relevance scores for all entities
    scored_entities = []
    seen_artists = set()  # Track diversity
    
    for entity in entities:
        try:
            relevance_score = calculate_relevance_score(
                entity, user_artists, user_genres, context_type, user_country, location, user_preferences
            )
            
            # Apply diversity penalty for repeated artists/similar content
            entity_name = entity.get('name', '')
            if isinstance(entity_name, str) and entity_name.strip():
                entity_artist = entity_name.split()[0].lower()  # Simple artist extraction
                if entity_artist in seen_artists:
                    relevance_score -= 0.5  # Penalty for lack of diversity
                else:
                    seen_artists.add(entity_artist)
            
            scored_entity = entity.copy()
            scored_entity['relevance_score'] = relevance_score
            scored_entities.append(scored_entity)
            
        except Exception as e:
            print(f"Error calculating relevance score for entity {entity.get('name', 'Unknown')}: {e}")
            # Add entity with default score if calculation fails
            scored_entity = entity.copy()
            scored_entity['relevance_score'] = 0.5  # Default fallback score
            scored_entities.append(scored_entity)
    
    # Sort by relevance score (highest first)
    scored_entities.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    return scored_entities

def get_user_listening_preferences(access_token):
    """
    Get comprehensive user listening preferences including genres, moods, and patterns
    """
    preferences = {
        'genres': set(),
        'artists': [],
        'tracks': [],
        'recent_artists': [],
        'mood_indicators': []
    }
    
    try:
        # Get top artists with genres
        top_artists = get_spotify_top_artists_with_images(access_token, limit=20)
        for artist in top_artists:
            preferences['artists'].append(artist['name'])
            preferences['genres'].update(artist.get('genres', []))
        
        # Get recently played for mood/context
        recent_artists = get_spotify_recently_played(access_token, limit=10)
        preferences['recent_artists'] = recent_artists
        
        # Get top tracks
        top_tracks = get_spotify_top_tracks(access_token, limit=20)
        preferences['tracks'] = top_tracks
        
        # Analyze mood indicators from track names and artists
        all_text = ' '.join(preferences['artists'] + preferences['tracks']).lower()
        mood_keywords = {
            'energetic': ['energy', 'power', 'strong', 'intense', 'pump', 'workout'],
            'calm': ['chill', 'calm', 'peaceful', 'quiet', 'soft', 'gentle'],
            'happy': ['happy', 'joy', 'fun', 'bright', 'sunny', 'cheerful'],
            'melancholic': ['sad', 'melancholy', 'blue', 'lonely', 'heartbreak'],
            'romantic': ['love', 'romance', 'heart', 'sweet', 'tender', 'passion']
        }
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                preferences['mood_indicators'].append(mood)
        
        # Convert sets to lists for JSON serialization
        preferences['genres'] = list(preferences['genres'])
        
    except Exception as e:
        print(f"Error getting user preferences: {e}")
    
    return preferences

def analyze_user_artist_preferences(access_token, context_type=None, language_preference=None, mood_preference=None):
    """Analyze user's artist preferences"""
    try:
        # Get user data from Spotify
        short_term_artists = get_spotify_top_artists(access_token, limit=20, time_range="short_term")
        medium_term_artists = get_spotify_top_artists(access_token, limit=20, time_range="medium_term")
        long_term_artists = get_spotify_top_artists(access_token, limit=20, time_range="long_term")
        recent_artists = get_spotify_recently_played(access_token, limit=30)
        
        # Get playlist artists (simplified version)
        playlist_artists = []
        try:
            # Get user playlists
            url = "https://api.spotify.com/v1/me/playlists"
            headers = {"Authorization": f"Bearer {access_token}"}
            params = {"limit": 10}
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            playlists = resp.json().get("items", [])
            
            # Get tracks from first few playlists
            for playlist in playlists[:5]:
                playlist_id = playlist["id"]
                tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
                tracks_resp = requests.get(tracks_url, headers=headers, params={"limit": 20})
                if tracks_resp.status_code == 200:
                    tracks_data = tracks_resp.json().get("items", [])
                    for track_item in tracks_data:
                        track = track_item.get("track", {})
                        if track and track.get("artists"):
                            artist_name = track["artists"][0].get("name", "")
                            if artist_name:
                                playlist_artists.append(artist_name)
        except Exception as e:
            print(f"Error getting playlist artists: {e}")
        
        # Combine all artists
        from collections import Counter
        all_artists = []
        all_artists.extend(short_term_artists)
        all_artists.extend(medium_term_artists)
        all_artists.extend(long_term_artists)
        all_artists.extend(recent_artists)
        all_artists.extend(playlist_artists)
        
        artist_counts = Counter(all_artists)
        
        # Apply language filtering if specified
        if language_preference and language_preference.get('primary_language') != 'any':
            original_artists = list(artist_counts.keys())
            filtered_artists = filter_artists_by_language(original_artists, language_preference, access_token)
            
            if filtered_artists:
                for artist in filtered_artists:
                    if artist in artist_counts:
                        artist_counts[artist] *= 1.5
        
        # Calculate priority scores
        artist_priority_scores = {}
        for artist, count in artist_counts.items():
            priority_score = 0
            
            if artist in short_term_artists:
                priority_score += 10
                short_term_index = short_term_artists.index(artist)
                priority_score += (20 - short_term_index) * 0.5
            
            if artist in medium_term_artists:
                priority_score += 8
                medium_term_index = medium_term_artists.index(artist)
                priority_score += (20 - medium_term_index) * 0.4
            
            if artist in long_term_artists:
                priority_score += 6
                long_term_index = long_term_artists.index(artist)
                priority_score += (20 - long_term_index) * 0.3
            
            if artist in recent_artists:
                priority_score += 5
                recent_index = recent_artists.index(artist)
                priority_score += (30 - recent_index) * 0.2
            
            if artist in playlist_artists:
                priority_score += 4
                playlist_count = playlist_artists.count(artist)
                priority_score += playlist_count * 0.5
            
            priority_score += count * 0.3
            artist_priority_scores[artist] = round(priority_score, 2)
        
        # Sort by priority score
        sorted_artists = sorted(artist_priority_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_artists": [artist for artist, score in sorted_artists[:10]],
            "artist_priority_scores": artist_priority_scores,
            "artist_frequency": dict(artist_counts.most_common(50))
        }
        
    except Exception as e:
        print(f"Error analyzing user artist preferences: {e}")
        return {"top_artists": [], "artist_priority_scores": {}, "artist_frequency": {}}

def get_enhanced_user_preferences(access_token, context_type=None, language_preference=None, mood_preference=None):
    """Get enhanced user preferences"""
    try:
        # Get basic artist preferences
        artist_preferences = analyze_user_artist_preferences(access_token, context_type, language_preference, mood_preference)
        
        # Get user profile
        user_profile = get_spotify_user_profile(access_token)
        user_country = user_profile.get("country") if user_profile else None
        
        # Get location
        location = get_location_from_country(user_country) if user_country else "New York"
        
        return {
            "artist_preferences": artist_preferences,
            "user_profile": user_profile,
            "user_country": user_country,
            "location": location
        }
        
    except Exception as e:
        print(f"Error getting enhanced user preferences: {e}")
        return {}

def validate_cultural_tags(tags, qloo_api_key):
    """Test and validate which cultural tags are recognized by Qloo's database"""
    print(f"[VALIDATION] Testing {len(tags)} tags with Qloo database...")
    
    valid_tags = []
    invalid_tags = []
    
    headers = {
        "X-API-KEY": qloo_api_key,
        "Content-Type": "application/json"
    }
    
    for tag in tags:
        try:
            params = {"filter.query": tag, "limit": 1}
            resp = requests.get("https://hackathon.api.qloo.com/v2/tags", headers=headers, params=params)
            if resp.ok and resp.json().get("results", {}).get("tags"):
                valid_tags.append(tag)
                print(f"[VALIDATION] ✓ '{tag}' - VALID")
            else:
                invalid_tags.append(tag)
                print(f"[VALIDATION] ✗ '{tag}' - NOT FOUND")
            time.sleep(0.1)
        except Exception as e:
            invalid_tags.append(tag)
            print(f"[VALIDATION] ✗ '{tag}' - ERROR: {e}")
    
    print(f"[VALIDATION] Results: {len(valid_tags)} valid, {len(invalid_tags)} invalid")
    print(f"[VALIDATION] Valid tags: {valid_tags}")
    print(f"[VALIDATION] Invalid tags: {invalid_tags}")
    
    return valid_tags, invalid_tags

def ai_based_track_filtering(tracks, context_type, user_context, gemini_api_key, limit=10):
    """AI-based track filtering using Gemini to analyze track names and context"""
    if not tracks:
        return tracks[:limit]
    
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[AI FILTER] No valid Gemini API key provided, returning tracks without AI filtering")
        return tracks[:limit]
    
    print(f"[AI FILTER] Filtering {len(tracks)} tracks for context: {context_type}")
    
    # Create track analysis prompt
    prompt = f"""
You are a music recommendation expert. Analyze these track names and determine if they match the user's context.

User Context: "{user_context}"
Context Type: {context_type}

For each track, respond with ONLY "YES" or "NO" based on whether the track name suggests it would be appropriate for this context.

Track names to analyze:
"""
    
    # Add track names to prompt
    for i, track in enumerate(tracks[:20]):  # Limit to first 20 for efficiency
        track_name = track.get("name", "Unknown")
        artist_name = track.get("artist", "Unknown")
        prompt += f"{i+1}. {track_name} by {artist_name}\n"
    
    prompt += "\nRespond with only YES/NO for each track, separated by commas."
    
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': gemini_api_key
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        result = data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Parse YES/NO responses
        responses = [r.strip().upper() for r in result.split(',')]
        filtered_tracks = []
        
        for i, track in enumerate(tracks[:20]):
            if i < len(responses) and responses[i] == "YES":
                filtered_tracks.append(track)
                print(f"[AI FILTER] ✓ {track.get('name', 'Unknown')} - APPROVED")
            else:
                print(f"[AI FILTER] ✗ {track.get('name', 'Unknown')} - REJECTED")
        
        print(f"[AI FILTER] AI filtered {len(filtered_tracks)}/{len(tracks[:20])} tracks")
        return filtered_tracks[:limit]
        
    except Exception as e:
        print(f"[AI FILTER] Error in AI filtering: {e}")
        # Fallback to basic filtering
        return tracks[:limit]

def validate_context_detection(user_context, detected_context, gemini_api_key):
    """Validate if the detected context matches the user input"""
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[CONTEXT VALIDATION] No valid Gemini API key provided, using detected context as-is")
        return detected_context
    
    prompt = f"""
You are a context validation expert. 

User Input: "{user_context}"
Detected Context: "{detected_context}"

Does the detected context accurately represent the user's input? 

Consider:
- If user mentions "breakup", "sad", "heartbreak" → should be "breakup" or "sad"
- If user mentions "party", "celebration", "victory" → should be "party" or "upbeat"
- If user mentions "workout", "gym", "exercise" → should be "workout"
- If user mentions "study", "work", "focus" → should be "studying"

Respond with ONLY "CORRECT" or "INCORRECT" followed by the correct context type.
Example: "CORRECT" or "INCORRECT: breakup"
"""
    
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': gemini_api_key
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        result = data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        if result.startswith("CORRECT"):
            print(f"[CONTEXT VALIDATION] ✓ Context '{detected_context}' is correct for '{user_context}'")
            return detected_context
        elif result.startswith("INCORRECT:"):
            corrected_context = result.split(":", 1)[1].strip()
            print(f"[CONTEXT VALIDATION] ✗ Context '{detected_context}' is wrong for '{user_context}', corrected to '{corrected_context}'")
            return corrected_context
        else:
            print(f"[CONTEXT VALIDATION] ? Could not validate context, using '{detected_context}'")
            return detected_context
            
    except Exception as e:
        print(f"[CONTEXT VALIDATION] Error validating context: {e}")
        return detected_context

def ai_calculate_relevance_score(entity, user_context, user_artists, user_genres, context_type, user_country, location, user_preferences=None, gemini_api_key=None):
    """
    AI-powered relevance scoring using Gemini to provide more intelligent and contextual scores
    """
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        # Fallback to traditional scoring if no valid AI key
        return calculate_relevance_score(entity, user_artists, user_genres, context_type, user_country, location, user_preferences)
    
    try:
        entity_name = entity.get('name', '')
        entity_artist = entity.get('artist', '')
        entity_tags = []
        
        # Extract tags
        for tag in entity.get('tags', []):
            if isinstance(tag, str):
                entity_tags.append(tag)
            elif isinstance(tag, dict) and 'name' in tag:
                entity_tags.append(tag['name'])
            elif isinstance(tag, dict) and 'tag' in tag:
                entity_tags.append(tag['tag'])
        
        # Create comprehensive prompt for AI analysis with enhanced user data, artist priority, and most played songs
        prompt = f"""
You are an expert music recommendation AI. Analyze the relevance of this track for the user.

TRACK INFORMATION:
- Name: {entity_name}
- Artist: {entity_artist}
- Tags: {', '.join(entity_tags[:10])}  # Limit to first 10 tags
- Popularity: {entity.get('popularity', 'Unknown')}

USER CONTEXT:
- User Input: "{user_context}"
- Context Type: {context_type}
- User Country: {user_country}
- User Location: {location}

ENHANCED USER PREFERENCES:
- Favorite Artists (Highest Priority): {user_preferences.get('favorite_artists', [])[:5] if user_preferences else []}
- Top Artists: {user_preferences.get('top_artists', [])[:10] if user_preferences else []}
- Recent Artists: {user_preferences.get('recent_artists', [])[:5] if user_preferences else []}
- Playlist Artists: {user_preferences.get('playlist_artists', [])[:5] if user_preferences else []}
- Most Played Artists: {user_preferences.get('most_played_artists', [])[:5] if user_preferences else []}
- User's Genres: {', '.join(user_genres[:5])}  # Top 5 genres
- Playlist Themes: {user_preferences.get('playlist_themes', [])[:5] if user_preferences else []}
- Mood Indicators: {user_preferences.get('mood_indicators', [])[:3] if user_preferences else []}
- Total Playlists: {user_preferences.get('total_playlists', 0) if user_preferences else 0}
- Listening Patterns: {user_preferences.get('listening_patterns', {}) if user_preferences else {}}

MOST PLAYED SONGS ANALYSIS:
- Most Played Songs: {user_preferences.get('most_played_songs', [])[:10] if user_preferences else []}
- Song Play Counts: {dict(list(user_preferences.get('song_play_counts', {}).items())[:5]) if user_preferences else {}}
- Artist Play Counts: {dict(list(user_preferences.get('artist_play_counts', {}).items())[:5]) if user_preferences else {}}

ARTIST PRIORITY ANALYSIS:
- Artist Priority Scores: {dict(list(user_preferences.get('artist_priority_scores', {}).items())[:10]) if user_preferences else {}}
- Artist Frequency: {dict(list(user_preferences.get('artist_frequency', {}).items())[:10]) if user_preferences else {}}

ANALYSIS TASK:
Rate the relevance of this track from 0.0 to 10.0 based on:

CONTEXT MATCHING (40% weight):
- How perfectly does this track match the user's emotional state and situation?
- Does the track's mood, tempo, and lyrical content align with the user's context?
- For breakup context: prioritize sad, melancholic, heartbreak songs
- For party context: prioritize upbeat, energetic, danceable songs
- For work/study: prioritize calm, ambient, instrumental tracks

ARTIST PREFERENCE MATCHING (35% weight):
- Give MAXIMUM priority (9-10 score) to tracks by artists in user's "Favorite Artists" list
- Give HIGH priority (7-9 score) to tracks by artists in user's "Most Played Artists" list
- Give MEDIUM priority (5-7 score) to tracks by artists in user's "Top Artists" list
- Consider artist similarity and genre overlap with user's preferred artists

CULTURAL & REGIONAL RELEVANCE (15% weight):
- For Indian users: prioritize Bollywood, Hindi Pop, Indian Classical, Regional music
- Consider user's location and cultural background
- Match with user's most played songs and artists

OVERALL TASTE ALIGNMENT (10% weight):
- Does this track fit the user's overall listening patterns?
- Consider playlist themes and mood indicators
- Match with user's preferred genres and styles

SCORING GUIDELINES:
- 9-10: Perfect match for context + favorite artist
- 8-9: Excellent context match + high-priority artist
- 7-8: Good context match + medium-priority artist
- 6-7: Decent context match + some artist preference
- 4-6: Moderate context match
- 2-4: Poor context match
- 0-2: Completely inappropriate for context

Respond with ONLY a number between 0.0 and 10.0 (e.g., "7.5" or "3.2")
"""
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': gemini_api_key
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        result = data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Parse the AI score
        try:
            ai_score = float(result)
            # Ensure score is within valid range
            ai_score = max(0.0, min(10.0, ai_score))
            
            # Combine AI score with traditional score for robustness
            traditional_score = calculate_relevance_score(entity, user_artists, user_genres, context_type, user_country, location, user_preferences)
            
            # Weight AI score higher (70%) but keep traditional score as backup (30%)
            final_score = (ai_score * 0.7) + (traditional_score * 0.3)
            
            print(f"[AI SCORING] {entity_name} by {entity_artist} - AI: {ai_score:.1f}, Traditional: {traditional_score:.1f}, Final: {final_score:.1f}")
            
            return round(final_score, 2)
            
        except (ValueError, TypeError):
            print(f"[AI SCORING] Failed to parse AI score '{result}', falling back to traditional scoring")
            return calculate_relevance_score(entity, user_artists, user_genres, context_type, user_country, location, user_preferences)
            
        except Exception as e:
            print(f"[AI SCORING] Error in AI scoring for {entity.get('name', 'Unknown')}: {e}")
            # Fallback to traditional scoring
            return calculate_relevance_score(entity, user_artists, user_genres, context_type, user_country, location, user_preferences)
    
    except Exception as e:
        print(f"[AI SCORING] Error in AI scoring for {entity.get('name', 'Unknown')}: {e}")
        # Fallback to traditional scoring
        return calculate_relevance_score(entity, user_artists, user_genres, context_type, user_country, location, user_preferences)

def ai_sort_by_relevance(entities, user_context, user_artists, user_genres, context_type, user_country, location, user_preferences=None, gemini_api_key=None):
    """
    AI-enhanced sorting by relevance score with intelligent analysis and context filtering
    """
    if not entities:
        return entities
    
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here" or gemini_api_key == DEFAULT_GEMINI_API_KEY:
        print(f"[AI SORTING] No valid Gemini API key provided, using traditional sorting")
        # Use traditional sorting instead
        return sort_by_relevance(entities, user_artists, user_genres, context_type, user_country, location, user_preferences)
    
    print(f"[AI SORTING] Sorting {len(entities)} entities with AI-powered relevance scoring")
    
    # Calculate AI-enhanced relevance scores for all entities
    scored_entities = []
    seen_artists = set()  # Track diversity
    
    for entity in entities:
        try:
            relevance_score = ai_calculate_relevance_score(
                entity, user_context, user_artists, user_genres, context_type, 
                user_country, location, user_preferences, gemini_api_key
            )
            
            # Apply diversity penalty for repeated artists/similar content
            entity_name = entity.get('name', '')
            if isinstance(entity_name, str) and entity_name.strip():
                entity_artist = entity_name.split()[0].lower()  # Simple artist extraction
                if entity_artist in seen_artists:
                    relevance_score -= 0.3  # Reduced penalty for AI scoring
                else:
                    seen_artists.add(entity_artist)
            
            scored_entity = entity.copy()
            scored_entity['relevance_score'] = relevance_score
            scored_entity['ai_scored'] = True  # Flag to indicate AI scoring was used
            scored_entities.append(scored_entity)
            
        except Exception as e:
            print(f"Error calculating AI relevance score for entity {entity.get('name', 'Unknown')}: {e}")
            # Add entity with traditional score if AI calculation fails
            traditional_score = calculate_relevance_score(entity, user_artists, user_genres, context_type, user_country, location, user_preferences)
            scored_entity = entity.copy()
            scored_entity['relevance_score'] = traditional_score
            scored_entity['ai_scored'] = False  # Flag to indicate traditional scoring was used
            scored_entities.append(scored_entity)
    
    # Sort by relevance score (highest first)
    scored_entities.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    # Apply context-based filtering to ensure better alignment
    filtered_entities = []
    context_keywords = get_context_keywords(context_type)
    
    for entity in scored_entities:
        entity_name = entity.get('name', '').lower()
        entity_artist = entity.get('artist', '').lower()
        score = entity.get('relevance_score', 0)
        
        # Check if entity matches context keywords
        context_match = any(keyword in entity_name or keyword in entity_artist for keyword in context_keywords)
        
        # For high-scoring entities or context matches, include them
        if score >= 7.0 or context_match:
            filtered_entities.append(entity)
        elif len(filtered_entities) < len(scored_entities) * 0.8:  # Keep top 80% even if lower score
            filtered_entities.append(entity)
    
    # If we filtered too many, add back some high-scoring ones
    if len(filtered_entities) < len(scored_entities) * 0.6:
        for entity in scored_entities:
            if entity not in filtered_entities and entity.get('relevance_score', 0) >= 5.0:
                filtered_entities.append(entity)
                if len(filtered_entities) >= len(scored_entities) * 0.6:
                    break
    
    print(f"[AI SORTING] Top 5 scores: {[(e.get('name', 'Unknown'), e.get('relevance_score', 0)) for e in filtered_entities[:5]]}")
    print(f"[AI SORTING] Sorted {len(filtered_entities)} tracks with AI, filtered {len(scored_entities) - len(filtered_entities)} low-relevance tracks")
    
    return filtered_entities



@app.route('/cross-domain-recommendations', methods=['POST', 'OPTIONS'])
def get_cross_domain_recommendations():
    """Enhanced cross-domain recommendations using the same unified function as /crossdomain-recommendations"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Clean up expired cache entries periodically
        cleanup_expired_cache()
        
        # Validate request data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        spotify_token = data.get('spotify_token', '')
        user_context = data.get('user_context', '')
        music_artists = data.get('music_artists', [])  # From music recommendation output
        top_scored_artists = data.get('top_scored_artists', [])  # Top 5 artists based on scores
        user_tags = data.get('user_tags', [])  # User-provided tags like sad, melancholic, etc.
        qloo_api_key = data.get('qloo_api_key', DEFAULT_QLOO_API_KEY)
        gemini_api_key = data.get('gemini_api_key', DEFAULT_GEMINI_API_KEY)
        recommendations_per_domain = data.get('limit', 10)  # Default to 10, but allow override
        
        # Validate required parameters
        if not spotify_token:
            return jsonify({'error': 'spotify_token is required'}), 400
        
        print(f"[CROSS-DOMAIN] Starting cross-domain recommendations with token length: {len(spotify_token)}")
        print(f"[CROSS-DOMAIN] User tags: {user_tags}")
        
        # Use the unified function
        response_data = generate_cross_domain_recommendations_unified(
            spotify_token=spotify_token,
            qloo_api_key=qloo_api_key,
            gemini_api_key=gemini_api_key,
            user_context=user_context,
            music_artists=music_artists,
            top_scored_artists=top_scored_artists,
            user_tags=user_tags,
            recommendations_per_domain=recommendations_per_domain
        )
        
        # Cache the result
        try:
            user_profile = get_spotify_user_profile(spotify_token)
            user_id = user_profile.get('id') if user_profile else 'unknown'
            cache_context = f"crossdomain_{hashlib.md5(user_context.encode()).hexdigest()[:8]}"
            cache_recommendation(user_id, cache_context, response_data, "crossdomain")
        except Exception as e:
            print(f"Could not cache result: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in cross-domain recommendations: {e}")
        return jsonify({'error': f'Failed to generate cross-domain recommendations: {str(e)}'}), 500
        
        # Get user ID for caching
        try:
            user_profile = get_spotify_user_profile(spotify_token)
            if not user_profile:
                return jsonify({
                    'error': 'Could not fetch user profile from Spotify API. The token may be invalid, expired, or have insufficient permissions.',
                    'token_length': len(spotify_token) if spotify_token else 0
                }), 400
            
            user_id = user_profile.get('id')
            if not user_id:
                return jsonify({'error': 'Could not fetch user profile - no user ID in response'}), 400
                
            print(f"[CROSS-DOMAIN] Successfully fetched user profile for ID: {user_id}")
            
            # Initialize progress tracking with more steps for artist processing
            crossdomain_progress[user_id] = {
                'current_step': 0,
                'total_steps': 25,  # 5 artists × 5 domains
                'current_artist': '',
                'current_domain': '',
                'percentage': 0,
                'status': 'starting'
            }
            print(f"[PROGRESS] Initialized progress tracking for user {user_id}")
        except Exception as e:
            print(f"[CROSS-DOMAIN] Error fetching user profile: {e}")
            return jsonify({'error': f'Failed to fetch user profile: {str(e)}'}), 400
    
        # Check cache first (use a generic context for cross-domain recs)
        cache_context = f"crossdomain_{hashlib.md5(user_context.encode()).hexdigest()[:8]}"
        cached_result = get_cached_recommendation(user_id, cache_context, "crossdomain")
        if cached_result:
            cached_result['from_cache'] = True
            cached_result['cache_timestamp'] = datetime.now().isoformat()
            return jsonify(cached_result)

        # Get user's profile including country and location
        try:
            user_profile = get_spotify_user_profile(spotify_token)
            user_country = user_profile.get('country', 'US') if user_profile else 'US'
            location = get_location_from_country(user_country)
            print(f"User country: {user_country} -> Location: {location}")
            
            # Update progress - profile loaded
            if user_id in crossdomain_progress:
                crossdomain_progress[user_id].update({
                    'current_step': 1,
                    'percentage': 4,
                    'status': 'profile_loaded'
                })
        except Exception as e:
            print(f"Could not fetch user profile: {e}")
            user_country = 'US'  # Default fallback
            location = "New York"

        # Use top scored artists if provided, otherwise fall back to music_artists or fetch from Spotify
        if top_scored_artists:
            print(f"[CROSS-DOMAIN] Using top scored artists: {top_scored_artists}")
            music_artists = top_scored_artists[:5]  # Use top 5 scored artists
            # Get detailed info for these artists
            try:
                user_artists_with_images = []
                for artist_name in music_artists:
                    # Try to get artist info from Spotify
                    try:
                        print(f"[DEBUG] Fetching genres for artist: {artist_name}")
                        artist_genres = get_spotify_artist_genres(artist_name, spotify_token)
                        print(f"[DEBUG] Retrieved genres for {artist_name}: {artist_genres}")
                        user_artists_with_images.append({
                            'name': artist_name,
                            'genres': artist_genres,  # Fixed: use the correct variable name
                            'image': None  # We'll get image later if needed
                        })
                    except Exception as e:
                        print(f"[ERROR] Failed to get genres for {artist_name}: {e}")
                        user_artists_with_images.append({
                            'name': artist_name,
                            'genres': [],
                            'image': None
                        })
            except Exception as e:
                print(f"Could not fetch artist details: {e}")
                user_artists_with_images = [{'name': name, 'genres': [], 'image': None} for name in music_artists]
        elif not music_artists:
            try:
                user_artists_with_images = get_spotify_top_artists_with_images(spotify_token, limit=6)
                music_artists = [artist['name'] for artist in user_artists_with_images]
            except Exception as e:
                print(f"Could not fetch top artists: {e}")
                music_artists = []
        else:
            # If music_artists provided, get their detailed info
            try:
                user_artists_with_images = get_spotify_top_artists_with_images(spotify_token, limit=6)
                # Use provided artists but get their detailed info
                music_artists = music_artists[:6]  # Limit to 6 artists
            except Exception as e:
                print(f"Could not fetch artist details: {e}")
                user_artists_with_images = [{'name': name, 'genres': [], 'image': None} for name in music_artists[:6]]

        # Get comprehensive user preferences
        try:
            user_preferences = get_user_listening_preferences(spotify_token)
            print(f"[PREFERENCES] User genres: {user_preferences.get('genres', [])}")
            print(f"[PREFERENCES] Mood indicators: {user_preferences.get('mood_indicators', [])}")
        except Exception as e:
            print(f"Could not fetch user preferences: {e}")
            user_preferences = {'genres': [], 'mood_indicators': []}
        
        # Update progress - preferences loaded
        if user_id in crossdomain_progress:
            crossdomain_progress[user_id].update({
                'current_step': 2,
                'percentage': 8,
                'status': 'preferences_loaded'
            })

        # Get artist genres for each artist
        artist_genres_map = {}
        for artist in user_artists_with_images:
            artist_name = artist['name']
            artist_genres = artist.get('genres', [])
            artist_genres_map[artist_name] = artist_genres
            print(f"[ARTIST] {artist_name}: genres = {artist_genres}")

        # Entity types for cross-domain recommendations
        ENTITY_TYPES = [
            ("urn:entity:artist", "music artist"),
            ("urn:entity:book", "book"),
            ("urn:entity:movie", "movie"),
            ("urn:entity:podcast", "podcast"),
            ("urn:entity:tv_show", "TV show")
        ]

        # Initialize Qloo client (using existing class structure)
        qloo_client = QlooAPIClient(qloo_api_key)
        all_recommendations = {}

        # Process each artist with rate limiting
        for i, artist_name in enumerate(music_artists[:5]):  # Only use last 5 unique artists
            print(f"Processing recommendations for: {artist_name}")
            
            # Update progress for current artist
            if user_id in crossdomain_progress:
                current_step = 4 + (i * 5)  # Start at step 4, each artist gets 5 steps
                percentage = min(100, int((current_step / crossdomain_progress[user_id]['total_steps']) * 100))
                crossdomain_progress[user_id].update({
                    'current_step': current_step,
                    'current_artist': artist_name,
                    'current_domain': '',
                    'percentage': percentage,
                    'status': f'processing_artist_{i+1}'
                })
                print(f"[PROGRESS] Processing artist {i+1}/5: {artist_name} - {percentage}%")
            
            # Add delay between artists to prevent rate limiting
            if i > 0:
                print(f"[RATE LIMIT] Waiting 2 seconds between artists to prevent rate limiting")
                time.sleep(2)
            
            # Search for artist in Qloo
            try:
                entity = qloo_client.search_entity(artist_name)
                if not entity:
                    print(f"Could not find Qloo entity for '{artist_name}'. Skipping.")
                    continue
                    
                entity_id = entity.get('entity_id', entity.get('id'))
                if not entity_id:
                    print(f"No entity ID found for '{artist_name}'. Skipping.")
                    continue
                    
                artist_ids = [entity_id]
                artist_results = {}
            except Exception as e:
                print(f"Error searching for artist '{artist_name}' in Qloo: {e}")
                continue

            # Get recommendations for each domain with rate limiting
            for domain_i, (etype, domain) in enumerate(ENTITY_TYPES):
                print(f"Getting {domain} recommendations for {artist_name}")
                
                # Update progress for current domain
                if user_id in crossdomain_progress:
                    current_step = 4 + (i * 5) + domain_i + 1  # Artist step + domain step
                    percentage = min(100, int((current_step / crossdomain_progress[user_id]['total_steps']) * 100))
                    crossdomain_progress[user_id].update({
                        'current_step': current_step,
                        'current_domain': domain,
                        'percentage': percentage,
                        'status': f'processing_{domain}'
                    })
                    print(f"[PROGRESS] Processing {domain} for {artist_name} - {percentage}%")
                
                # Add small delay between domains to prevent rate limiting
                if domain_i > 0:
                    time.sleep(0.5)
                
                try:
                    # Get dynamic tags for this domain with user country, location, and artist context
                    artist_genres = artist_genres_map.get(artist_name, [])
                    
                    # Combine user tags with dynamic tags for better recommendations
                    combined_tags = []
                    
                    # Add user-provided tags if available
                    if user_tags:
                        # Convert user tags to Qloo tag format
                        for tag in user_tags:
                            combined_tags.append({
                                'name': tag,
                                'id': f"user_tag_{tag}",
                                'source': 'user'
                            })
                    
                    # Try enhanced dynamic tag generation first with user context
                    dynamic_tags = get_dynamic_tags_for_domain_enhanced(
                        domain, qloo_api_key, user_country, location, limit=25, 
                        artist_name=artist_name, artist_genres=artist_genres, gemini_api_key=gemini_api_key
                    )
                    
                    # Add dynamic tags to combined list
                    if dynamic_tags:
                        for tag in dynamic_tags:
                            tag['source'] = 'dynamic'
                            combined_tags.append(tag)
                    
                    # If no dynamic tags, try genre-based domain tags as fallback
                    if not dynamic_tags and artist_genres:
                        print(f"[FALLBACK] Trying genre-based domain tags for {domain}")
                        tag_ids, dynamic_tags = get_domain_appropriate_tags(
                            artist_genres, domain, qloo_api_key, gemini_api_key, user_country, location
                        )
                        if dynamic_tags:
                            # Convert to the expected format
                            for tag, tag_id in zip(dynamic_tags, tag_ids):
                                if tag_id:
                                    combined_tags.append({
                                        'name': tag,
                                        'id': tag_id,
                                        'source': 'genre_fallback'
                                    })
                
                    if not combined_tags:
                        print(f"No tags found for {domain}. Skipping.")
                        continue
                    
                    tag_names = [tag['name'] for tag in combined_tags]
                    tag_ids = {tag['name']: tag['id'] for tag in combined_tags}
                    
                    # Ask Gemini to pick the best tag based on user context, artist, and user tags
                    context = f"{artist_name} and the domain '{domain}' with user context: {user_context} and user tags: {user_tags}"
                    gemini_tag = call_gemini_for_tag_enhanced(context, tag_names, gemini_api_key, user_country)
                
                    # Find the best working tag using robust fallback system
                    selected_tag, selected_tag_id = find_working_tag(
                        gemini_tag, tag_names, tag_ids, qloo_client, etype, artist_ids
                    )
                    
                    if not selected_tag:
                        print(f"No working tags found for {domain}")
                        continue
                    
                    # Get final recommendations - request more to have options for top 5 selection
                    rec = qloo_client.get_recommendations(etype, selected_tag_id, artist_ids, take=recommendations_per_domain * 3)
                    entities = rec.get("results", {}).get("entities", []) if rec else []
                    
                    # Clean up and format entities with rich metadata
                    cleaned_entities = []
                    for ent in entities:
                        if 'akas' in ent:
                            del ent['akas']
                        
                        # Extract rich metadata
                        properties = ent.get('properties', {})
                        image_url = None
                        if properties and 'image' in properties:
                            image_url = properties.get('image', {}).get('url')
                        
                        # Calculate affinity score based on popularity and cultural relevance
                        popularity = ent.get('popularity', 0)
                        cultural_relevance = ent.get('cultural_relevance', 0)
                        affinity_score = (popularity * 0.6) + (cultural_relevance * 0.4)
                        
                        # Enhance metadata with additional information
                        enhanced_properties = {
                            'image': {'url': image_url} if image_url else None,
                            'description': properties.get('description'),
                            'year': properties.get('year'),
                            'genre': properties.get('genre'),
                            'director': properties.get('director'),
                            'author': properties.get('author'),
                            'publisher': properties.get('publisher'),
                            'runtime': properties.get('runtime'),
                            'rating': properties.get('rating'),
                            'language': properties.get('language'),
                            'country': properties.get('country'),
                            'url': properties.get('url'),
                            'external_urls': properties.get('external_urls', {})
                        }
                        
                        # Add domain-specific metadata
                        if domain == "movie":
                            enhanced_properties.update({
                                'cast': properties.get('cast', []),
                                'plot': properties.get('plot'),
                                'imdb_rating': properties.get('imdb_rating'),
                                'box_office': properties.get('box_office')
                            })
                        elif domain == "TV show":
                            enhanced_properties.update({
                                'seasons': properties.get('seasons'),
                                'episodes': properties.get('episodes'),
                                'network': properties.get('network'),
                                'status': properties.get('status')
                            })
                        elif domain == "book":
                            enhanced_properties.update({
                                'isbn': properties.get('isbn'),
                                'pages': properties.get('pages'),
                                'series': properties.get('series'),
                                'awards': properties.get('awards')
                            })
                        elif domain == "podcast":
                            enhanced_properties.update({
                                'episodes_count': properties.get('episodes_count'),
                                'duration': properties.get('duration'),
                                'host': properties.get('host'),
                                'category': properties.get('category')
                            })
                        elif domain == "music artist":
                            enhanced_properties.update({
                                'followers': properties.get('followers'),
                                'albums': properties.get('albums'),
                                'genres': properties.get('genres', []),
                                'spotify_url': properties.get('spotify_url')
                            })
                        
                        cleaned_entities.append({
                            'name': ent.get('name', 'Unknown'),
                            'id': ent.get('id'),
                            'popularity': popularity,
                            'affinity_score': round(affinity_score, 2),
                            'cultural_relevance': cultural_relevance,
                            'tags': ent.get('tags', []),
                            'properties': enhanced_properties,
                            'selected_tag': selected_tag,
                            'source_artist': artist_name
                        })
                    
                    artist_results[domain] = cleaned_entities
                
                except Exception as e:
                    print(f"Error processing {domain} recommendations for {artist_name}: {e}")
                    continue
            
            all_recommendations[artist_name] = artist_results

        # Update progress - starting aggregation
        if user_id in crossdomain_progress:
            crossdomain_progress[user_id].update({
                'current_step': 29,
                'current_domain': '',
                'percentage': 90,
                'status': 'aggregating_results'
            })
            print(f"[PROGRESS] Aggregating results - 90%")
        
        # Aggregate results by domain across all artists with deduplication
        domain_aggregated = {}
        for domain_info in ENTITY_TYPES:
            domain = domain_info[1]
            domain_aggregated[domain] = []
            seen_entities = set()  # Track unique entities by name
            
            for artist_name, artist_results in all_recommendations.items():
                if domain in artist_results:
                    for entity in artist_results[domain]:
                        entity_name = entity.get('name', '').strip().lower()
                        
                        # Skip if we've already seen this entity
                        if entity_name in seen_entities:
                            continue
                        
                        seen_entities.add(entity_name)
                        entity['source_artist'] = artist_name
                        domain_aggregated[domain].append(entity)
        
        # Sort by relevance instead of popularity for all domains
        user_genres = []
        for artist_name in music_artists:
            artist_genres = artist_genres_map.get(artist_name, [])
            user_genres.extend(artist_genres)
        user_genres = list(set(user_genres))  # Remove duplicates
        
        # Update progress - sorting results
        if user_id in crossdomain_progress:
            crossdomain_progress[user_id].update({
                'current_step': 30,
                'current_domain': '',
                'percentage': 95,
                'status': 'sorting_results'
            })
            print(f"[PROGRESS] Sorting results - 95%")
        
        # Apply sorting and limiting to all domains
        for domain in domain_aggregated:
            domain_aggregated[domain] = sort_by_relevance(
                domain_aggregated[domain], 
                music_artists, 
                user_genres, 
                "upbeat",  # Default context for cross-domain
                user_country, 
                location,
                user_preferences
            )
            # Fixed: Return exactly top 10 recommendations per domain
            domain_aggregated[domain] = domain_aggregated[domain][:recommendations_per_domain]

        # Update progress - finalizing
        if user_id in crossdomain_progress:
            crossdomain_progress[user_id].update({
                'current_step': 31,
                'current_domain': '',
                'percentage': 98,
                'status': 'finalizing'
            })
            print(f"[PROGRESS] Finalizing recommendations - 98%")
        
        # Prepare response data
        response_data = {
            "top_artists": music_artists,
            "top_artists_with_images": user_artists_with_images,
            "top_scored_artists": top_scored_artists,  # Include the original top scored artists
            "recommendations_by_domain": domain_aggregated,
            "detailed_results": all_recommendations,
            "total_domains": len([d for d in domain_aggregated.values() if d]),
            "recommendations_per_domain": recommendations_per_domain,
            "from_cache": False,
            "generated_timestamp": datetime.now().isoformat(),
            "location_used": location,
            "user_country": user_country,
            "user_context": user_context,  # Include user context in response
            "user_tags": user_tags  # Include user tags in response
        }
        
        # Cache the result
        cache_recommendation(user_id, cache_context, response_data, "crossdomain")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[CROSS-DOMAIN] Error: {e}")
        return jsonify({'error': f'Failed to get cross-domain recommendations: {str(e)}'}), 500

@app.route('/artist-priority-recommendations', methods=['POST'])
def artist_priority_recommendations():
    """Enhanced recommendations with artist priority scoring"""
    
    data = request.get_json()
    user_context = data.get('user_context', '')
    spotify_token = data.get('spotify_token', '')
    gemini_api_key = data.get('gemini_api_key', DEFAULT_GEMINI_API_KEY)
    qloo_api_key = data.get('qloo_api_key', DEFAULT_QLOO_API_KEY)
    
    if not all([user_context, spotify_token, gemini_api_key, qloo_api_key]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Get user profile and location
        user_profile = get_spotify_user_profile(spotify_token)
        user_country = user_profile.get('country', 'US') if user_profile else 'US'
        location = get_location_from_country(user_country)
        
        # Detect context type first
        detected_context = detect_context_type_llm(user_context, gemini_api_key)
        
        # Get enhanced user preferences with artist priority analysis and mood filtering
        enhanced_preferences = get_enhanced_user_preferences(spotify_token, detected_context)
        
        # Get user data
        user_artists = get_spotify_top_artists(spotify_token, limit=20)
        user_tracks = get_spotify_top_tracks(spotify_token, limit=20)
        
        # Enhanced Gemini analysis
        enhanced_tags = call_gemini_for_enhanced_tags(user_context, gemini_api_key, user_country, location)
        cultural_context = generate_cultural_context(user_context, user_country, location, gemini_api_key)
        
        # Enhanced Qloo client
        qloo_client = EnhancedQlooClient(qloo_api_key)
        
        # Filter and rank tags to get 5 most relevant tags for music
        filtered_tags = filter_and_rank_tags_for_music(enhanced_tags, user_context, user_country, location, gemini_api_key)
        print(f"[TAG FILTERING] Filtered from {len(enhanced_tags)} to {len(filtered_tags)} most relevant tags")
        
        # Get tag IDs for filtered tags (exactly 5 tags)
        tag_ids = get_qloo_tag_ids(filtered_tags, qloo_api_key)
        
        # Get enhanced recommendations
        enhanced_recommendations = qloo_client.get_enhanced_recommendations(
            tag_ids=tag_ids,
            user_artists=user_artists,
            user_tracks=user_tracks,
            location=location,
            location_radius=50000,
            cultural_context=cultural_context,
            limit=30
        )
        
        # Generate playlist from recommendations with artist priority scoring
        playlist = []
        all_tracks = []
        
        # Collect all tracks from recommended artists
        for artist_info in enhanced_recommendations[:15]:
            artist_name = artist_info["name"]
            artist_id = get_spotify_artist_id(artist_name, spotify_token)
            if artist_id:
                tracks = get_artist_tracks_smart(artist_id, spotify_token, limit=5)
                for track in tracks[:2]:
                    track_data = {
                        "name": track.get("name", "Unknown"),
                        "artist": track.get("artist", artist_name),
                        "url": track.get("url", ""),
                        "context_score": track.get("context_score", 0.0),
                        "album_name": track.get("album_name", "Unknown"),
                        "album_art_url": track.get("album_art_url", None),
                        "release_year": track.get("release_year", ""),
                        "preview_url": track.get("preview_url", None),
                        "qloo_affinity_score": artist_info.get("affinity_score", 0),
                        "cultural_relevance": artist_info.get("cultural_relevance", 0),
                        "tags": artist_info.get("tags", [])
                    }
                    all_tracks.append(track_data)
        
        # Get user genres for AI scoring
        user_genres = []
        for artist in user_artists[:10]:
            genres = get_spotify_artist_genres(artist, spotify_token)
            user_genres.extend(genres)
        user_genres = list(set(user_genres))
        
        # Use AI-powered relevance scoring with artist priority enhancement
        if all_tracks:
            ai_sorted_tracks = ai_sort_by_relevance(
                entities=all_tracks,
                user_context=user_context,
                user_artists=user_artists,
                user_genres=user_genres,
                context_type=detect_context_type_llm(user_context, gemini_api_key),
                user_country=user_country,
                location=location,
                user_preferences=enhanced_preferences,
                gemini_api_key=gemini_api_key
            )
            
            playlist = ai_sorted_tracks  # Show all AI-sorted tracks
        else:
            playlist = all_tracks  # Show all tracks
        
        # Get cultural insights
        try:
            cultural_insights = qloo_client.get_cultural_insights(location, "music")
        except Exception as e:
            print(f"[WARNING] Cultural insights failed: {e}")
            cultural_insights = []
        
        # Artist priority showcase
        artist_priority_showcase = {
            "artist_priority_enabled": True,
            "total_artists_analyzed": len(enhanced_preferences.get('artist_priority_scores', {})),
            "favorite_artists_identified": len(enhanced_preferences.get('favorite_artists', [])),
            "top_artists_identified": len(enhanced_preferences.get('top_artists', [])),
            "recent_artists_tracked": len(enhanced_preferences.get('recent_artists', [])),
            "playlist_artists_analyzed": len(enhanced_preferences.get('playlist_artists', [])),
            "highest_priority_score": max(enhanced_preferences.get('artist_priority_scores', {}).values()) if enhanced_preferences.get('artist_priority_scores') else 0,
            "priority_features": [
                "Multi-term Artist Analysis",
                "Playlist Artist Extraction",
                "Recent Listening Tracking",
                "Frequency-based Scoring",
                "Priority Score Calculation",
                "AI-Enhanced Artist Matching"
            ]
        }
        
        response_data = {
            "playlist": playlist,  # Show all tracks, not limited to 15
            "enhanced_tags": enhanced_tags,
            "cultural_context": cultural_context,
            "qloo_artists": [artist["name"] for artist in enhanced_recommendations[:20]],
            "cultural_insights": cultural_insights[:5],
            "artist_priority_showcase": artist_priority_showcase,
            "user_artist_analysis": {
                "favorite_artists": enhanced_preferences.get('favorite_artists', [])[:10],
                "top_artists": enhanced_preferences.get('top_artists', [])[:15],
                "recent_artists": enhanced_preferences.get('recent_artists', [])[:10],
                "playlist_artists": enhanced_preferences.get('playlist_artists', [])[:10],
                "artist_priority_scores": dict(list(enhanced_preferences.get('artist_priority_scores', {}).items())[:10]),
                "artist_frequency": dict(list(enhanced_preferences.get('artist_frequency', {}).items())[:10])
            },
            "ai_scoring_info": {
                "ai_scoring_enabled": True,
                "artist_priority_scoring": True,
                "tracks_ai_scored": len([t for t in playlist if t.get('ai_scored', False)]),
                "total_tracks": len(playlist),
                "scoring_method": "Artist-Priority Enhanced AI Relevance Scoring",
                "ai_score_weight": 0.7,
                "traditional_score_weight": 0.3,
                "favorite_artist_boost": "4.0 + priority_score * 0.2",
                "top_artist_boost": "2.0 + priority_score * 0.1"
            },
            "enhanced_features": [
                "Cultural Intelligence",
                "Location-Aware Recommendations", 
                "Multi-Strategy Analysis",
                "Affinity Scoring",
                "Cross-Domain Insights",
                "Enhanced Gemini Integration",
                "Real-time Cultural Analysis",
                "AI-Powered Relevance Scoring",
                "Playlist-Based Enhancement",
                "Listening History Integration",
                "Artist Priority Scoring",
                "Multi-term Artist Analysis"
            ],
            "location_used": location,
            "user_country": user_country,
            "generated_timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in artist priority recommendations: {e}")
        return jsonify({'error': 'Failed to generate artist priority recommendations'}), 500


def generate_cross_domain_recommendations_unified(
    spotify_token, 
    qloo_api_key=None, 
    gemini_api_key=None,
    user_context="", 
    music_artists=None, 
    top_scored_artists=None, 
    user_tags=None,
    tracks=None,
    recommendations_per_domain=10
):
    """
    Unified function for generating cross-domain recommendations that can handle both use cases:
    1. Basic cross-domain recommendations (from MusicDashboard)
    2. Enhanced cross-domain recommendations with user context and tags (from CrossDomainRecommendations)
    """
    
    # Set default API keys
    qloo_api_key = qloo_api_key or DEFAULT_QLOO_API_KEY
    gemini_api_key = gemini_api_key or DEFAULT_GEMINI_API_KEY
    
    # Clear expired Gemini cache entries
    clear_expired_gemini_cache()
    
    print(f"[CROSS-DOMAIN UNIFIED] Starting unified cross-domain recommendations")
    print(f"[CROSS-DOMAIN UNIFIED] User context: {user_context}")
    print(f"[CROSS-DOMAIN UNIFIED] User tags: {user_tags}")
    print(f"[CROSS-DOMAIN UNIFIED] Tracks count: {len(tracks) if tracks else 0}")
    if tracks:
        print(f"[CROSS-DOMAIN UNIFIED] Sample tracks: {[track.get('name', 'Unknown') for track in tracks[:3]]}")
    
    # Get user ID for caching and progress tracking
    user_id = None
    user_profile = None
    
    try:
        print(f"[CROSS-DOMAIN UNIFIED] Attempting to fetch user profile with token: {spotify_token[:20]}..." if spotify_token else "[CROSS-DOMAIN UNIFIED] No Spotify token provided")
        
        if spotify_token:
            user_profile = get_spotify_user_profile(spotify_token)
            if not user_profile:
                print("[CROSS-DOMAIN UNIFIED] Could not fetch user profile from Spotify API - token may be expired")
                # Continue without user profile for testing
                user_id = "anonymous_user"
            else:
                user_id = user_profile.get('id')
                if not user_id:
                    print("[CROSS-DOMAIN UNIFIED] No user ID in Spotify response")
                    user_id = "anonymous_user"
                else:
                    print(f"[CROSS-DOMAIN UNIFIED] Successfully fetched user profile for ID: {user_id}")
        else:
            print("[CROSS-DOMAIN UNIFIED] No Spotify token provided, using anonymous user")
            user_id = "anonymous_user"
        
        # Initialize progress tracking
        crossdomain_progress[user_id] = {
            'current_step': 0,
            'total_steps': 25,  # 5 artists × 5 domains
            'current_artist': '',
            'current_domain': '',
            'percentage': 0,
            'status': 'starting'
        }
        print(f"[PROGRESS] Initialized progress tracking for user {user_id}")
    except Exception as e:
        print(f"[CROSS-DOMAIN UNIFIED] Error in user profile handling: {e}")
        # Use anonymous user as fallback
        user_id = "anonymous_user"
        user_profile = None
        crossdomain_progress[user_id] = {
            'current_step': 0,
            'total_steps': 25,
            'current_artist': '',
            'current_domain': '',
            'percentage': 0,
            'status': 'starting'
        }
        print(f"[PROGRESS] Using anonymous user due to error: {user_id}")

    # Get user's profile including country and location
    try:
        user_country = user_profile.get('country', 'US') if user_profile else 'US'
        location = get_location_from_country(user_country)
        print(f"User country: {user_country} -> Location: {location}")
        
        # Update progress - profile loaded
        if user_id in crossdomain_progress:
            crossdomain_progress[user_id].update({
                'current_step': 1,
                'percentage': 4,
                'status': 'profile_loaded'
            })
    except Exception as e:
        print(f"Could not fetch user profile: {e}")
        user_country = 'US'  # Default fallback
        location = "New York"

    # Determine which artists to use - prioritize user data from Spotify
    if top_scored_artists:
        print(f"[CROSS-DOMAIN UNIFIED] Using top scored artists: {top_scored_artists}")
        music_artists = top_scored_artists[:5]  # Use top 5 scored artists
        # Get detailed info for these artists
        try:
            user_artists_with_images = []
            for artist_name in music_artists:
                # Try to get artist info from Spotify
                try:
                    print(f"[DEBUG] Fetching genres for artist: {artist_name}")
                    # First validate token before making API calls
                    if not validate_token(spotify_token):
                        print(f"[WARNING] Spotify token is invalid for {artist_name}, using fallback genres")
                        # Use fallback genres based on artist name patterns
                        fallback_genres = get_fallback_genres_for_artist(artist_name)
                        user_artists_with_images.append({
                            'name': artist_name,
                            'genres': fallback_genres,
                            'image': None
                        })
                    else:
                        artist_genres = get_spotify_artist_genres(artist_name, spotify_token)
                        print(f"[DEBUG] Retrieved genres for {artist_name}: {artist_genres}")
                        if not artist_genres:
                            # If no genres returned, use fallback
                            fallback_genres = get_fallback_genres_for_artist(artist_name)
                            print(f"[DEBUG] Using fallback genres for {artist_name}: {fallback_genres}")
                            artist_genres = fallback_genres
                        user_artists_with_images.append({
                            'name': artist_name,
                            'genres': artist_genres,
                            'image': None
                        })
                except Exception as e:
                    print(f"[ERROR] Failed to get genres for {artist_name}: {e}")
                    # Use fallback genres on error
                    fallback_genres = get_fallback_genres_for_artist(artist_name)
                    user_artists_with_images.append({
                        'name': artist_name,
                        'genres': fallback_genres,
                        'image': None
                    })
        except Exception as e:
            print(f"Could not fetch artist details: {e}")
            user_artists_with_images = []
            for name in music_artists:
                fallback_genres = get_fallback_genres_for_artist(name)
                user_artists_with_images.append({'name': name, 'genres': fallback_genres, 'image': None})
    elif music_artists:
        print(f"[CROSS-DOMAIN UNIFIED] Using provided music artists: {music_artists}")
        # Get detailed info for provided artists
        try:
            user_artists_with_images = []
            for artist_name in music_artists[:6]:  # Limit to 6 artists
                try:
                    # First validate token before making API calls
                    if not validate_token(spotify_token):
                        print(f"[WARNING] Spotify token is invalid for {artist_name}, using fallback genres")
                        fallback_genres = get_fallback_genres_for_artist(artist_name)
                        user_artists_with_images.append({
                            'name': artist_name,
                            'genres': fallback_genres,
                            'image': None
                        })
                    else:
                        artist_genres = get_spotify_artist_genres(artist_name, spotify_token)
                        if not artist_genres:
                            # If no genres returned, use fallback
                            fallback_genres = get_fallback_genres_for_artist(artist_name)
                            print(f"[DEBUG] Using fallback genres for {artist_name}: {fallback_genres}")
                            artist_genres = fallback_genres
                        user_artists_with_images.append({
                            'name': artist_name,
                            'genres': artist_genres,
                            'image': None
                        })
                except Exception as e:
                    print(f"Could not fetch details for {artist_name}: {e}")
                    # Use fallback genres on error
                    fallback_genres = get_fallback_genres_for_artist(artist_name)
                    user_artists_with_images.append({
                        'name': artist_name,
                        'genres': fallback_genres,
                        'image': None
                    })
        except Exception as e:
            print(f"Could not fetch artist details: {e}")
            user_artists_with_images = []
            for name in music_artists[:6]:
                fallback_genres = get_fallback_genres_for_artist(name)
                user_artists_with_images.append({'name': name, 'genres': fallback_genres, 'image': None})
    else:
        # No artists provided, try to fetch from Spotify first
        print("[CROSS-DOMAIN UNIFIED] No artists provided, fetching from Spotify...")
        music_artists = []
        user_artists_with_images = []
        
        try:
            if spotify_token:
                print("[CROSS-DOMAIN UNIFIED] Attempting to fetch recent artists from Spotify...")
                # First try to get recently played artists
                recent_artists = get_spotify_recently_played(spotify_token, limit=20)
                unique_recent_artists = []
                seen = set()
                for artist in recent_artists:
                    if artist not in seen:
                        unique_recent_artists.append(artist)
                        seen.add(artist)
                    if len(unique_recent_artists) == 5:
                        break
                
                if unique_recent_artists:
                    print(f"[CROSS-DOMAIN UNIFIED] Found {len(unique_recent_artists)} recent artists: {unique_recent_artists}")
                    music_artists = unique_recent_artists
                else:
                    print("[CROSS-DOMAIN UNIFIED] No recent artists found, trying top artists...")
                    # Fallback to top artists if no recent data
                    top_artists = get_spotify_top_artists(spotify_token, limit=5)
                    if top_artists:
                        print(f"[CROSS-DOMAIN UNIFIED] Found {len(top_artists)} top artists: {top_artists}")
                        music_artists = top_artists
                    else:
                        print("[CROSS-DOMAIN UNIFIED] No top artists found either")
                
                # Get detailed artist info with images and genres
                if music_artists:
                    user_artists_with_images = get_spotify_top_artists_with_images(spotify_token, limit=6)
                    if user_artists_with_images:
                        music_artists = [artist['name'] for artist in user_artists_with_images]
                        print(f"[CROSS-DOMAIN UNIFIED] Successfully fetched {len(music_artists)} artists from Spotify")
                    else:
                        print("[CROSS-DOMAIN UNIFIED] Could not get detailed artist info")
                else:
                    print("[CROSS-DOMAIN UNIFIED] No artists found from Spotify")
            else:
                print("[CROSS-DOMAIN UNIFIED] No Spotify token provided")
        except Exception as e:
            print(f"[CROSS-DOMAIN UNIFIED] Error fetching from Spotify: {e}")
        
        # If still no artists, use fallback
        if not music_artists:
            print("[CROSS-DOMAIN UNIFIED] No artists from Spotify, using fallback artists")
            fallback_artists = ["Arijit Singh", "Amit Trivedi", "Pritam", "Shreya Ghoshal", "Neha Kakkar"]
            music_artists = fallback_artists
            user_artists_with_images = []
            for artist_name in music_artists:
                user_artists_with_images.append({
                    'name': artist_name,
                    'genres': ['bollywood', 'indian pop', 'film music'],  # Default genres
                    'image': None
                })
            print(f"[CROSS-DOMAIN UNIFIED] Using fallback artists: {music_artists}")

    # Get comprehensive user preferences
    try:
        if spotify_token:
            user_preferences = get_user_listening_preferences(spotify_token)
            print(f"[PREFERENCES] User genres: {user_preferences.get('genres', [])}")
            print(f"[PREFERENCES] Mood indicators: {user_preferences.get('mood_indicators', [])}")
        else:
            print("[PREFERENCES] No Spotify token, using default preferences")
            user_preferences = {'genres': ['bollywood', 'indian pop', 'film music'], 'mood_indicators': ['romantic', 'melodious']}
    except Exception as e:
        print(f"Could not fetch user preferences: {e}")
        user_preferences = {'genres': ['bollywood', 'indian pop', 'film music'], 'mood_indicators': ['romantic', 'melodious']}
    
    # Update progress - preferences loaded
    if user_id in crossdomain_progress:
        crossdomain_progress[user_id].update({
            'current_step': 2,
            'percentage': 8,
            'status': 'preferences_loaded'
        })

    # Get artist genres for each artist
    artist_genres_map = {}
    for artist in user_artists_with_images:
        artist_name = artist['name']
        artist_genres = artist.get('genres', [])
        artist_genres_map[artist_name] = artist_genres
        print(f"[ARTIST] {artist_name}: genres = {artist_genres}")

    # Entity types for cross-domain recommendations
    ENTITY_TYPES = [
        ("urn:entity:artist", "music artist"),
        ("urn:entity:book", "book"),
        ("urn:entity:movie", "movie"),
        ("urn:entity:podcast", "podcast"),
        ("urn:entity:tv_show", "TV show")
    ]

    # Initialize Qloo client
    qloo_client = QlooAPIClient(qloo_api_key)
    all_recommendations = {}

    # Process each artist with rate limiting
    for i, artist_name in enumerate(music_artists[:5]):  # Only use last 5 unique artists
        print(f"Processing recommendations for: {artist_name}")
        
        # Update progress for current artist
        if user_id in crossdomain_progress:
            current_step = 4 + (i * 5)  # Start at step 4, each artist gets 5 steps
            percentage = min(100, int((current_step / crossdomain_progress[user_id]['total_steps']) * 100))
            crossdomain_progress[user_id].update({
                'current_step': current_step,
                'current_artist': artist_name,
                'current_domain': '',
                'percentage': percentage,
                'status': f'processing_artist_{i+1}'
            })
            print(f"[PROGRESS] Processing artist {i+1}/5: {artist_name} - {percentage}%")
        
        # Add delay between artists to prevent rate limiting
        if i > 0:
            print(f"[RATE LIMIT] Waiting 2 seconds between artists to prevent rate limiting")
            time.sleep(2)
        
        # Search for artist in Qloo
        try:
            entity = qloo_client.search_entity(artist_name)
            if not entity:
                print(f"Could not find Qloo entity for '{artist_name}'. Skipping.")
                continue
                
            entity_id = entity.get('entity_id', entity.get('id'))
            if not entity_id:
                print(f"No entity ID found for '{artist_name}'. Skipping.")
                continue
                
            artist_ids = [entity_id]
            artist_results = {}
        except Exception as e:
            print(f"Error searching for artist '{artist_name}' in Qloo: {e}")
            continue

        # Get recommendations for each domain with rate limiting
        for domain_i, (etype, domain) in enumerate(ENTITY_TYPES):
            print(f"Getting {domain} recommendations for {artist_name}")
            
            # Update progress for current domain
            if user_id in crossdomain_progress:
                domain_step = current_step + domain_i + 1
                domain_percentage = min(100, int((domain_step / crossdomain_progress[user_id]['total_steps']) * 100))
                crossdomain_progress[user_id].update({
                    'current_step': domain_step,
                    'current_domain': domain,
                    'percentage': domain_percentage,
                    'status': f'processing_{domain.lower().replace(" ", "_")}'
                })
                print(f"[PROGRESS] Processing {domain} for {artist_name} - {domain_percentage}%")
            
            # Add delay between domains to prevent Gemini API rate limiting
            if domain_i > 0:
                print(f"[GEMINI RATE LIMIT] Waiting 1 second between domains to prevent rate limiting")
                time.sleep(1)
            
            try:
                # Get dynamic tags for this domain with user country, location, and artist context
                artist_genres = artist_genres_map.get(artist_name, [])
                
                # Combine user tags with dynamic tags for better recommendations (if user_tags provided)
                combined_tags = []
                
                # Add user-provided tags if available
                if user_tags:
                    # Convert user tags to Qloo tag format
                    for tag in user_tags:
                        combined_tags.append({
                            'name': tag,
                            'id': f"user_tag_{tag}",
                            'source': 'user'
                        })
                
                # Try enhanced dynamic tag generation first with user context
                print(f"[DEBUG] Getting dynamic tags for {domain} with artist {artist_name}")
                dynamic_tags = get_dynamic_tags_for_domain_enhanced(
                    domain, qloo_api_key, user_country, location, limit=25, 
                    artist_name=artist_name, artist_genres=artist_genres, gemini_api_key=gemini_api_key
                )
                
                print(f"[DEBUG] Dynamic tags for {domain}: {len(dynamic_tags) if dynamic_tags else 0} tags")
                
                # Add dynamic tags to combined list
                if dynamic_tags:
                    for tag in dynamic_tags:
                        tag['source'] = 'dynamic'
                        combined_tags.append(tag)
                
                # If no dynamic tags, try genre-based domain tags as fallback
                if not dynamic_tags and artist_genres:
                    print(f"[FALLBACK] Trying genre-based domain tags for {domain}")
                    tag_ids, dynamic_tags = get_domain_appropriate_tags(
                        artist_genres, domain, qloo_api_key, gemini_api_key, user_country, location
                    )
                    if dynamic_tags:
                        # Convert to the expected format
                        for tag, tag_id in zip(dynamic_tags, tag_ids):
                            if tag_id:
                                combined_tags.append({
                                    'name': tag,
                                    'id': tag_id,
                                    'source': 'fallback'
                                })
                
                # If still no tags, add basic fallback tags for each domain
                if not combined_tags:
                    print(f"[EMERGENCY FALLBACK] Adding basic tags for {domain}")
                    basic_tags = {
                        "movie": ["action", "drama", "comedy", "thriller", "romance", "sci-fi", "horror", "adventure", "mystery", "documentary"],
                        "book": ["fiction", "romance", "mystery", "biography", "self-help", "fantasy", "sci-fi", "thriller", "historical", "young-adult"],
                        "podcast": ["comedy", "news", "true-crime", "business", "health", "education", "technology", "sports", "politics", "entertainment"],
                        "TV show": ["drama", "comedy", "reality", "documentary", "action", "thriller", "sci-fi", "horror", "adventure", "mystery"],
                        "music artist": ["pop", "rock", "hip-hop", "jazz", "classical", "electronic", "country", "r&b", "indie", "alternative"]
                    }
                    
                    domain_basic_tags = basic_tags.get(domain, [])
                    for tag_name in domain_basic_tags:
                        try:
                            # Try to get the tag ID from Qloo
                            tag_id = get_qloo_tag_id(tag_name, qloo_api_key)
                            if tag_id:
                                combined_tags.append({
                                    'name': tag_name,
                                    'id': tag_id,
                                    'source': 'emergency_fallback'
                                })
                        except Exception as e:
                            print(f"Could not get tag ID for {tag_name}: {e}")
                            continue
            
                if not combined_tags:
                    print(f"No tags found for {domain}. Skipping.")
                    continue
                
                tag_names = [tag['name'] for tag in combined_tags]
                tag_ids = {tag['name']: tag['id'] for tag in combined_tags}
                
                # Ask Gemini to pick the best tag with domain-specific context
                context = f"Artist: {artist_name}, Domain: {domain}, Available tags: {', '.join(tag_names[:10])}"
                gemini_tag = call_gemini_for_tag_enhanced(context, tag_names, gemini_api_key, user_country)
                
                # Find the best working tag using robust fallback system
                print(f"[DEBUG] Finding working tag for {domain} with {len(tag_names)} available tags")
                selected_tag, selected_tag_id = find_working_tag(
                    gemini_tag, tag_names, tag_ids, qloo_client, etype, artist_ids
                )
                
                if not selected_tag:
                    print(f"No working tags found for {domain}")
                    continue
                
                print(f"[DEBUG] Selected tag for {domain}: {selected_tag} (ID: {selected_tag_id})")
                
                # Get final recommendations - request more to have options for top selection
                print(f"[DEBUG] Final Qloo API call for {domain}: etype={etype}, tag_id={selected_tag_id}, take={recommendations_per_domain * 3}")
                rec = qloo_client.get_recommendations(etype, selected_tag_id, artist_ids, take=recommendations_per_domain * 3)
                entities = rec.get("results", {}).get("entities", []) if rec else []
                print(f"[DEBUG] Qloo response for {domain}: {len(entities)} entities found")
                
                # If no entities found, try alternative tags
                if not entities:
                    print(f"[FALLBACK] No entities found for {domain} with tag '{selected_tag}', trying alternative tags...")
                    alternative_tags = {
                        "movie": ["drama", "comedy", "action"],
                        "book": ["fiction", "romance", "mystery"],
                        "podcast": ["comedy", "news", "education"],
                        "TV show": ["drama", "comedy", "reality"],
                        "music artist": ["pop", "rock", "hip-hop"]
                    }
                    
                    domain_alt_tags = alternative_tags.get(domain, [])
                    for alt_tag in domain_alt_tags:
                        try:
                            alt_tag_id = get_qloo_tag_id(alt_tag, qloo_api_key)
                            if alt_tag_id and alt_tag_id != selected_tag_id:
                                print(f"[FALLBACK] Trying alternative tag '{alt_tag}' for {domain}")
                                alt_rec = qloo_client.get_recommendations(etype, alt_tag_id, artist_ids, take=recommendations_per_domain * 3)
                                alt_entities = alt_rec.get("results", {}).get("entities", []) if alt_rec else []
                                if alt_entities:
                                    entities = alt_entities
                                    selected_tag = alt_tag
                                    selected_tag_id = alt_tag_id
                                    print(f"[FALLBACK] Found {len(entities)} entities with alternative tag '{alt_tag}'")
                                    break
                        except Exception as e:
                            print(f"[FALLBACK] Could not try alternative tag {alt_tag}: {e}")
                            continue
                
                # Clean up and format entities with rich metadata
                cleaned_entities = []
                for ent in entities:
                    if 'akas' in ent:
                        del ent['akas']
                    
                    # Extract rich metadata
                    properties = ent.get('properties', {})
                    image_url = None
                    if properties and 'image' in properties:
                        image_url = properties.get('image', {}).get('url')
                    
                    # Calculate affinity score based on popularity and cultural relevance
                    popularity = ent.get('popularity', 0)
                    cultural_relevance = ent.get('cultural_relevance', 0)
                    affinity_score = (popularity * 0.6) + (cultural_relevance * 0.4)
                    
                    # Enhanced properties based on domain
                    enhanced_properties = {
                        'image': {'url': image_url} if image_url else None,
                        'description': properties.get('description'),
                        'url': properties.get('url'),
                        'external_urls': properties.get('external_urls', {})
                    }
                    
                    # Domain-specific properties
                    if domain == "movie":
                        enhanced_properties.update({
                            'year': properties.get('year'),
                            'genre': properties.get('genre'),
                            'director': properties.get('director'),
                            'runtime': properties.get('runtime'),
                            'rating': properties.get('rating')
                        })
                    elif domain == "book":
                        enhanced_properties.update({
                            'author': properties.get('author'),
                            'publisher': properties.get('publisher'),
                            'year': properties.get('year'),
                            'genre': properties.get('genre'),
                            'language': properties.get('language')
                        })
                    elif domain == "podcast":
                        enhanced_properties.update({
                            'host': properties.get('host'),
                            'episodes': properties.get('episodes'),
                            'genre': properties.get('genre'),
                            'language': properties.get('language')
                        })
                    elif domain == "TV show":
                        enhanced_properties.update({
                            'year': properties.get('year'),
                            'genre': properties.get('genre'),
                            'seasons': properties.get('seasons'),
                            'episodes': properties.get('episodes'),
                            'rating': properties.get('rating')
                        })
                    elif domain == "music artist":
                        enhanced_properties.update({
                            'followers': properties.get('followers'),
                            'albums': properties.get('albums'),
                            'genres': properties.get('genres', []),
                            'spotify_url': properties.get('spotify_url')
                        })
                    
                    cleaned_entities.append({
                        'name': ent.get('name', 'Unknown'),
                        'id': ent.get('id'),
                        'popularity': popularity,
                        'affinity_score': round(affinity_score, 2),
                        'cultural_relevance': cultural_relevance,
                        'tags': ent.get('tags', []),
                        'properties': enhanced_properties,
                        'selected_tag': selected_tag,
                        'source_artist': artist_name
                    })
                
                artist_results[domain] = cleaned_entities
            
            except Exception as e:
                print(f"Error processing {domain} recommendations for {artist_name}: {e}")
                continue
        
        all_recommendations[artist_name] = artist_results

    # Update progress - starting aggregation
    if user_id in crossdomain_progress:
        crossdomain_progress[user_id].update({
            'current_step': 29,
            'current_domain': '',
            'percentage': 90,
            'status': 'aggregating_results'
        })
        print(f"[PROGRESS] Aggregating results - 90%")
    
    # Aggregate results by domain across all artists with deduplication
    domain_aggregated = {}
    for domain_info in ENTITY_TYPES:
        domain = domain_info[1]
        domain_aggregated[domain] = []
        seen_entities = set()  # Track unique entities by name
        
        for artist_name, artist_results in all_recommendations.items():
            if domain in artist_results:
                for entity in artist_results[domain]:
                    entity_name = entity.get('name', '').strip().lower()
                    
                    # Skip if we've already seen this entity
                    if entity_name in seen_entities:
                        continue
                    
                    seen_entities.add(entity_name)
                    entity['source_artist'] = artist_name
                    domain_aggregated[domain].append(entity)
    
    # Sort by relevance instead of popularity for all domains
    user_genres = []
    for artist_name in music_artists:
        artist_genres = artist_genres_map.get(artist_name, [])
        user_genres.extend(artist_genres)
    user_genres = list(set(user_genres))  # Remove duplicates
    
    # Update progress - sorting results
    if user_id in crossdomain_progress:
        crossdomain_progress[user_id].update({
            'current_step': 30,
            'current_domain': '',
            'percentage': 95,
            'status': 'sorting_results'
        })
        print(f"[PROGRESS] Sorting results - 95%")
    
    # Apply sorting and limiting to all domains
    for domain in domain_aggregated:
        domain_aggregated[domain] = sort_by_relevance(
            domain_aggregated[domain], 
            music_artists, 
            user_genres, 
            "upbeat",  # Default context for cross-domain
            user_country, 
            location,
            user_preferences
        )
        # Return exactly top N recommendations per domain
        domain_aggregated[domain] = domain_aggregated[domain][:recommendations_per_domain]

    # Update progress - finalizing
    if user_id in crossdomain_progress:
        crossdomain_progress[user_id].update({
            'current_step': 31,
            'current_domain': '',
            'percentage': 98,
            'status': 'finalizing'
        })
        print(f"[PROGRESS] Finalizing recommendations - 98%")
    
    # Prepare response data
    response_data = {
        "top_artists": music_artists,
        "top_artists_with_images": user_artists_with_images,
        "top_scored_artists": top_scored_artists,  # Include the original top scored artists
        "recommendations_by_domain": domain_aggregated,
        "detailed_results": all_recommendations,
        "total_domains": len([d for d in domain_aggregated.values() if d]),
        "recommendations_per_domain": recommendations_per_domain,
        "from_cache": False,
        "generated_timestamp": datetime.now().isoformat(),
        "location_used": location,
        "user_country": user_country,
        "user_context": user_context,  # Include user context in response
        "user_tags": user_tags  # Include user tags in response
    }
    
    return response_data


if __name__ == '__main__':
    app.run(debug=True, port=5500)
