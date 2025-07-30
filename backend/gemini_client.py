import requests
import json
import re
import time
import random
import hashlib
from functools import wraps
from utils import retry_on_failure
import redis
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API rate limiting configuration
GEMINI_RATE_LIMIT_DELAY = float(os.getenv('GEMINI_RATE_LIMIT_DELAY', '1.0'))  # Base delay between calls in seconds
GEMINI_MAX_RETRIES = int(os.getenv('GEMINI_MAX_RETRIES', '3'))
GEMINI_BACKOFF_FACTOR = float(os.getenv('GEMINI_BACKOFF_FACTOR', '2.0'))
GEMINI_CACHE_DURATION = int(os.getenv('GEMINI_CACHE_DURATION', '3600'))  # 1 hour cache

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', "redis://default:iArh1BsXVMS8qcvz5dxx6DI5Le0H4svu@redis-12272.c85.us-east-1-2.ec2.redns.redis-cloud.com:12272")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Global cache for Gemini API responses (fallback if Redis fails)
gemini_cache = {}
gemini_last_call_time = 0

def gemini_rate_limit():
    """Rate limiting decorator for Gemini API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global gemini_last_call_time
            
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

@retry_on_gemini_rate_limit()
@gemini_rate_limit()
def detect_language_preference(user_context, gemini_api_key):
    """
    Detect language preference from user context using Gemini
    Returns: {'primary_language': 'english', 'secondary_languages': ['hindi'], 'confidence': 0.9}
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print(f"[LANGUAGE DETECTION] No valid Gemini API key provided, using fallback detection")
        # Fallback to keyword-based detection
        context_lower = user_context.lower()
        if 'english' in context_lower or 'only english' in context_lower:
            return {
                'primary_language': 'english',
                'secondary_languages': [],
                'confidence': 0.8,
                'language_keywords': ['english'],
                'explicit_language_request': True
            }
        elif 'hindi' in context_lower or 'bollywood' in context_lower:
            return {
                'primary_language': 'hindi',
                'secondary_languages': [],
                'confidence': 0.8,
                'language_keywords': ['hindi'],
                'explicit_language_request': True
            }
        else:
            return {
                'primary_language': 'any',
                'secondary_languages': [],
                'confidence': 0.5,
                'language_keywords': [],
                'explicit_language_request': False
            }
    
    # Check cache first
    cache_key = get_gemini_cache_key(user_context, gemini_api_key)
    cached_response = get_cached_gemini_response(cache_key)
    if cached_response:
        return cached_response
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
Analyze this user context and detect language preferences: '{user_context}'

Detect:
1. **Primary Language**: What is the main language the user wants music in?
2. **Secondary Languages**: What other languages are acceptable?
3. **Confidence**: How confident are you in this detection (0.0-1.0)?
4. **Keywords**: What language-related keywords did you find?
5. **Explicit Request**: Is there an explicit language request?

Common language indicators:
- English: "english", "only english", "no hindi", "western", "international"
- Hindi: "hindi", "bollywood", "indian", "desi", "hindi songs"
- Spanish: "spanish", "latino", "hispanic", "español"
- French: "french", "français", "france"
- Korean: "korean", "k-pop", "korea"
- Japanese: "japanese", "j-pop", "japan"

Return ONLY a JSON object with this exact format:
{{
    "primary_language": "english",
    "secondary_languages": ["hindi", "spanish"],
    "confidence": 0.9,
    "language_keywords": ["english", "motivational"],
    "explicit_language_request": true
}}
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 500
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            raise Exception("Invalid response structure from Gemini")
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            raise Exception("Empty response from Gemini")
        
        print(f"[LANGUAGE DETECTION] Raw response: {result_text[:200]}...")
        
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
            language_pref = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"[LANGUAGE DETECTION] JSON parsing failed: {e}")
            print(f"[LANGUAGE DETECTION] Attempting to extract JSON from response...")
            
            # Try to extract JSON using regex
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                try:
                    language_pref = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise Exception("Could not parse JSON from response")
            else:
                raise Exception("No JSON found in response")
        
        print(f"[LANGUAGE DETECTION] Detected: {language_pref['primary_language']} (confidence: {language_pref['confidence']})")
        
        # Cache the response
        cache_gemini_response(cache_key, language_pref)
        
        return language_pref
        
    except Exception as e:
        print(f"Error detecting language preference: {e}")
        # Fallback to keyword-based detection
        context_lower = user_context.lower()
        if 'english' in context_lower or 'only english' in context_lower:
            return {
                'primary_language': 'english',
                'secondary_languages': [],
                'confidence': 0.8,
                'language_keywords': ['english'],
                'explicit_language_request': True
            }
        elif 'hindi' in context_lower or 'bollywood' in context_lower:
            return {
                'primary_language': 'hindi',
                'secondary_languages': [],
                'confidence': 0.8,
                'language_keywords': ['hindi'],
                'explicit_language_request': True
            }
        else:
            return {
                'primary_language': 'any',
                'secondary_languages': [],
                'confidence': 0.5,
                'language_keywords': [],
                'explicit_language_request': False
            }

@retry_on_gemini_rate_limit()
@gemini_rate_limit()
def detect_specific_mood_activity(user_context, gemini_api_key):
    """
    Detect specific mood/activity from user context using Gemini
    Returns: {'primary_mood': 'dance', 'secondary_moods': ['party', 'energetic'], 'confidence': 0.9, 'activity_type': 'dancing'}
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
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
    
    # Check cache first
    cache_key = get_gemini_cache_key(user_context, gemini_api_key)
    cached_response = get_cached_gemini_response(cache_key)
    if cached_response:
        return cached_response
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
    "mood_keywords": ["dance", "party"]
}}
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 500
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            raise Exception("Invalid response structure from Gemini")
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            raise Exception("Empty response from Gemini")
        
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
        
        # Cache the response
        cache_gemini_response(cache_key, mood_pref)
        
        return mood_pref
        
    except Exception as e:
        print(f"Error detecting mood preference: {e}")
        # Fallback to keyword-based detection
        context_lower = user_context.lower()
        if any(word in context_lower for word in ['dance', 'party', 'club']):
            return {
                'primary_mood': 'dance',
                'secondary_moods': ['party', 'energetic'],
                'activity_type': 'dancing',
                'energy_level': 'high',
                'confidence': 0.8,
                'mood_keywords': ['dance', 'party']
            }
        elif any(word in context_lower for word in ['study', 'work', 'focus']):
            return {
                'primary_mood': 'study',
                'secondary_moods': ['focused', 'motivated'],
                'activity_type': 'studying',
                'energy_level': 'medium',
                'confidence': 0.8,
                'mood_keywords': ['study', 'focus']
            }
        else:
            return {
                'primary_mood': 'general',
                'secondary_moods': [],
                'activity_type': 'general',
                'energy_level': 'medium',
                'confidence': 0.5,
                'mood_keywords': []
            }

def enhance_context_detection_with_mood_and_language(user_context, gemini_api_key):
    """
    Enhanced context detection that includes mood and language preferences
    Returns: {'context_type': 'study', 'language_preference': {...}, 'mood_preference': {...}}
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print(f"[ENHANCED CONTEXT] No valid Gemini API key provided, using fallback detection")
        # Return fallback enhanced context
        return {
            'context_type': 'general',
            'language_preference': {
                'primary_language': 'any',
                'secondary_languages': [],
                'confidence': 0.5,
                'language_keywords': [],
                'explicit_language_request': False
            },
            'mood_preference': {
                'primary_mood': 'general',
                'secondary_moods': [],
                'activity_type': 'general',
                'energy_level': 'medium',
                'confidence': 0.5,
                'mood_keywords': [],
                'explicit_mood_request': False
            }
        }
    
    try:
        # Detect language preference
        language_preference = detect_language_preference(user_context, gemini_api_key)
        
        # Detect mood/activity preference
        mood_preference = detect_specific_mood_activity(user_context, gemini_api_key)
        
        # Determine context type from mood preference
        context_type = mood_preference.get('activity_type', 'general')
        if context_type == 'dancing':
            context_type = 'dance'
        elif context_type == 'studying':
            context_type = 'study'
        elif context_type == 'working out':
            context_type = 'workout'
        
        enhanced_context = {
            'context_type': context_type,
            'language_preference': language_preference,
            'mood_preference': mood_preference
        }
        
        print(f"[ENHANCED CONTEXT] Context: {context_type}, Mood: {mood_preference['primary_mood']}, Language: {language_preference['primary_language']}")
        print(f"[ENHANCED CONTEXT] Activity: {mood_preference['activity_type']}, Energy: {mood_preference['energy_level']}")
        
        return enhanced_context
        
    except Exception as e:
        print(f"Error in enhanced context detection: {e}")
        # Fallback to basic context detection
        return {
            'context_type': 'general',
            'language_preference': {'primary_language': 'any', 'confidence': 0.5},
            'mood_preference': {'primary_mood': 'general', 'confidence': 0.5}
        }

def call_gemini_for_enhanced_tags(user_context, gemini_api_key, user_country=None, location=None):
    """
    Generate enhanced tags using Gemini with cultural context
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
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
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    # Build context-aware prompt
    context_info = f"User Context: {user_context}"
    if user_country:
        context_info += f"\nUser Country: {user_country}"
    if location:
        context_info += f"\nLocation: {location}"
    
    prompt = f"""
Generate focused music tags for this request: {context_info}

Requirements:
- Generate 15-20 relevant tags
- Focus on genres, moods, styles, and cultural elements
- Consider the user's location and cultural context
- Include both general and specific tags
- Tags should be comma-separated, no quotes

Examples of good tags:
- upbeat, energetic, pop, rock, electronic, motivational, focused, inspiring
- bollywood, hindi pop, indian classical, desi, mumbai, study music
- dance, party, workout, relaxing, romantic, driving, cooking

Return ONLY a comma-separated list of tags, no additional text.
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 300
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            raise Exception("Invalid response structure from Gemini")
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            raise Exception("Empty response from Gemini")
        
        # Parse tags from response
        tags_text = result_text.strip()
        # Remove any markdown formatting
        if tags_text.startswith('```'):
            tags_text = tags_text.split('\n', 1)[1] if '\n' in tags_text else tags_text[3:]
        if tags_text.endswith('```'):
            tags_text = tags_text[:-3]
        
        # Split by comma and clean
        tags = [tag.strip().lower() for tag in tags_text.split(',') if tag.strip()]
        
        print(f"[ENHANCED GEMINI] Generated {len(tags)} focused tags: {tags}")
        return tags
        
    except Exception as e:
        print(f"Error generating enhanced tags: {e}")
        # Fallback to basic tags
        return ["upbeat", "energetic", "pop", "motivational"]

def generate_cultural_context(user_context, user_country, location, gemini_api_key):
    """
    Generate cultural context for recommendations
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print(f"[CULTURAL CONTEXT] No valid Gemini API key provided, using fallback cultural context")
        return _get_fallback_cultural_context(location, user_country)
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
Generate cultural context keywords for music recommendations.

User Context: {user_context}
User Country: {user_country}
Location: {location}

Generate 5-10 cultural keywords that would help with music recommendations.
Focus on:
- Cultural influences
- Local music preferences
- Regional styles
- Cultural keywords for Qloo API

Return ONLY a comma-separated list of keywords, no additional text.
Keep it short and focused (max 200 characters total).
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 200
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            raise Exception("Invalid response structure from Gemini")
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            raise Exception("Empty response from Gemini")
        
        # Parse keywords from response
        keywords_text = result_text.strip()
        # Remove any markdown formatting
        if keywords_text.startswith('```'):
            keywords_text = keywords_text.split('\n', 1)[1] if '\n' in keywords_text else keywords_text[3:]
        if keywords_text.endswith('```'):
            keywords_text = keywords_text[:-3]
        
        # Split by comma and clean
        keywords = [keyword.strip() for keyword in keywords_text.split(',') if keyword.strip()]
        cultural_context = ', '.join(keywords)
        
        print(f"[CULTURAL CONTEXT] Generated: {cultural_context}")
        return cultural_context
        
    except Exception as e:
        print(f"Error generating cultural context: {e}")
        # Fallback cultural context
        return _get_fallback_cultural_context(location, user_country)

def _get_fallback_cultural_context(location, user_country):
    """Fallback cultural context when Gemini fails"""
    if location and 'mumbai' in location.lower():
        return "Mumbai, Bollywood influence, Indian pop, youth culture"
    elif user_country and user_country.lower() == 'in':
        return "India, Bollywood, Indian pop, cultural fusion"
    else:
        return "International, pop, contemporary, global"

def detect_context_type_llm(user_context, gemini_api_key):
    """
    Detect context type using Gemini
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print(f"[CONTEXT] No valid Gemini API key provided, using fallback context detection")
        # Fallback to keyword-based detection
        context_lower = user_context.lower()
        if any(word in context_lower for word in ['dance', 'party', 'club']):
            return 'dance'
        elif any(word in context_lower for word in ['workout', 'gym', 'exercise']):
            return 'workout'
        elif any(word in context_lower for word in ['study', 'work', 'focus']):
            return 'study'
        else:
            return 'general'
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
Analyze this user context and determine the music context type: '{user_context}'

Context types:
- dance: dancing, party, club, nightclub, disco
- workout: gym, exercise, running, fitness, training
- study: study, work, focus, concentration, office
- relaxing: sleep, bedtime, relax, calm, soothing
- driving: driving, road trip, commute, travel, car
- cooking: cooking, kitchen, food, recipe, chef
- romantic: romance, love, date, romantic, intimate
- sad: sad, breakup, heartbreak, melancholic
- happy: happy, joyful, cheerful, upbeat, positive
- general: anything else

Return ONLY the context type word, no additional text.
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 50
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            raise Exception("Invalid response structure from Gemini")
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            raise Exception("Empty response from Gemini")
        
        context_type = result_text.strip().lower()
        print(f"[CONTEXT] Detected context type: {context_type}")
        return context_type
        
    except Exception as e:
        print(f"Error detecting context type: {e}")
        # Fallback to keyword-based detection
        context_lower = user_context.lower()
        if any(word in context_lower for word in ['dance', 'party', 'club']):
            return 'dance'
        elif any(word in context_lower for word in ['workout', 'gym', 'exercise']):
            return 'workout'
        elif any(word in context_lower for word in ['study', 'work', 'focus']):
            return 'study'
        else:
            return 'general'

def validate_context_detection(user_context, detected_context, gemini_api_key):
    """
    Validate the detected context using Gemini
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print(f"[CONTEXT VALIDATION] No valid Gemini API key provided, using detected context as-is")
        return detected_context
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
Validate if this context detection is correct:

User Context: '{user_context}'
Detected Context: '{detected_context}'

Context types:
- dance: dancing, party, club, nightclub, disco
- workout: gym, exercise, running, fitness, training
- study: study, work, focus, concentration, office
- relaxing: sleep, bedtime, relax, calm, soothing
- driving: driving, road trip, commute, travel, car
- cooking: cooking, kitchen, food, recipe, chef
- romantic: romance, love, date, romantic, intimate
- sad: sad, breakup, heartbreak, melancholic
- happy: happy, joyful, cheerful, upbeat, positive
- general: anything else

Is the detected context correct? If yes, return the same context. If no, return the correct context.

Return ONLY the context type word, no additional text.
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 50
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            raise Exception("Invalid response structure from Gemini")
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            raise Exception("Empty response from Gemini")
        
        validated_context = result_text.strip().lower()
        print(f"[CONTEXT VALIDATION] ✓ Context '{detected_context}' is correct for '{user_context}'")
        return validated_context
        
    except Exception as e:
        print(f"Error validating context: {e}")
        return detected_context

def ai_calculate_relevance_score(entity, user_context, user_artists, user_genres, context_type, user_country, location, user_preferences=None, gemini_api_key=None):
    """
    Calculate AI-powered relevance score using Gemini
    """
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        return 5.0  # Default score if no valid API key
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    entity_name = entity.get('name', 'Unknown')
    entity_artist = entity.get('artist', 'Unknown')
    
    prompt = f"""
Calculate a relevance score (1-10) for this music track in the context of the user's request.

Track: "{entity_name}" by {entity_artist}
User Context: "{user_context}"
Context Type: {context_type}
User Country: {user_country}
Location: {location}

Consider:
- How well the track matches the user's request
- Cultural relevance to the user's location
- Genre and style appropriateness
- Energy level and mood fit

Return ONLY a number between 1-10, no additional text.
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 10
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            return 5.0
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            return 5.0
        
        # Extract score from response
        score_text = result_text.strip()
        try:
            score = float(score_text)
            return max(1.0, min(10.0, score))  # Clamp between 1-10
        except ValueError:
            return 5.0
        
    except Exception as e:
        print(f"Error calculating AI relevance score: {e}")
        return 5.0

def ai_sort_by_relevance(entities, user_context, user_artists, user_genres, context_type, user_country, location, user_preferences=None, gemini_api_key=None):
    """
    Sort entities by AI-powered relevance using Gemini
    """
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        return entities  # Return unsorted if no valid API key
    
    print(f"[AI SORTING] Sorting {len(entities)} entities with AI-powered relevance scoring")
    
    # Calculate AI scores for each entity
    scored_entities = []
    for entity in entities:
        ai_score = ai_calculate_relevance_score(
            entity, user_context, user_artists, user_genres, 
            context_type, user_country, location, user_preferences, gemini_api_key
        )
        
        # Combine AI score with traditional score
        traditional_score = entity.get('relevance_score', 5.0)
        final_score = (ai_score + traditional_score) / 2
        
        entity['ai_score'] = ai_score
        entity['final_score'] = final_score
        
        scored_entities.append(entity)
        
        print(f"[AI SCORING] {entity.get('name', 'Unknown')} by {entity.get('artist', 'Unknown')} - AI: {ai_score:.1f}, Traditional: {traditional_score:.1f}, Final: {final_score:.1f}")
    
    # Sort by final score
    sorted_entities = sorted(scored_entities, key=lambda x: x.get('final_score', 0), reverse=True)
    
    # Show top scores
    top_scores = [(entity.get('name', 'Unknown'), entity.get('final_score', 0)) for entity in sorted_entities[:5]]
    print(f"[AI SORTING] Top 5 scores: {top_scores}")
    
    return sorted_entities

def detect_artist_language_with_gemini(artist_name, genres, gemini_api_key):
    """
    Detect artist language using Gemini AI for more accurate detection
    Returns: language string ('english', 'hindi', 'spanish', 'french', 'korean', 'japanese', etc.)
    """
    # Check if we have a valid API key
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print(f"[GEMINI LANGUAGE] No valid Gemini API key provided, using fallback detection")
        return detect_artist_language_fallback(artist_name, genres)
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': gemini_api_key
    }
    
    prompt = f"""
Analyze this artist and determine their primary language based on name and genres:

Artist Name: "{artist_name}"
Genres: {genres}

Detect the primary language this artist sings/performs in.

Language indicators:
- English: English names, western genres (pop, rock, hip-hop, country, electronic)
- Hindi: Indian names, Bollywood, Indian pop, Punjabi, Bhangra, Desi
- Spanish: Spanish/Latin names, Reggaeton, Latin pop, Mexican, Colombian
- French: French names, French pop, Chanson, Francophone
- Korean: Korean names, K-pop, Korean pop, Korean names
- Japanese: Japanese names, J-pop, Japanese pop, Japanese names
- Arabic: Arabic names, Arabic pop, Middle Eastern
- Chinese: Chinese names, C-pop, Chinese pop
- Portuguese: Portuguese/Brazilian names, Brazilian pop, Portuguese pop

Consider:
1. Artist name origin and meaning
2. Genre associations
3. Cultural context
4. Common language patterns

Return ONLY the language name (lowercase), no additional text.
Examples: "english", "hindi", "spanish", "french", "korean", "japanese"
"""
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 50
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        if not result.get('candidates') or not result['candidates'][0].get('content'):
            raise Exception("Invalid response structure from Gemini")
        
        result_text = result['candidates'][0]['content']['parts'][0]['text']
        if not result_text.strip():
            raise Exception("Empty response from Gemini")
        
        detected_language = result_text.strip().lower()
        print(f"[GEMINI LANGUAGE] {artist_name} detected as: {detected_language}")
        return detected_language
        
    except Exception as e:
        print(f"Error detecting artist language with Gemini: {e}")
        # Fallback to hard-coded detection
        return detect_artist_language_fallback(artist_name, genres)

def detect_artist_language_fallback(artist_name, genres):
    """
    Fallback hard-coded language detection when Gemini fails
    """
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
    
    # French indicators
    french_indicators = ['french', 'français', 'chanson', 'francophone']
    if any(indicator in artist_name_lower for indicator in french_indicators) or \
       any(indicator in genres_lower for indicator in french_indicators):
        return "french"
    
    # Korean indicators
    korean_indicators = ['korean', 'k-pop', 'korea']
    if any(indicator in artist_name_lower for indicator in korean_indicators) or \
       any(indicator in genres_lower for indicator in korean_indicators):
        return "korean"
    
    # Japanese indicators
    japanese_indicators = ['japanese', 'j-pop', 'japan']
    if any(indicator in artist_name_lower for indicator in japanese_indicators) or \
       any(indicator in genres_lower for indicator in japanese_indicators):
        return "japanese"
    
    # Default to English for unknown
    return "english"

 
