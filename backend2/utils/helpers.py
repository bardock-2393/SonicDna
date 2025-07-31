import re
import time
from typing import Dict, List, Optional

def extract_playlist_id_from_url(spotify_url: str) -> Optional[str]:
    """Extract playlist ID from Spotify URL"""
    patterns = [
        r'spotify\.com/playlist/([a-zA-Z0-9]+)',
        r'spotify\.com/playlist/([a-zA-Z0-9]+)\?',
        r'playlist/([a-zA-Z0-9]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, spotify_url)
        if match:
            return match.group(1)
    
    return None

def rank_recommendations_fast(entities: List[Dict], user_preferences: Dict) -> List[Dict]:
    """Simple scoring algorithm for fast ranking"""
    if not entities:
        return []
    
    for entity in entities:
        score = 0.0
        
        # Base popularity score
        popularity = entity.get("popularity", 0)
        score += popularity * 0.3
        
        # User preference match (if available)
        if user_preferences:
            # Simple keyword matching
            entity_name = entity.get("name", "").lower()
            user_artists = [artist.lower() for artist in user_preferences.get("artists", [])]
            
            for artist in user_artists:
                if artist in entity_name or entity_name in artist:
                    score += 0.4
                    break
        
        entity["relevance_score"] = score
    
    # Sort by relevance score
    return sorted(entities, key=lambda x: x.get("relevance_score", 0), reverse=True)

def apply_cultural_intelligence_fast(recommendations: List[Dict], user_country: str) -> List[Dict]:
    """Lightweight cultural filtering using predefined rules"""
    if not user_country or not recommendations:
        return recommendations
    
    # Simple cultural adjustments based on country
    cultural_boosts = {
        "US": ["pop", "hip hop", "country"],
        "UK": ["pop", "rock", "indie"],
        "JP": ["j-pop", "j-rock", "anime"],
        "KR": ["k-pop", "k-hip hop"],
        "IN": ["bollywood", "indian pop"],
        "BR": ["samba", "bossa nova", "brazilian pop"],
        "MX": ["mexican pop", "ranchera", "norteño"],
        "FR": ["french pop", "chanson"],
        "DE": ["german pop", "schlager"],
        "IT": ["italian pop", "opera"]
    }
    
    boost_tags = cultural_boosts.get(user_country.upper(), [])
    
    for entity in recommendations:
        entity_name = entity.get("name", "").lower()
        for tag in boost_tags:
            if tag.lower() in entity_name:
                entity["relevance_score"] = entity.get("relevance_score", 0) + 0.2
                break
    
    return recommendations

def calculate_context_relevance_score(track: Dict, context_type: str, search_query: str = "") -> float:
    """Calculate relevance score for track in context"""
    score = 0.0
    
    # Base score from popularity
    popularity = track.get("popularity", 0)
    score += popularity * 0.3
    
    # Context matching
    track_name = track.get("name", "").lower()
    artist_name = track.get("artist", "").lower()
    
    context_keywords = {
        "workout": ["energetic", "fast", "motivational", "pump", "energy"],
        "study": ["calm", "focus", "concentration", "ambient", "instrumental"],
        "party": ["dance", "upbeat", "fun", "celebration", "vibe"],
        "relaxation": ["chill", "calm", "peaceful", "ambient", "soft"],
        "romantic": ["love", "romantic", "intimate", "passionate", "sweet"]
    }
    
    keywords = context_keywords.get(context_type, [])
    for keyword in keywords:
        if keyword in track_name or keyword in artist_name:
            score += 0.3
            break
    
    # Search query matching
    if search_query:
        query_lower = search_query.lower()
        if query_lower in track_name or query_lower in artist_name:
            score += 0.4
    
    return min(score, 1.0)

def get_context_keywords(context_type: str) -> List[str]:
    """Get keywords for context type"""
    context_keywords = {
        "workout": ["energetic", "motivational", "pump", "energy", "fast"],
        "study": ["focus", "concentration", "calm", "ambient", "instrumental"],
        "party": ["dance", "upbeat", "fun", "celebration", "vibe"],
        "relaxation": ["chill", "calm", "peaceful", "ambient", "soft"],
        "romantic": ["love", "romantic", "intimate", "passionate", "sweet"],
        "general": ["popular", "mainstream", "trending", "hit", "chart"]
    }
    
    return context_keywords.get(context_type, ["popular", "mainstream"])

def get_cultural_keywords(user_country: str, location: str = None) -> List[str]:
    """Get cultural keywords for location"""
    cultural_keywords = {
        "US": ["american", "pop", "hip hop", "country"],
        "UK": ["british", "pop", "rock", "indie"],
        "JP": ["japanese", "j-pop", "j-rock", "anime"],
        "KR": ["korean", "k-pop", "k-hip hop"],
        "IN": ["indian", "bollywood", "indian pop"],
        "BR": ["brazilian", "samba", "bossa nova"],
        "MX": ["mexican", "mexican pop", "ranchera"],
        "FR": ["french", "french pop", "chanson"],
        "DE": ["german", "german pop", "schlager"],
        "IT": ["italian", "italian pop", "opera"]
    }
    
    return cultural_keywords.get(user_country.upper(), ["international", "global"])

def sort_by_relevance(entities: List[Dict], user_artists: List[str], user_genres: List[str], 
                     context_type: str, user_country: str, location: str = None) -> List[Dict]:
    """Sort entities by relevance score"""
    if not entities:
        return []
    
    for entity in entities:
        score = 0.0
        
        # Popularity score
        popularity = entity.get("popularity", 0)
        score += popularity * 0.2
        
        # User preference match
        entity_name = entity.get("name", "").lower()
        for artist in user_artists:
            if artist.lower() in entity_name or entity_name in artist.lower():
                score += 0.3
                break
        
        # Genre match
        entity_genres = entity.get("genres", [])
        for genre in user_genres:
            if genre.lower() in [g.lower() for g in entity_genres]:
                score += 0.2
                break
        
        # Context match
        context_keywords = get_context_keywords(context_type)
        for keyword in context_keywords:
            if keyword.lower() in entity_name:
                score += 0.2
                break
        
        # Cultural match
        cultural_keywords = get_cultural_keywords(user_country, location)
        for keyword in cultural_keywords:
            if keyword.lower() in entity_name:
                score += 0.1
                break
        
        entity["relevance_score"] = score
    
    return sorted(entities, key=lambda x: x.get("relevance_score", 0), reverse=True)

def get_fallback_recommendations(context_type: str) -> List[Dict]:
    """Get fallback recommendations when APIs fail"""
    fallback_artists = {
        "workout": [
            {"name": "The Weeknd", "popularity": 0.9},
            {"name": "Dua Lipa", "popularity": 0.8},
            {"name": "Post Malone", "popularity": 0.8},
            {"name": "Ariana Grande", "popularity": 0.9},
            {"name": "Drake", "popularity": 0.9}
        ],
        "study": [
            {"name": "Lofi Girl", "popularity": 0.7},
            {"name": "Chillhop Music", "popularity": 0.6},
            {"name": "Spotify", "popularity": 0.8},
            {"name": "Peaceful Piano", "popularity": 0.6},
            {"name": "Deep Focus", "popularity": 0.7}
        ],
        "party": [
            {"name": "The Weeknd", "popularity": 0.9},
            {"name": "Dua Lipa", "popularity": 0.8},
            {"name": "Bad Bunny", "popularity": 0.9},
            {"name": "Taylor Swift", "popularity": 0.9},
            {"name": "Drake", "popularity": 0.9}
        ],
        "relaxation": [
            {"name": "Lofi Girl", "popularity": 0.7},
            {"name": "Chillhop Music", "popularity": 0.6},
            {"name": "Peaceful Piano", "popularity": 0.6},
            {"name": "Nature Sounds", "popularity": 0.5},
            {"name": "Ambient Music", "popularity": 0.6}
        ],
        "romantic": [
            {"name": "Ed Sheeran", "popularity": 0.8},
            {"name": "Adele", "popularity": 0.9},
            {"name": "John Legend", "popularity": 0.7},
            {"name": "Sam Smith", "popularity": 0.8},
            {"name": "Lewis Capaldi", "popularity": 0.7}
        ]
    }
    
    return fallback_artists.get(context_type, fallback_artists["party"])

def validate_input_data(data: Dict, required_fields: List[str]) -> bool:
    """Validate input data has required fields"""
    if not data:
        return False
    
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    
    return True

def sanitize_string(text: str) -> str:
    """Sanitize string input"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    return sanitized.strip()

def get_timestamp() -> float:
    """Get current timestamp"""
    return time.time() 