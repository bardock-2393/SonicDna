import requests
import time
from utils import retry_on_failure

class QlooAPIClient:
    """Basic Qloo API client for simple operations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://hackathon.api.qloo.com/v2"
        self.headers = {"X-API-Key": api_key}
    
    def search_entity(self, query: str):
        """Search for entities in Qloo database"""
        url = f"{self.base_url}/search"
        params = {"query": query}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching Qloo entity: {e}")
            return None
    
    def get_recommendations(self, entity_type: str, tag_id: str, artist_ids=None, take: int = 1):
        """Get basic recommendations from Qloo"""
        url = f"{self.base_url}/insights"
        params = {
            "filter.type": f"urn:entity:{entity_type}",
            "filter.tags": tag_id,
            "limit": take
        }
        
        if artist_ids:
            params["signals"] = ",".join([f"urn:entity:artist:{artist}" for artist in artist_ids])
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting Qloo recommendations: {e}")
            return None

class EnhancedQlooClient:
    """Enhanced Qloo client with advanced features"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://hackathon.api.qloo.com/v2"
        self.headers = {"X-API-Key": api_key}
    
    def search_entities(self, query: str, entity_type: str = "artist", limit: int = 10):
        """Search for entities with enhanced parameters"""
        url = f"{self.base_url}/search"
        params = {
            "query": query,
            "filter.type": f"urn:entity:{entity_type}",
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("results", {}).get("entities", [])
        except Exception as e:
            print(f"Error searching entities: {e}")
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
            
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            
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
        
        # Add basic location if available
        if location:
            params["signal.location.query"] = location
            params["signal.location.radius"] = 50000
        
        # Add basic user signals
        signals = []
        for artist in user_artists[:5]:  # Even fewer signals
            signals.append(f"urn:entity:artist:{artist}")
        for track in user_tracks[:5]:
            signals.append(f"urn:entity:track:{track}")
        
        if signals:
            params["signals"] = ",".join(signals)
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
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
            
            print(f"[QLOO FALLBACK] Found {len(recommended_artists)} fallback recommendations")
            return recommended_artists
            
        except Exception as e:
            print(f"Error in fallback recommendations: {e}")
            return []
    
    def get_cultural_insights(self, location, domain="music"):
        """Get cultural insights for a location"""
        url = f"{self.base_url}/cultural-insights"
        params = {
            "location": location,
            "domain": domain,
            "limit": 10
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            insights = data.get("results", {}).get("insights", [])
            print(f"[QLOO CULTURAL] Found {len(insights)} cultural insights for {location}")
            return insights
        except Exception as e:
            print(f"Error getting cultural insights: {e}")
            return []

@retry_on_failure(max_retries=2, delay=1)
def get_qloo_tag_ids(tags, qloo_api_key):
    """Convert tag names to Qloo tag IDs with retry logic"""
    qloo_client = QlooAPIClient(qloo_api_key)
    tag_ids = []
    
    print(f"[QLOO TAGS] Converting {len(tags)} tags to Qloo tag IDs...")
    
    # Known tag mappings for common tags
    known_tags = {
        'upbeat': 'urn:tag:genre:music:upbeat',
        'energetic': 'urn:tag:style:qloo:energetic',
        'positive': 'urn:tag:audience:qloo:positive',
        'confident': 'urn:tag:character:qloo:confident',
        'determined': 'urn:tag:character:qloo:determined',
        'focused': 'urn:tag:style:qloo:focused',
        'inspiring': 'urn:tag:plot:qloo:inspiring',
        'pop': 'urn:tag:genre:music:pop',
        'electronic': 'urn:tag:genre:music:electronic',
        'hindi pop': 'urn:tag:genre:music:hindi_pop',
        'bollywood': 'urn:tag:genre:music:bollywood',
        'instrumental': 'urn:tag:genre:music:instrumental',
        'study music': 'urn:tag:genre:music:study_music',
        'india': 'urn:tag:genre:media:india',
        'acoustic': 'urn:tag:genre:music:acoustic',
        'high energy': 'urn:tag:genre:music:high_energy',
        'inspired': 'urn:tag:audience:qloo:inspired'
    }
    
    successful_tags = []
    
    for tag in tags:
        # First check known tags
        if tag.lower() in known_tags:
            tag_ids.append(known_tags[tag.lower()])
            successful_tags.append(tag)
            print(f"[QLOO TAGS] ✓ '{tag}' → {known_tags[tag.lower()]}")
            continue
        
        # Try to get tag ID from Qloo
        try:
            search_result = qloo_client.search_entity(tag)
            if search_result and search_result.get("results", {}).get("entities"):
                entity = search_result["results"]["entities"][0]
                if entity.get("subtype", "").startswith("urn:tag:"):
                    tag_id = entity.get("id")
                    tag_ids.append(tag_id)
                    successful_tags.append(tag)
                    print(f"[QLOO TAGS] ✓ '{tag}' → {tag_id}")
                else:
                    print(f"[QLOO TAGS] ✗ '{tag}' - not a tag entity")
            else:
                print(f"[QLOO TAGS] ✗ '{tag}' - not found in Qloo database")
        except Exception as e:
            print(f"[QLOO TAGS] ✗ '{tag}' - error: {e}")
    
    print(f"[QLOO TAGS] Successfully converted {len(successful_tags)}/{len(tags)} tags")
    print(f"[QLOO TAGS] Successful tags: {successful_tags}")
    
    return tag_ids

def get_qloo_artist_recommendations(tag_ids, artists, tracks, qloo_api_key, limit=15, location=None, location_radius=None):
    """Get artist recommendations from Qloo using tags and user data"""
    qloo_client = EnhancedQlooClient(qloo_api_key)
    
    try:
        recommendations = qloo_client.get_enhanced_recommendations(
            tag_ids=tag_ids,
            user_artists=artists,
            user_tracks=tracks,
            location=location,
            location_radius=location_radius,
            limit=limit
        )
        
        return recommendations
    except Exception as e:
        print(f"Error getting Qloo artist recommendations: {e}")
        return []

def get_qloo_artist_id(artist_name, qloo_api_key):
    """Get Qloo artist ID for a given artist name"""
    qloo_client = QlooAPIClient(qloo_api_key)
    
    try:
        search_result = qloo_client.search_entity(artist_name)
        if search_result and search_result.get("results", {}).get("entities"):
            for entity in search_result["results"]["entities"]:
                if entity.get("subtype") == "urn:entity:artist":
                    return entity.get("id")
        return None
    except Exception as e:
        print(f"Error getting Qloo artist ID for {artist_name}: {e}")
        return None

def get_qloo_tag_id(tag_name, qloo_api_key):
    """Get Qloo tag ID for a given tag name"""
    qloo_client = QlooAPIClient(qloo_api_key)
    
    try:
        search_result = qloo_client.search_entity(tag_name)
        if search_result and search_result.get("results", {}).get("entities"):
            for entity in search_result["results"]["entities"]:
                if entity.get("subtype", "").startswith("urn:tag:"):
                    return entity.get("id")
        return None
    except Exception as e:
        print(f"Error getting Qloo tag ID for {tag_name}: {e}")
        return None 
