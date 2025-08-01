import requests
import time
from typing import Dict, List, Optional
import os

class QlooService:
    """Optimized Qloo API service with minimal overhead"""
    
    def __init__(self):
        self.api_key = os.getenv('QLOO_API_KEY')
        self.base_url = "https://hackathon.api.qloo.com/v2"
        self.headers = {"X-API-Key": self.api_key}
    
    def get_tag_ids_fast(self, tags: List[str]) -> List[str]:
        """Get tag IDs efficiently - search Qloo database for existing tags (limit to 8)"""
        tag_ids = []
        successful_tags = []
        
        print(f"Searching Qloo database for {len(tags)} tags (max 8)...")
        
        # Fallback tags that are known to work
        fallback_tags = {
            "movie": ["drama", "romantic", "action", "comedy", "family"],
            "tv_show": ["drama", "romantic", "comedy", "family", "reality"],
            "podcast": ["entertainment", "interviews", "music", "cultural", "lifestyle"],
            "book": ["romance", "drama", "fiction", "cultural", "contemporary"],
            "artist": ["pop", "mainstream", "indian", "cultural", "contemporary"]
        }
        
        # Try original tags first
        for tag in tags[:8]:
            try:
                # Search for the tag in Qloo's database
                url = f"{self.base_url}/tags"
                params = {
                    "filter.query": tag,
                    "limit": 3,  # Get fewer results to find best match
                    "sort": "relevance"  # Sort by relevance, not popularity
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", {}).get("tags", [])
                    
                    if results:
                        # Find the most relevant tag (not necessarily most popular)
                        best_tag = None
                        for result in results:
                            tag_id = result.get("id", "")
                            tag_name = result.get("name", "").lower()
                            tag_type = result.get("type", "")
                            
                            # Check if this is a relevant tag type for the domain
                            if any(category in tag_type.lower() for category in [
                                'music', 'genre', 'style', 'audience', 'character', 
                                'theme', 'plot', 'subgenre', 'mood', 'emotion'
                            ]):
                                best_tag = result
                                break
                        
                        # If no specific relevant tag found, use the first result
                        if not best_tag and results:
                            best_tag = results[0]
                        
                        if best_tag:
                            tag_ids.append(best_tag["id"])
                            successful_tags.append(tag)
                            print(f"✓ '{tag}' → {best_tag['name']} ({best_tag['id']})")
                        else:
                            print(f"✗ '{tag}' - no relevant tags found")
                    else:
                        print(f"✗ '{tag}' - not found in Qloo database")
                else:
                    print(f"✗ '{tag}' - API error: {response.status_code}")
                
                time.sleep(0.05)  # 50ms delay between requests
                
            except Exception as e:
                print(f"Error searching for tag '{tag}': {e}")
                continue
        
        # If we didn't find enough tags, try fallback tags
        if len(tag_ids) < 4:
            print(f"Only found {len(tag_ids)} tags, trying fallback tags...")
            
            # Determine domain from tags or use general fallback
            domain = "artist"  # default
            if any(tag in ["drama", "romantic", "action", "comedy", "family"] for tag in tags):
                domain = "movie"
            elif any(tag in ["entertainment", "interviews", "music"] for tag in tags):
                domain = "podcast"
            elif any(tag in ["romance", "fiction", "literary"] for tag in tags):
                domain = "book"
            
            fallback_list = fallback_tags.get(domain, ["drama", "romantic", "cultural", "entertainment", "contemporary"])
            
            for fallback_tag in fallback_list:
                if len(tag_ids) >= 8:  # Stop if we have enough tags
                    break
                    
                try:
                    url = f"{self.base_url}/tags"
                    params = {
                        "filter.query": fallback_tag,
                        "limit": 2,
                        "sort": "relevance"
                    }
                    
                    response = requests.get(url, headers=self.headers, params=params, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", {}).get("tags", [])
                        
                        if results:
                            best_tag = results[0]
                            tag_ids.append(best_tag["id"])
                            successful_tags.append(fallback_tag)
                            print(f"✓ Fallback '{fallback_tag}' → {best_tag['name']} ({best_tag['id']})")
                    
                    time.sleep(0.05)
                    
                except Exception as e:
                    print(f"Error searching for fallback tag '{fallback_tag}': {e}")
                    continue
        
        print(f"Successfully found {len(successful_tags)}/{len(tags[:5])} tags in Qloo database")
        print(f"Successful tags: {successful_tags}")
        return tag_ids
    
    def get_recommendations_fast(self, tag_ids: List[str], limit: int = 15) -> List[Dict]:
        """Get recommendations efficiently - single API call with retry logic"""
        if not tag_ids:
            print("No tag IDs provided for recommendations")
            return []
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/insights"
                params = {
                    "filter.type": "urn:entity:artist",
                    "filter.tags": ",".join(tag_ids[:8]),  # Use up to 8 tags
                    "limit": limit * 2,  # Get more to filter later
                    "sort": "relevance"
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    entities = data.get("entities", [])
                    
                    # Extract only essential data
                    recommendations = []
                    for entity in entities[:limit]:
                        recommendations.append({
                            "id": entity.get("id"),
                            "name": entity.get("name"),
                            "type": entity.get("type"),
                            "popularity": entity.get("popularity", 0)
                        })
                    
                    print(f"✓ Got {len(recommendations)} recommendations from Qloo (attempt {attempt + 1})")
                    return recommendations
                elif response.status_code == 429:  # Rate limit
                    print(f"⚠️ Rate limited by Qloo API (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                else:
                    print(f"❌ Qloo recommendations failed: {response.status_code} (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ Qloo API timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            except Exception as e:
                print(f"Error getting Qloo recommendations (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
        
        print("❌ All Qloo recommendation attempts failed")
        return []
    
    def get_cross_domain_recommendations(self, tag_ids: List[str], domain: str, limit: int = 10, location: str = None, location_radius: int = 50000) -> List[Dict]:
        """Get cross-domain recommendations efficiently with enhanced data and location support"""
        if not tag_ids:
            print(f"No tag IDs provided for domain: {domain}")
            return []
        
        # Map domain names to correct Qloo entity types
        domain_type_mapping = {
            "movie": "movie",
            "tv_show": "tv_show", 
            "podcast": "podcast",
            "book": "book",
            "artist": "artist"
        }
        
        entity_type = domain_type_mapping.get(domain, domain)
        
        try:
            # Use the correct endpoint and parameters like the old backend
            url = f"{self.base_url}/insights"
            params = {
                "filter.type": f"urn:entity:{entity_type}",
                "filter.tags": ",".join(tag_ids[:5]),
                "filter.popularity.min": 0.05,  # Very low threshold to get more results
                "limit": max(limit * 6, 60),  # Get much more to ensure we have enough recommendations
                "sort": "relevance"  # Sort by relevance, not popularity
            }
            
            # Add location-based signals if provided
            if location:
                params["signal.location.query"] = location
                params["signal.location.radius"] = location_radius
                print(f"Using location-based recommendations for: {location} (radius: {location_radius}m)")
            
            print(f"Requesting {domain} recommendations with tags: {tag_ids[:3]}...")
            print(f"Using entity type: urn:entity:{entity_type}")
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get("results", {}).get("entities", [])  # Use correct path
                print(f"Received {len(entities)} entities for domain: {domain}")
                
                recommendations = []
                for entity in entities[:limit * 3]:  # Process more entities to get enough recommendations
                    # Extract properties based on domain type
                    properties = entity.get("properties", {})
                    
                    recommendation = {
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "type": entity.get("type"),
                        "popularity": entity.get("popularity", 0),
                        "affinity_score": round(entity.get("popularity", 0) * 0.8, 2),
                        "cultural_relevance": entity.get("cultural_relevance", 0),
                        "tags": entity.get("tags", []),
                        "location_based": location is not None,
                        "location": location if location else None,
                        "properties": {
                            "image": {"url": properties.get("image", {}).get("url")} if properties.get("image") else None,
                            "description": properties.get("description"),
                            "url": properties.get("url"),
                            "external_urls": properties.get("external_urls", {})
                        }
                    }
                    
                    # Domain-specific properties
                    if domain == "movie":
                        recommendation["properties"].update({
                            "year": properties.get("year"),
                            "genre": properties.get("genre"),
                            "director": properties.get("director"),
                            "runtime": properties.get("runtime"),
                            "rating": properties.get("rating")
                        })
                    elif domain == "book":
                        recommendation["properties"].update({
                            "author": properties.get("author"),
                            "publisher": properties.get("publisher"),
                            "year": properties.get("year"),
                            "genre": properties.get("genre"),
                            "language": properties.get("language")
                        })
                    elif domain == "podcast":
                        recommendation["properties"].update({
                            "host": properties.get("host"),
                            "episodes": properties.get("episodes"),
                            "genre": properties.get("genre"),
                            "language": properties.get("language")
                        })
                    elif domain == "tv_show":
                        recommendation["properties"].update({
                            "year": properties.get("year"),
                            "genre": properties.get("genre"),
                            "seasons": properties.get("seasons"),
                            "episodes": properties.get("episodes"),
                            "rating": properties.get("rating")
                        })
                    elif domain == "artist":
                        recommendation["properties"].update({
                            "followers": properties.get("followers"),
                            "albums": properties.get("albums"),
                            "genres": properties.get("genres", []),
                            "spotify_url": properties.get("spotify_url")
                        })
                    
                    recommendations.append(recommendation)
                
                # Ensure we return at least the requested number of recommendations
                final_recommendations = recommendations[:limit]
                print(f"Processed {len(final_recommendations)} recommendations for domain: {domain}")
                return final_recommendations
            else:
                print(f"Qloo API error for domain {domain}: {response.status_code} - {response.text}")
                print(f"Response content: {response.text[:500]}...")
            
        except Exception as e:
            print(f"Cross-domain recommendations error for {domain}: {e}")
        
        return []
    
    def search_entity(self, query: str, entity_type: str = "artist") -> Optional[Dict]:
        """Search for entity - single API call"""
        try:
            url = f"{self.base_url}/search"
            params = {
                "query": query,
                "filter.type": f"urn:entity:{entity_type}",
                "limit": 1
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                entities = data.get("results", {}).get("entities", [])
                if entities:
                    return entities[0]
            
        except Exception as e:
            print(f"Entity search error: {e}")
        
        return None
    
    def get_artist_recommendations_by_tags(self, tag_ids: List[str], artist_ids: List[str] = None, limit: int = 15) -> List[Dict]:
        """Get artist recommendations using tags and optional artist signals"""
        if not tag_ids:
            return []
        
        try:
            url = f"{self.base_url}/insights"
            params = {
                "filter.type": "urn:entity:artist",
                "filter.tags": ",".join(tag_ids[:5]),
                "limit": limit,
                "sort": "relevance"
            }
            
            # Add artist signals if provided
            if artist_ids:
                signals = [f"urn:entity:artist:{artist_id}" for artist_id in artist_ids[:5]]
                params["signals"] = ",".join(signals)
            
            response = requests.get(url, headers=self.headers, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                entities = data.get("entities", [])
                
                recommendations = []
                for entity in entities:
                    recommendations.append({
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "popularity": entity.get("popularity", 0)
                    })
                
                return recommendations
            
        except Exception as e:
            print(f"Artist recommendations error: {e}")
        
        return []
    
    def get_fallback_tag_ids(self, domain: str) -> List[str]:
        """Get fallback tag IDs for when primary tag search fails"""
        # Known working tag IDs from Qloo database
        fallback_tag_ids = {
            "music": [
                "urn:tag:pop",  # Pop music
                "urn:tag:mainstream",  # Mainstream
                "urn:tag:indian",  # Indian music
                "urn:tag:cultural",  # Cultural
                "urn:tag:contemporary"  # Contemporary
            ],
            "movie": [
                "urn:tag:drama",  # Drama
                "urn:tag:romantic",  # Romantic
                "urn:tag:action",  # Action
                "urn:tag:comedy",  # Comedy
                "urn:tag:family"  # Family
            ],
            "book": [
                "urn:tag:romance",  # Romance
                "urn:tag:drama",  # Drama
                "urn:tag:fiction",  # Fiction
                "urn:tag:cultural",  # Cultural
                "urn:tag:contemporary"  # Contemporary
            ],
            "podcast": [
                "urn:tag:entertainment",  # Entertainment
                "urn:tag:interviews",  # Interviews
                "urn:tag:music",  # Music
                "urn:tag:cultural",  # Cultural
                "urn:tag:lifestyle"  # Lifestyle
            ]
        }
        
        return fallback_tag_ids.get(domain, fallback_tag_ids["music"])
    
    def get_enhanced_recommendations(self, tag_ids: List[str], user_artists: List[Dict] = None, 
                                   user_tracks: List[Dict] = None, location: str = None, 
                                   location_radius: int = 5000000, cultural_context: Dict = None, 
                                   limit: int = 15) -> List[Dict]:
        """Get enhanced recommendations with cultural intelligence and user signals"""
        try:
            if not tag_ids:
                print("No tag IDs provided for enhanced recommendations")
                return []
            
            print(f"Getting enhanced recommendations with {len(tag_ids)} tag IDs: {tag_ids[:3]}")
            if location:
                print(f"📍 Location-aware recommendations for: {location}")
            
            # First try the standard recommendations endpoint
            recommendations = self.get_recommendations_fast(tag_ids, limit * 2)
            
            if not recommendations:
                print("Standard recommendations failed, trying fallback approach...")
                # Fallback: try with individual tag IDs
                for tag_id in tag_ids[:3]:
                    try:
                        fallback_recs = self.get_recommendations_fast([tag_id], limit)
                        if fallback_recs:
                            recommendations.extend(fallback_recs)
                            print(f"Got {len(fallback_recs)} recommendations from tag {tag_id}")
                    except Exception as e:
                        print(f"Fallback recommendation failed for tag {tag_id}: {e}")
                        continue
            
            # If still no recommendations, use hardcoded fallback artists
            if not recommendations:
                print("No Qloo recommendations found, using hardcoded fallback artists")
                recommendations = self.get_hardcoded_fallback_artists(cultural_context, limit)
                if recommendations:
                    print(f"✓ Using {len(recommendations)} hardcoded fallback artists")
                else:
                    print("⚠️ Even hardcoded fallback failed, returning empty list")
            
            # Enhance recommendations with cultural context and location awareness
            enhanced_recommendations = []
            for entity in recommendations:
                # Add cultural relevance score
                cultural_relevance = 0.0
                if cultural_context and cultural_context.get("cultural_elements"):
                    cultural_elements = cultural_context["cultural_elements"]
                    entity_name = entity.get("name", "").lower()
                    entity_description = entity.get("description", "").lower()
                    
                    for element in cultural_elements:
                        if element.lower() in entity_name or element.lower() in entity_description:
                            cultural_relevance += 0.3
                
                # Add location relevance score
                location_relevance = 0.0
                if location:
                    entity_name = entity.get("name", "").lower()
                    entity_description = entity.get("description", "").lower()
                    
                    # Check if entity is relevant to the location
                    if location.lower() in entity_name or location.lower() in entity_description:
                        location_relevance = 0.8
                    elif any(loc_word in entity_name for loc_word in ["local", "regional", "native"]):
                        location_relevance = 0.5
                    elif any(loc_word in entity_name for loc_word in ["indian", "bollywood", "hindi", "punjabi", "desi"]):
                        # For Indian locations, give relevance to Indian artists
                        if "mumbai" in location.lower() or "india" in location.lower():
                            location_relevance = 0.4
                    elif any(loc_word in entity_name for loc_word in ["western", "pop", "rock", "hip_hop"]):
                        # For Western locations, give relevance to Western artists
                        if any(western_loc in location.lower() for western_loc in ["new york", "london", "toronto", "sydney", "us", "uk", "canada", "australia"]):
                            location_relevance = 0.4
                
                # Create enhanced entity
                enhanced_entity = {
                    "id": entity.get("id", f"fallback_{len(enhanced_recommendations)}"),
                    "name": entity.get("name", "Unknown Artist"),
                    "type": entity.get("type", "urn:entity:artist"),
                    "popularity": entity.get("popularity", 0.5),
                    "cultural_relevance": min(cultural_relevance, 1.0),
                    "location_relevance": min(location_relevance, 1.0),
                    "description": entity.get("description", ""),
                    "tags": entity.get("tags", [])
                }
                
                enhanced_recommendations.append(enhanced_entity)
            
            # Sort by cultural relevance, location relevance, and popularity
            enhanced_recommendations.sort(key=lambda x: (
                x.get("cultural_relevance", 0) * 0.35 + 
                x.get("location_relevance", 0) * 0.35 + 
                x.get("popularity", 0) * 0.3
            ), reverse=True)
            
            print(f"Returning {len(enhanced_recommendations)} enhanced recommendations")
            return enhanced_recommendations[:limit]
            
        except Exception as e:
            print(f"Error in enhanced recommendations: {e}")
            # Return hardcoded fallback
            return self.get_hardcoded_fallback_artists(cultural_context, limit)
    
    def get_cultural_insights(self, location: str, domain: str = "music") -> List[Dict]:
        """Get cultural insights for a location and domain"""
        try:
            # Simple cultural insights based on location
            cultural_insights = []
            
            if "mumbai" in location.lower() or "india" in location.lower():
                cultural_insights = [
                    {
                        "insight": "Bollywood music dominates the local scene",
                        "relevance": 0.9,
                        "category": "music_style"
                    },
                    {
                        "insight": "Hindi and regional languages are popular",
                        "relevance": 0.8,
                        "category": "language"
                    },
                    {
                        "insight": "Traditional instruments blend with modern beats",
                        "relevance": 0.7,
                        "category": "instrumentation"
                    }
                ]
            elif "new york" in location.lower() or "us" in location.lower():
                cultural_insights = [
                    {
                        "insight": "Diverse music scene with multiple genres",
                        "relevance": 0.9,
                        "category": "diversity"
                    },
                    {
                        "insight": "Hip-hop and pop dominate mainstream",
                        "relevance": 0.8,
                        "category": "music_style"
                    },
                    {
                        "insight": "English is the primary language",
                        "relevance": 0.7,
                        "category": "language"
                    }
                ]
            elif "london" in location.lower() or "uk" in location.lower():
                cultural_insights = [
                    {
                        "insight": "Strong indie and alternative scene",
                        "relevance": 0.9,
                        "category": "music_style"
                    },
                    {
                        "insight": "British rock and pop heritage",
                        "relevance": 0.8,
                        "category": "heritage"
                    },
                    {
                        "insight": "Multicultural influences in modern music",
                        "relevance": 0.7,
                        "category": "diversity"
                    }
                ]
            else:
                # Generic insights for other locations
                cultural_insights = [
                    {
                        "insight": "Global music trends influence local preferences",
                        "relevance": 0.6,
                        "category": "global_trends"
                    },
                    {
                        "insight": "Local artists blend traditional and modern styles",
                        "relevance": 0.5,
                        "category": "fusion"
                    }
                ]
            
            return cultural_insights[:5]  # Return top 5 insights
            
        except Exception as e:
            print(f"Error getting cultural insights: {e}")
            return []

    def get_hardcoded_fallback_artists(self, cultural_context: Dict = None, limit: int = 15) -> List[Dict]:
        """Get hardcoded fallback artists when Qloo API fails"""
        try:
            # Determine cultural context for appropriate fallback artists
            region = cultural_context.get("region", "global") if cultural_context else "global"
            language_preference = cultural_context.get("language_preference", "any") if cultural_context else "any"
            
            # Hardcoded artist lists by region and language
            fallback_artists = {
                "south_asia": {
                    "hindi": [
                        {"id": "fallback_1", "name": "Arijit Singh", "type": "urn:entity:artist", "popularity": 0.9, "description": "Popular Hindi playback singer"},
                        {"id": "fallback_2", "name": "Neha Kakkar", "type": "urn:entity:artist", "popularity": 0.8, "description": "Bollywood playback singer"},
                        {"id": "fallback_3", "name": "Badshah", "type": "urn:entity:artist", "popularity": 0.8, "description": "Indian rapper and music producer"},
                        {"id": "fallback_4", "name": "Harrdy Sandhu", "type": "urn:entity:artist", "popularity": 0.7, "description": "Punjabi singer and actor"},
                        {"id": "fallback_5", "name": "Shreya Ghoshal", "type": "urn:entity:artist", "popularity": 0.9, "description": "Classical and playback singer"},
                        {"id": "fallback_6", "name": "Atif Aslam", "type": "urn:entity:artist", "popularity": 0.8, "description": "Pakistani singer and actor"},
                        {"id": "fallback_7", "name": "Pritam", "type": "urn:entity:artist", "popularity": 0.8, "description": "Bollywood music composer"},
                        {"id": "fallback_8", "name": "A.R. Rahman", "type": "urn:entity:artist", "popularity": 0.9, "description": "Oscar-winning music composer"},
                        {"id": "fallback_9", "name": "Diljit Dosanjh", "type": "urn:entity:artist", "popularity": 0.8, "description": "Punjabi singer and actor"},
                        {"id": "fallback_10", "name": "Guru Randhawa", "type": "urn:entity:artist", "popularity": 0.7, "description": "Punjabi singer and songwriter"}
                    ],
                    "any": [
                        {"id": "fallback_11", "name": "The Weeknd", "type": "urn:entity:artist", "popularity": 0.9, "description": "Canadian singer and songwriter"},
                        {"id": "fallback_12", "name": "Ed Sheeran", "type": "urn:entity:artist", "popularity": 0.9, "description": "English singer-songwriter"},
                        {"id": "fallback_13", "name": "Taylor Swift", "type": "urn:entity:artist", "popularity": 0.9, "description": "American singer-songwriter"},
                        {"id": "fallback_14", "name": "Justin Bieber", "type": "urn:entity:artist", "popularity": 0.8, "description": "Canadian pop singer"},
                        {"id": "fallback_15", "name": "Ariana Grande", "type": "urn:entity:artist", "popularity": 0.9, "description": "American singer and actress"}
                    ]
                },
                "western": {
                    "english": [
                        {"id": "fallback_16", "name": "The Weeknd", "type": "urn:entity:artist", "popularity": 0.9, "description": "Canadian singer and songwriter"},
                        {"id": "fallback_17", "name": "Ed Sheeran", "type": "urn:entity:artist", "popularity": 0.9, "description": "English singer-songwriter"},
                        {"id": "fallback_18", "name": "Taylor Swift", "type": "urn:entity:artist", "popularity": 0.9, "description": "American singer-songwriter"},
                        {"id": "fallback_19", "name": "Post Malone", "type": "urn:entity:artist", "popularity": 0.8, "description": "American rapper and singer"},
                        {"id": "fallback_20", "name": "Dua Lipa", "type": "urn:entity:artist", "popularity": 0.8, "description": "English singer and songwriter"},
                        {"id": "fallback_21", "name": "Billie Eilish", "type": "urn:entity:artist", "popularity": 0.8, "description": "American singer and songwriter"},
                        {"id": "fallback_22", "name": "Harry Styles", "type": "urn:entity:artist", "popularity": 0.8, "description": "English singer and actor"},
                        {"id": "fallback_23", "name": "Coldplay", "type": "urn:entity:artist", "popularity": 0.8, "description": "English rock band"},
                        {"id": "fallback_24", "name": "Imagine Dragons", "type": "urn:entity:artist", "popularity": 0.7, "description": "American pop rock band"},
                        {"id": "fallback_25", "name": "Maroon 5", "type": "urn:entity:artist", "popularity": 0.7, "description": "American pop rock band"}
                    ],
                    "any": [
                        {"id": "fallback_26", "name": "Martin Garrix", "type": "urn:entity:artist", "popularity": 0.8, "description": "Dutch DJ and record producer"},
                        {"id": "fallback_27", "name": "The Chainsmokers", "type": "urn:entity:artist", "popularity": 0.7, "description": "American DJ duo"},
                        {"id": "fallback_28", "name": "Calvin Harris", "type": "urn:entity:artist", "popularity": 0.8, "description": "Scottish DJ and record producer"},
                        {"id": "fallback_29", "name": "David Guetta", "type": "urn:entity:artist", "popularity": 0.7, "description": "French DJ and record producer"},
                        {"id": "fallback_30", "name": "Marshmello", "type": "urn:entity:artist", "popularity": 0.7, "description": "American electronic music producer"}
                    ]
                }
            }
            
            # Get appropriate artist list
            if region in fallback_artists:
                if language_preference in fallback_artists[region]:
                    artists = fallback_artists[region][language_preference]
                else:
                    artists = fallback_artists[region]["any"]
            else:
                # Global fallback
                artists = [
                    {"id": "fallback_31", "name": "The Weeknd", "type": "urn:entity:artist", "popularity": 0.9, "description": "Canadian singer and songwriter"},
                    {"id": "fallback_32", "name": "Ed Sheeran", "type": "urn:entity:artist", "popularity": 0.9, "description": "English singer-songwriter"},
                    {"id": "fallback_33", "name": "Arijit Singh", "type": "urn:entity:artist", "popularity": 0.9, "description": "Popular Hindi playback singer"},
                    {"id": "fallback_34", "name": "Taylor Swift", "type": "urn:entity:artist", "popularity": 0.9, "description": "American singer-songwriter"},
                    {"id": "fallback_35", "name": "Post Malone", "type": "urn:entity:artist", "popularity": 0.8, "description": "American rapper and singer"}
                ]
            
            # Add cultural relevance score
            for artist in artists:
                cultural_relevance = 0.0
                if cultural_context and cultural_context.get("cultural_elements"):
                    cultural_elements = cultural_context["cultural_elements"]
                    artist_name = artist.get("name", "").lower()
                    
                    for element in cultural_elements:
                        if element.lower() in artist_name:
                            cultural_relevance += 0.3
                
                artist["cultural_relevance"] = min(cultural_relevance, 1.0)
                artist["tags"] = []
            
            print(f"Using {len(artists)} hardcoded fallback artists for region: {region}, language: {language_preference}")
            return artists[:limit]
            
        except Exception as e:
            print(f"Error getting hardcoded fallback artists: {e}")
            # Ultimate fallback
            return [
                {"id": "fallback_ultimate", "name": "The Weeknd", "type": "urn:entity:artist", "popularity": 0.9, "cultural_relevance": 0.0, "description": "Canadian singer and songwriter", "tags": []}
            ]
    
    def get_music_recommendations_with_user_signals(self, tag_ids: List[str], user_artist_ids: List[str], user_track_ids: List[str], user_country: str = None, limit: int = 15) -> List[Dict]:
        """Get music recommendations using user's listening history as signals"""
        if not tag_ids:
            print(f"No tag IDs provided for music recommendations")
            return []
        
        try:
            # Use the insights endpoint for music recommendations
            url = f"{self.base_url}/insights"
            params = {
                "filter.type": "urn:entity:artist",
                "filter.tags": ",".join(tag_ids[:5]),
                "filter.popularity.min": 0.05,  # Low threshold for diverse results
                "limit": limit * 3,  # Get more to ensure enough recommendations
                "sort": "relevance"  # Sort by relevance
            }
            
            # Add user's listening history as signals (prioritizing most played)
            signals = []
            
            # Add user's top artists as signals (these get priority)
            for artist_id in user_artist_ids[:8]:  # Top 8 most played artists
                signals.append(f"urn:entity:artist:{artist_id}")
            
            # Add user's top tracks as signals
            for track_id in user_track_ids[:8]:  # Top 8 most played tracks
                signals.append(f"urn:entity:track:{track_id}")
            
            if signals:
                params["signals"] = ",".join(signals)
                print(f"Using {len(signals)} user listening signals")
            
            # Add location-based signals if user country is provided
            if user_country:
                location = self._get_location_from_country(user_country)
                if location:
                    params["signal.location.query"] = location
                    params["signal.location.radius"] = 50000  # 50km radius
                    print(f"Using location-based signals for: {location}")
            
            print(f"Requesting music recommendations with tags: {tag_ids[:3]}...")
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get("results", {}).get("entities", [])
                print(f"Received {len(entities)} entities for music recommendations")
                
                recommendations = []
                for entity in entities[:limit * 2]:  # Process more entities to get enough recommendations
                    # Extract properties
                    properties = entity.get("properties", {})
                    
                    recommendation = {
                        "id": entity.get("id"),
                        "name": entity.get("name"),
                        "type": entity.get("type"),
                        "popularity": entity.get("popularity", 0),
                        "affinity_score": round(entity.get("popularity", 0) * 0.8, 2),
                        "cultural_relevance": entity.get("cultural_relevance", 0),
                        "tags": entity.get("tags", []),
                        "properties": {
                            "image": {"url": properties.get("image", {}).get("url")} if properties.get("image") else None,
                            "description": properties.get("description"),
                            "url": properties.get("url"),
                            "external_urls": properties.get("external_urls", {}),
                            "followers": properties.get("followers"),
                            "albums": properties.get("albums"),
                            "genres": properties.get("genres", []),
                            "spotify_url": properties.get("spotify_url")
                        }
                    }
                    
                    recommendations.append(recommendation)
                
                # Ensure we return at least the requested number of recommendations
                final_recommendations = recommendations[:limit]
                print(f"Processed {len(final_recommendations)} music recommendations")
                return final_recommendations
            else:
                print(f"Qloo API error for music recommendations: {response.status_code} - {response.text}")
                print(f"Response content: {response.text[:500]}...")
            
        except Exception as e:
            print(f"Music recommendations error: {e}")
        
        return []
    
    def _get_location_from_country(self, country_code: str) -> str:
        """Convert country code to location name for Qloo"""
        country_location_map = {
            "IN": "Mumbai, India",
            "US": "New York, USA", 
            "GB": "London, UK",
            "CA": "Toronto, Canada",
            "AU": "Sydney, Australia",
            "DE": "Berlin, Germany",
            "FR": "Paris, France",
            "JP": "Tokyo, Japan",
            "KR": "Seoul, South Korea",
            "BR": "São Paulo, Brazil"
        }
        return country_location_map.get(country_code.upper(), "New York, USA") 