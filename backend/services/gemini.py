import requests
import json
import time
from typing import Dict, List, Optional
import os

class GeminiService:
    """Optimized Gemini API service with minimal overhead"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY', "AIzaSyBNb-EtpmciV73x0VzhQdHUtaJysd4aRKM")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    
    def analyze_context_fast(self, user_context: str) -> Dict:
        """Fast context analysis - single Gemini call with focused prompt"""
        prompt = f"""
        Analyze this user context and return ONLY a JSON object with these exact fields:
        - primary_mood: string (happy, sad, energetic, calm, romantic, etc.)
        - activity_type: string (workout, study, party, relaxation, etc.)
        - energy_level: string (high, medium, low)
        - confidence: float (0.0 to 1.0)

        User context: {user_context}

        Return only the JSON object, no other text.
        """
        
        try:
            response = self._call_gemini(prompt)
            if response:
                # Try to parse JSON from response
                try:
                    # Extract JSON from response text
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                        return result
                except json.JSONDecodeError:
                    pass
                
                # Fallback parsing
                return self._parse_context_fallback(response)
            
        except Exception as e:
            print(f"Context analysis error: {e}")
        
        # Default fallback
        return {
            "primary_mood": "neutral",
            "activity_type": "general",
            "energy_level": "medium",
            "confidence": 0.5
        }
    
    def generate_optimized_tags(self, context: str, user_country: str = None) -> List[str]:
        """Generate exactly 8 most relevant tags for cross-domain recommendations"""
        # Use simple, proven working tags instead of complex ones
        simple_working_tags = [
            "romantic", "drama", "comedy", "thriller", "adventure",
            "sad", "happy", "energetic", "calm", "upbeat",
            "emotional", "nostalgic", "party", "workout", "study",
            "mystery", "inspirational", "relaxation", "motivational",
            "pop", "mainstream", "cultural", "contemporary", "indian"
        ]
        
        # Select 8 based on context
        context_lower = context.lower()
        if any(word in context_lower for word in ['happy', 'joy', 'excited', 'party']):
            return ["happy", "upbeat", "energetic", "party", "celebration", "pop", "mainstream", "contemporary"]
        elif any(word in context_lower for word in ['sad', 'melancholy', 'crying']):
            return ["sad", "emotional", "melancholic", "nostalgic", "ballad", "drama", "romantic", "cultural"]
        elif any(word in context_lower for word in ['romantic', 'love', 'passionate']):
            return ["romantic", "love", "passionate", "intimate", "ballad", "emotional", "drama", "cultural"]
        elif any(word in context_lower for word in ['workout', 'gym', 'running']):
            return ["energetic", "workout", "high_energy", "motivational", "upbeat", "pop", "mainstream", "contemporary"]
        elif any(word in context_lower for word in ['study', 'work', 'focus']):
            return ["calm", "ambient", "study", "focus", "relaxation", "emotional", "cultural", "contemporary"]
        else:
            return ["drama", "romantic", "adventure", "comedy", "mystery", "pop", "mainstream", "cultural"]
    
    def generate_domain_specific_tags(self, domain: str, context: str = "") -> List[str]:
        """Generate domain-specific tags that are proven to work with Qloo"""
        
        # Domain-specific tag mappings based on Qloo's database
        domain_tags = {
            "movie": {
                "romantic": ["romantic", "drama", "love", "passionate", "emotional"],
                "action": ["thriller", "adventure", "action", "exciting", "energetic"],
                "comedy": ["comedy", "funny", "humorous", "lighthearted", "upbeat"],
                "drama": ["drama", "emotional", "serious", "intense", "powerful"],
                "mystery": ["mystery", "thriller", "suspense", "intriguing", "dark"],
                "family": ["family", "heartwarming", "feel-good", "wholesome", "uplifting"]
            },
            "tv_show": {
                "romantic": ["romantic", "drama", "love", "passionate", "emotional"],
                "action": ["thriller", "adventure", "action", "exciting", "energetic"],
                "comedy": ["comedy", "funny", "humorous", "lighthearted", "upbeat"],
                "drama": ["drama", "emotional", "serious", "intense", "powerful"],
                "mystery": ["mystery", "thriller", "suspense", "intriguing", "dark"],
                "reality": ["reality", "drama", "entertaining", "popular", "trending"]
            },
            "podcast": {
                "educational": ["educational", "informative", "knowledge", "learning", "insightful"],
                "comedy": ["comedy", "funny", "humorous", "entertaining", "lighthearted"],
                "news": ["news", "current_events", "politics", "informative", "serious"],
                "business": ["business", "entrepreneurship", "success", "motivational", "professional"],
                "health": ["health", "wellness", "fitness", "lifestyle", "self_improvement"],
                "true_crime": ["true_crime", "mystery", "thriller", "suspense", "dark"]
            },
            "book": {
                "romance": ["romance", "romantic", "love", "passionate", "emotional"],
                "mystery": ["mystery", "thriller", "suspense", "detective", "crime"],
                "fantasy": ["fantasy", "adventure", "magical", "epic", "imaginative"],
                "self_help": ["self_help", "motivational", "inspirational", "personal_growth", "success"],
                "biography": ["biography", "memoir", "inspirational", "real_life", "success"],
                "fiction": ["fiction", "drama", "emotional", "compelling", "engaging"]
            },
            "artist": {
                "pop": ["pop", "mainstream", "popular", "upbeat", "catchy"],
                "rock": ["rock", "energetic", "powerful", "guitar", "band"],
                "hip_hop": ["hip_hop", "rap", "urban", "rhythm", "beats"],
                "electronic": ["electronic", "dance", "techno", "synth", "beats"],
                "jazz": ["jazz", "smooth", "sophisticated", "instrumental", "classy"],
                "classical": ["classical", "orchestral", "sophisticated", "elegant", "timeless"]
            }
        }
        
        # Get tags for the specific domain
        if domain in domain_tags:
            # Select a category based on context or use default
            context_lower = context.lower()
            
            if any(word in context_lower for word in ['romantic', 'love', 'passionate']):
                category = "romantic" if "romantic" in domain_tags[domain] else list(domain_tags[domain].keys())[0]
            elif any(word in context_lower for word in ['action', 'thriller', 'exciting']):
                category = "action" if "action" in domain_tags[domain] else list(domain_tags[domain].keys())[0]
            elif any(word in context_lower for word in ['comedy', 'funny', 'humorous']):
                category = "comedy" if "comedy" in domain_tags[domain] else list(domain_tags[domain].keys())[0]
            elif any(word in context_lower for word in ['mystery', 'suspense', 'dark']):
                category = "mystery" if "mystery" in domain_tags[domain] else list(domain_tags[domain].keys())[0]
            else:
                # Use first available category
                category = list(domain_tags[domain].keys())[0]
            
            return domain_tags[domain][category][:5]  # Return exactly 5 tags
        
        # Fallback to general tags
        return ["drama", "romantic", "adventure", "comedy", "mystery"]
    
    def generate_genre_based_tags(self, artist_genres: List[str], context: str = "", user_country: str = None, domain: str = "artist") -> List[str]:
        """Generate domain-specific tags based on artist genres - limit to 5 most relevant"""
        
        # Extract unique genres from user's artists
        unique_genres = list(set(artist_genres))
        print(f"Extracted genres from user artists: {unique_genres}")
        
        # Simple, proven working tags for each domain
        proven_working_tags = {
            "movie": ["drama", "romantic", "action", "comedy", "family"],
            "tv_show": ["drama", "romantic", "comedy", "family", "reality"],
            "podcast": ["entertainment", "interviews", "music", "cultural", "lifestyle"],
            "book": ["romance", "drama", "fiction", "cultural", "contemporary"],
            "artist": ["pop", "mainstream", "indian", "cultural", "contemporary"]
        }
        
        # Get the proven tags for this domain
        domain_tags = proven_working_tags.get(domain, ["drama", "romantic", "cultural"])
        
        # Add some context-based tags
        context_tags = []
        if "romantic" in context.lower() or "love" in context.lower():
            context_tags.extend(["romantic", "drama"])
        if "action" in context.lower() or "energetic" in context.lower():
            context_tags.extend(["action", "adventure"])
        if "cultural" in context.lower() or "traditional" in context.lower():
            context_tags.extend(["cultural", "traditional"])
        
        # Add cultural tags based on user country
        cultural_tags = []
        if user_country and ("india" in user_country.lower() or "indian" in user_country.lower()):
            cultural_tags.extend(["indian", "south asian"])
        elif user_country and ("american" in user_country.lower() or "usa" in user_country.lower()):
            cultural_tags.extend(["american", "western"])
        
        # Combine all tags and remove duplicates
        all_tags = domain_tags + context_tags + cultural_tags
        unique_tags = list(dict.fromkeys(all_tags))  # Preserve order while removing duplicates
        
        # Limit to 5 tags
        final_tags = unique_tags[:5]
        
        print(f"Domain {domain} - Selected {len(final_tags)} most relevant tags: {final_tags}")
        return final_tags
        proven_working_tags = {
            "movie": ["drama", "romantic", "action", "comedy", "family"],
            "tv_show": ["drama", "romantic", "comedy", "family", "reality"],
            "podcast": ["entertainment", "interviews", "music", "cultural", "lifestyle"],
            "book": ["romance", "drama", "fiction", "cultural", "contemporary"],
            "artist": ["pop", "mainstream", "indian", "cultural", "contemporary"]
        }
        
        # If we have less than 3 tags, use proven working tags
        if len(unique_search_terms) < 3:
            unique_search_terms = proven_working_tags.get(domain, ["drama", "romantic", "cultural", "entertainment", "contemporary"])
        
        print(f"Domain {domain} - Selected 5 most relevant tags: {unique_search_terms}")
        
        return unique_search_terms
    
    def analyze_artist_context(self, artist_name: str, context: str) -> Dict:
        """Analyze artist in context - single focused call"""
        prompt = f"""
        Analyze this artist in the given context and return a JSON object:
        - mood_match: float (0.0 to 1.0) - how well artist matches context mood
        - energy_match: float (0.0 to 1.0) - how well artist matches context energy
        - relevance_score: float (0.0 to 1.0) - overall relevance to context
        
        Artist: {artist_name}
        Context: {context}
        
        Return only the JSON object.
        """
        
        try:
            response = self._call_gemini(prompt)
            if response:
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = response[json_start:json_end]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            print(f"Artist analysis error: {e}")
        
        # Default fallback
        return {
            "mood_match": 0.5,
            "energy_match": 0.5,
            "relevance_score": 0.5
        }
    
    def generate_cultural_context(self, user_country: str, location: str = None) -> Dict:
        """Generate cultural context for recommendations"""
        try:
            # Simple cultural context based on country and location
            cultural_context = {
                "region": "global",
                "language_preference": "any",
                "cultural_elements": [],
                "popular_genres": []
            }
            
            if user_country == "IN":
                cultural_context.update({
                    "region": "south_asia",
                    "language_preference": "hindi",
                    "cultural_elements": ["bollywood", "indian", "desi", "hindi", "punjabi", "bhangra", "cultural", "romantic", "drama"],
                    "popular_genres": ["bollywood", "hindi_pop", "punjabi", "bhangra", "romantic", "drama"]
                })
            elif user_country in ["US", "CA", "GB", "AU"]:
                cultural_context.update({
                    "region": "western",
                    "language_preference": "english",
                    "cultural_elements": ["western", "pop", "rock", "hip_hop", "electronic", "cultural", "romantic", "drama"],
                    "popular_genres": ["pop", "rock", "hip_hop", "electronic", "romantic", "drama"]
                })
            elif user_country in ["JP", "KR"]:
                cultural_context.update({
                    "region": "east_asia",
                    "language_preference": "local",
                    "cultural_elements": ["k_pop", "j_pop", "anime", "cultural", "romantic", "drama"],
                    "popular_genres": ["k_pop", "j_pop", "anime", "j_rock", "romantic", "drama"]
                })
            
            if location:
                cultural_context["location"] = location
            else:
                # Add default location based on country for better cultural context
                if user_country == "IN":
                    cultural_context["location"] = "Mumbai"
                elif user_country == "US":
                    cultural_context["location"] = "New York"
                elif user_country == "GB":
                    cultural_context["location"] = "London"
                elif user_country == "CA":
                    cultural_context["location"] = "Toronto"
                elif user_country == "AU":
                    cultural_context["location"] = "Sydney"
                elif user_country == "JP":
                    cultural_context["location"] = "Tokyo"
                elif user_country == "KR":
                    cultural_context["location"] = "Seoul"
            
            return cultural_context
            
        except Exception as e:
            print(f"Error generating cultural context: {e}")
            return {
                "region": "global",
                "language_preference": "any",
                "cultural_elements": [],
                "popular_genres": []
            }
    
    def enhance_context_detection(self, user_context: str, user_country: str) -> Dict:
        """Enhanced context detection with mood and language preference"""
        try:
            # Analyze context for mood and activity
            context_analysis = self.analyze_context_fast(user_context)
            
            # Determine language preference based on country
            language_preference = {"primary_language": "any"}
            if user_country == "IN":
                language_preference = {"primary_language": "hindi"}
            elif user_country in ["US", "CA", "GB", "AU"]:
                language_preference = {"primary_language": "english"}
            
            # Determine context type
            context_type = "general"
            context_lower = user_context.lower()
            
            if any(word in context_lower for word in ['sad', 'melancholy', 'crying', 'heartbreak']):
                context_type = "sad"
            elif any(word in context_lower for word in ['happy', 'joy', 'excited', 'party']):
                context_type = "upbeat"
            elif any(word in context_lower for word in ['romantic', 'love', 'passionate']):
                context_type = "romantic"
            elif any(word in context_lower for word in ['workout', 'gym', 'running', 'exercise']):
                context_type = "energetic"
            elif any(word in context_lower for word in ['study', 'work', 'focus', 'concentration']):
                context_type = "calm"
            
            return {
                "context_type": context_type,
                "language_preference": language_preference,
                "mood_preference": {
                    "primary_mood": context_analysis.get("primary_mood", "neutral"),
                    "activity_type": context_analysis.get("activity_type", "general"),
                    "energy_level": context_analysis.get("energy_level", "medium")
                }
            }
            
        except Exception as e:
            print(f"Error in enhanced context detection: {e}")
            return {
                "context_type": "general",
                "language_preference": {"primary_language": "any"},
                "mood_preference": {
                    "primary_mood": "neutral",
                    "activity_type": "general",
                    "energy_level": "medium"
                }
            }
    
    def generate_enhanced_tags(self, user_context: str, user_country: str, location: str = None) -> List[str]:
        """Generate enhanced tags using Gemini with cultural context"""
        try:
            # Create enhanced prompt with cultural context
            cultural_context = self.generate_cultural_context(user_country, location)
            
            prompt = f"""
            Based on this user context: "{user_context}"
            And cultural context: {cultural_context}
            
            Generate 8-12 music-related tags that would be useful for finding similar artists and tracks.
            Consider:
            1. The user's mood and activity
            2. Cultural preferences for {user_country}
            3. Popular music styles in their region
            4. Emotional and thematic elements
            
            Return only a comma-separated list of tags, no explanations.
            Examples: upbeat, energetic, romantic, sad, bollywood, pop, rock, electronic, etc.
            """
            
            response = self._call_gemini(prompt)
            if response:
                # Parse tags from response
                tags = [tag.strip() for tag in response.split(",") if tag.strip()]
                return tags[:12]  # Limit to 12 tags
            
        except Exception as e:
            print(f"Error generating enhanced tags: {e}")
        
        # Fallback to simple tags
        return self.generate_optimized_tags(user_context, user_country)
    
    def generate_music_based_cross_domain_tags(self, user_tags: List[str], artists: List[str], user_context: str, user_country: str, domain: str) -> List[str]:
        """Generate cross-domain tags based on music recommendation data"""
        try:
            # Create enhanced prompt for cross-domain recommendations
            cultural_context = self.generate_cultural_context(user_country)
            
            prompt = f"""
            Based on this music recommendation data:
            - User context: "{user_context}"
            - User tags: {', '.join(user_tags) if user_tags else 'none'}
            - Similar artists: {', '.join(artists) if artists else 'none'}
            - Target domain: {domain}
            - User country: {user_country}
            - Cultural context: {cultural_context}
            
            Generate 6-10 tags for {domain} recommendations that match the user's music taste.
            Consider:
            1. The emotional and thematic elements from the music
            2. Cultural preferences for {user_country}
            3. Similarities between music and {domain} content
            4. Popular {domain} styles that match the music vibe
            
            Return only a comma-separated list of tags, no explanations.
            Examples for {domain}: action, drama, comedy, thriller, romance, documentary, etc.
            """
            
            response = self._call_gemini(prompt)
            if response:
                # Parse tags from response
                tags = [tag.strip() for tag in response.split(",") if tag.strip()]
                return tags[:10]  # Limit to 10 tags
            
        except Exception as e:
            print(f"Error generating music-based cross-domain tags: {e}")
        
        # Fallback to domain-specific tags
        return self.generate_domain_specific_tags(domain, user_context)
    
    def filter_and_rank_tags_for_music(self, tags: List[str], user_context: str, user_country: str, location: str = None) -> List[str]:
        """Filter and rank tags for music recommendations"""
        try:
            if not tags:
                return []
            
            # Create prompt for tag filtering
            prompt = f"""
            From this list of tags: {', '.join(tags)}
            
            Select the 5 most relevant tags for music recommendations based on:
            - User context: "{user_context}"
            - User country: {user_country}
            - Location: {location if location else 'Global'}
            
            Consider:
            1. Music genre relevance
            2. Cultural appropriateness
            3. Emotional context
            4. Popularity in the region
            
            Return only the 5 selected tags as a comma-separated list, no explanations.
            """
            
            response = self._call_gemini(prompt)
            if response:
                # Parse filtered tags
                filtered_tags = [tag.strip() for tag in response.split(",") if tag.strip()]
                return filtered_tags[:5]  # Ensure max 5 tags
            
        except Exception as e:
            print(f"Error filtering tags: {e}")
        
        # Fallback: return first 5 tags or generate new ones
        if len(tags) >= 5:
            return tags[:5]
        else:
            return self.generate_optimized_tags(user_context, user_country)
    
    def ai_sort_by_relevance(self, entities: List[Dict], user_context: str, user_artists: List[Dict], 
                           user_genres: List[List[str]], context_type: str, user_country: str, 
                           location: str = None, user_preferences: Dict = None) -> List[Dict]:
        """AI-powered sorting of entities by relevance"""
        try:
            if not entities:
                return []
            
            # Prepare entity data for sorting
            entity_data = []
            for i, entity in enumerate(entities[:25]):  # Limit to 25 for speed
                entity_info = {
                    "index": i,
                    "name": entity.get("name", ""),
                    "artist": entity.get("artist", ""),
                    "genre": entity.get("primary_genre", "unknown"),
                    "context_score": entity.get("context_score", 0.5),
                    "personalization_score": entity.get("personalization_score", 0)
                }
                entity_data.append(entity_info)
            
            # Create sorting prompt
            user_artist_names = []
            for artist in user_artists[:5]:
                if isinstance(artist, dict):
                    user_artist_names.append(artist.get("name", ""))
                elif isinstance(artist, str):
                    user_artist_names.append(artist)
                else:
                    user_artist_names.append(str(artist))
            
            user_genre_list = []
            for genres in user_genres:
                if isinstance(genres, list):
                    user_genre_list.extend(genres)
                elif isinstance(genres, str):
                    user_genre_list.append(genres)
            user_genre_list = list(set(user_genre_list))[:10]  # Top 10 unique genres
            
            prompt = f"""
            Sort these music tracks by relevance to the user:
            
            User context: "{user_context}"
            User's favorite artists: {', '.join(user_artist_names)}
            User's favorite genres: {', '.join(user_genre_list)}
            Context type: {context_type}
            User country: {user_country}
            
            Tracks to sort:
            {json.dumps(entity_data, indent=2)}
            
            Return only the sorted indices as a comma-separated list (e.g., "3,1,4,2,0").
            Consider:
            1. Genre similarity to user preferences
            2. Artist similarity to user favorites
            3. Context relevance
            4. Cultural appropriateness
            
            Return only the indices, no explanations.
            """
            
            response = self._call_gemini(prompt)
            if response:
                # Parse sorted indices
                try:
                    indices = [int(idx.strip()) for idx in response.split(",") if idx.strip().isdigit()]
                    # Create sorted list
                    sorted_entities = []
                    for idx in indices:
                        if 0 <= idx < len(entities):
                            sorted_entities.append(entities[idx])
                    
                    # Add any remaining entities that weren't in the sorted list
                    for i, entity in enumerate(entities):
                        if entity not in sorted_entities:
                            sorted_entities.append(entity)
                    
                    return sorted_entities
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing AI sorting response: {e}")
            
        except Exception as e:
            print(f"Error in AI sorting: {e}")
        
        # Fallback: return original order
        return entities
    
    def generate_music_tags_from_prompt_and_genres(self, user_prompt: str, user_genres: List[str], user_country: str = None) -> List[str]:
        """Generate music tags based on user prompt + user's artist genres"""
        
        # Extract unique genres from user's listening history
        unique_genres = list(set(user_genres))
        print(f"User's listening genres: {unique_genres[:10]}")  # Show first 10
        
        # Analyze user prompt for mood/context
        prompt_lower = user_prompt.lower()
        
        # Mood-based tag mapping
        mood_tags = []
        if any(word in prompt_lower for word in ['happy', 'joy', 'excited', 'energetic', 'upbeat']):
            mood_tags.extend(['upbeat', 'energetic', 'happy'])
        elif any(word in prompt_lower for word in ['sad', 'melancholy', 'emotional', 'heartbreak']):
            mood_tags.extend(['emotional', 'melancholy', 'romantic'])
        elif any(word in prompt_lower for word in ['romantic', 'love', 'passionate']):
            mood_tags.extend(['romantic', 'love', 'passionate'])
        elif any(word in prompt_lower for word in ['workout', 'gym', 'exercise', 'motivation']):
            mood_tags.extend(['energetic', 'motivational', 'upbeat'])
        elif any(word in prompt_lower for word in ['study', 'work', 'focus', 'concentration']):
            mood_tags.extend(['ambient', 'instrumental', 'calm'])
        elif any(word in prompt_lower for word in ['party', 'dance', 'celebration']):
            mood_tags.extend(['dance', 'party', 'upbeat'])
        elif any(word in prompt_lower for word in ['chill', 'relax', 'calm', 'peaceful']):
            mood_tags.extend(['chill', 'relaxing', 'ambient'])
        else:
            # Default mood tags based on common scenarios
            mood_tags.extend(['upbeat', 'mainstream'])
        
        # Genre-based tags from user's listening history
        genre_tags = []
        for genre in unique_genres[:5]:  # Use top 5 most common genres
            if genre in ['pop', 'mainstream', 'indian pop', 'hindi pop']:
                genre_tags.extend(['pop', 'mainstream'])
            elif genre in ['bollywood', 'film music']:
                genre_tags.extend(['bollywood', 'film music'])
            elif genre in ['indie', 'indian indie', 'hindi indie']:
                genre_tags.extend(['indie', 'alternative'])
            elif genre in ['sufi', 'qawwali', 'ghazal']:
                genre_tags.extend(['sufi', 'spiritual', 'traditional'])
            elif genre in ['hip hop', 'rap', 'hindi hip hop']:
                genre_tags.extend(['hip hop', 'rap', 'urban'])
            elif genre in ['electronic', 'edm', 'dance']:
                genre_tags.extend(['electronic', 'dance'])
            elif genre in ['rock', 'alternative']:
                genre_tags.extend(['rock', 'alternative'])
            else:
                # For unknown genres, use the genre name itself
                genre_tags.append(genre)
        
        # Cultural tags based on user country
        cultural_tags = []
        if user_country and ("india" in user_country.lower() or "indian" in user_country.lower()):
            cultural_tags.extend(['indian', 'south asian'])
        elif user_country and ("american" in user_country.lower() or "usa" in user_country.lower()):
            cultural_tags.extend(['american', 'western'])
        
        # Combine all tags and prioritize user's genres
        all_tags = genre_tags + mood_tags + cultural_tags  # User genres first for priority
        unique_tags = list(dict.fromkeys(all_tags))  # Remove duplicates while preserving order
        
        # Limit to 5 most relevant tags
        final_tags = unique_tags[:5]
        
        # Fallback to proven working tags if we don't have enough
        if len(final_tags) < 3:
            fallback_tags = ['pop', 'mainstream', 'indian', 'cultural', 'contemporary']
            for tag in fallback_tags:
                if tag not in final_tags and len(final_tags) < 5:
                    final_tags.append(tag)
        
        print(f"Generated music tags: {final_tags}")
        return final_tags
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Make Gemini API call with rate limiting"""
        time.sleep(0.1)  # 100ms delay for rate limiting
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 500
            }
        }
        
        try:
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except Exception as e:
            print(f"Gemini API call error: {e}")
        
        return None
    
    def _parse_context_fallback(self, response: str) -> Dict:
        """Fallback parsing for context analysis"""
        response_lower = response.lower()
        
        # Simple keyword matching
        mood_keywords = {
            "happy": "happy", "joy": "happy", "excited": "energetic",
            "sad": "sad", "melancholy": "sad", "energetic": "energetic",
            "calm": "calm", "relaxed": "calm", "romantic": "romantic"
        }
        
        activity_keywords = {
            "workout": "workout", "exercise": "workout", "gym": "workout",
            "study": "study", "work": "work", "party": "party",
            "relax": "relaxation", "sleep": "relaxation"
        }
        
        energy_keywords = {
            "high": "high", "energetic": "high", "fast": "high",
            "medium": "medium", "moderate": "medium",
            "low": "low", "slow": "low", "calm": "low"
        }
        
        # Extract values
        primary_mood = "neutral"
        for keyword, mood in mood_keywords.items():
            if keyword in response_lower:
                primary_mood = mood
                break
        
        activity_type = "general"
        for keyword, activity in activity_keywords.items():
            if keyword in response_lower:
                activity_type = activity
                break
        
        energy_level = "medium"
        for keyword, energy in energy_keywords.items():
            if keyword in response_lower:
                energy_level = energy
                break
        
        return {
            "primary_mood": primary_mood,
            "activity_type": activity_type,
            "energy_level": energy_level,
            "confidence": 0.6
        }
    
    def _parse_tags_fallback(self, response: str) -> List[str]:
        """Fallback parsing for tag generation"""
        response_lower = response.lower()
        
        # Extract potential tags
        potential_tags = []
        
        # Common music tags
        music_tags = [
            "pop", "rock", "hip hop", "electronic", "jazz", "classical",
            "energetic", "calm", "romantic", "sad", "happy", "upbeat",
            "workout", "study", "party", "relaxation", "motivational"
        ]
        
        for tag in music_tags:
            if tag in response_lower:
                potential_tags.append(tag)
        
        # Return up to 5 tags
        return potential_tags[:5] if potential_tags else ["pop", "energetic", "upbeat", "mainstream", "popular"] 