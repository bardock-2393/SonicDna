import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class DatabaseService:
    def __init__(self, db_path: str = "music_recommendations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                spotify_token TEXT,
                session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_end TIMESTAMP,
                user_country TEXT,
                user_context TEXT,
                UNIQUE(user_id, session_start)
            )
        ''')
        
        # Recommendation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id INTEGER,
                recommendation_type TEXT NOT NULL,
                user_context TEXT,
                generated_tags TEXT,
                qloo_artists TEXT,
                playlist_data TEXT,
                response_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES user_sessions(id)
            )
        ''')
        
        # Artist tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artist_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                recommendation_count INTEGER DEFAULT 1,
                is_new_artist BOOLEAN DEFAULT TRUE,
                genre TEXT,
                popularity_score REAL DEFAULT 0.0,
                UNIQUE(user_id, artist_name)
            )
        ''')
        
        # Cached recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cached_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                user_context TEXT,
                user_country TEXT,
                user_artists TEXT,
                recommendation_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 1
            )
        ''')
        
        # User taste analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_taste_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                genre TEXT NOT NULL,
                artist_count INTEGER DEFAULT 0,
                track_count INTEGER DEFAULT 0,
                total_playtime REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, genre)
            )
        ''')
        
        # User mood preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_mood_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                mood TEXT NOT NULL,
                preference_score REAL DEFAULT 0.0,
                context_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, mood)
            )
        ''')
        
        # New artists discovered table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS new_artists_discovered (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, artist_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user_session(self, user_id: str, spotify_token: str, user_country: str = None, user_context: str = None) -> int:
        """Create a new user session and return session ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, spotify_token, user_country, user_context)
            VALUES (?, ?, ?, ?)
        ''', (user_id, spotify_token, user_country, user_context))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_user_session(self, session_id: int):
        """End a user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE user_sessions 
            SET session_end = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
    
    def store_recommendation_history(self, user_id: str, session_id: int, recommendation_type: str, 
                                   user_context: str, generated_tags: List[str], qloo_artists: List[Dict], 
                                   playlist_data: Dict, response_time: float):
        """Store a recommendation in history"""
        try:
            print(f"[DATABASE] Storing recommendation history for user {user_id}, session {session_id}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO recommendation_history 
                (user_id, session_id, recommendation_type, user_context, generated_tags, qloo_artists, playlist_data, response_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, session_id, recommendation_type, user_context,
                json.dumps(generated_tags), json.dumps(qloo_artists), json.dumps(playlist_data), response_time
            ))
            
            conn.commit()
            conn.close()
            print(f"[DATABASE] Successfully stored recommendation history")
            
            # Update analytics based on the recommendation
            self._update_analytics_from_recommendation(user_id, generated_tags, qloo_artists, user_context)
            
        except Exception as e:
            print(f"[DATABASE] Error storing recommendation history: {e}")
            raise
    
    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get user's recommendation history"""
        try:
            print(f"[DATABASE] Getting history for user {user_id}, limit {limit}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, recommendation_type, user_context, generated_tags, qloo_artists, 
                       playlist_data, response_time, created_at
                FROM recommendation_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'recommendation_type': row[1],
                    'user_context': row[2],
                    'generated_tags': json.loads(row[3]) if row[3] else [],
                    'qloo_artists': json.loads(row[4]) if row[4] else [],
                    'playlist_data': json.loads(row[5]) if row[5] else {},
                    'response_time': row[6],
                    'created_at': row[7]
                })
            
            conn.close()
            print(f"[DATABASE] Found {len(results)} history records for user {user_id}")
            return results
        except Exception as e:
            print(f"[DATABASE] Error getting user history: {e}")
            return []
    
    def get_history_item(self, history_id: int) -> Optional[Dict]:
        """Get a specific history item by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, user_id, recommendation_type, user_context, generated_tags, 
                   qloo_artists, playlist_data, response_time, created_at
            FROM recommendation_history 
            WHERE id = ?
        ''', (history_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'user_id': row[1],
                'recommendation_type': row[2],
                'user_context': row[3],
                'generated_tags': json.loads(row[4]) if row[4] else [],
                'qloo_artists': json.loads(row[5]) if row[5] else [],
                'playlist_data': json.loads(row[6]) if row[6] else {},
                'response_time': row[7],
                'created_at': row[8]
            }
        return None
    
    def track_new_artists(self, user_id: str, artists: List[Dict]):
        """Track new artists for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for artist in artists:
            artist_name = artist.get('name', '')
            genre = artist.get('genre', '')
            popularity = artist.get('popularity', 0.0)
            
            if not artist_name:
                continue
            
            # Check if artist already exists
            cursor.execute('''
                SELECT id, recommendation_count, is_new_artist 
                FROM artist_tracking 
                WHERE user_id = ? AND artist_name = ?
            ''', (user_id, artist_name))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing artist
                cursor.execute('''
                    UPDATE artist_tracking 
                    SET last_seen = CURRENT_TIMESTAMP, 
                        recommendation_count = recommendation_count + 1,
                        genre = COALESCE(?, genre),
                        popularity_score = ?
                    WHERE id = ?
                ''', (genre, popularity, existing[0]))
            else:
                # Insert new artist
                cursor.execute('''
                    INSERT INTO artist_tracking 
                    (user_id, artist_name, genre, popularity_score)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, artist_name, genre, popularity))
                
                # Also insert into new_artists_discovered table
                cursor.execute('''
                    INSERT OR IGNORE INTO new_artists_discovered 
                    (user_id, artist_name)
                    VALUES (?, ?)
                ''', (user_id, artist_name))
        
        conn.commit()
        conn.close()
    
    def get_new_artists(self, user_id: str, days: int = 7) -> List[Dict]:
        """Get recently discovered new artists for a user within the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate the date threshold
        from datetime import datetime, timedelta
        threshold_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT artist_name, first_seen, last_seen, recommendation_count, 
                   is_new_artist, genre, popularity_score
            FROM artist_tracking 
            WHERE user_id = ? AND is_new_artist = TRUE AND first_seen >= ?
            ORDER BY first_seen DESC 
            LIMIT 20
        ''', (user_id, threshold_date))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'artist_name': row[0],
                'first_seen': row[1],
                'last_seen': row[2],
                'recommendation_count': row[3],
                'is_new_artist': row[4],
                'genre': row[5],
                'popularity_score': row[6]
            })
        
        conn.close()
        return results
    
    def is_new_artist(self, user_id: str, artist_name: str) -> bool:
        """Check if an artist is new to the user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT is_new_artist FROM artist_tracking 
            WHERE user_id = ? AND artist_name = ?
        ''', (user_id, artist_name))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is None or result[0]
    
    def mark_artist_as_new(self, user_id: str, artist_name: str):
        """Mark an artist as new to the user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE artist_tracking 
            SET is_new_artist = FALSE 
            WHERE user_id = ? AND artist_name = ?
        ''', (user_id, artist_name))
        
        conn.commit()
        conn.close()
    
    def store_cached_recommendation(self, cache_key: str, user_context: str, user_country: str, 
                                  user_artists: List[str], recommendation_data: Dict, 
                                  expires_in_hours: int = 24):
        """Store a cached recommendation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        cursor.execute('''
            INSERT OR REPLACE INTO cached_recommendations 
            (cache_key, user_context, user_country, user_artists, recommendation_data, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            cache_key, user_context, user_country, 
            json.dumps(user_artists), json.dumps(recommendation_data), expires_at
        ))
        
        conn.commit()
        conn.close()
    
    def get_cached_recommendation(self, cache_key: str) -> Optional[Dict]:
        """Get a cached recommendation if it exists and is not expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_context, user_country, user_artists, recommendation_data, hit_count
            FROM cached_recommendations 
            WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
        ''', (cache_key,))
        
        result = cursor.fetchone()
        
        if result:
            # Update hit count
            cursor.execute('''
                UPDATE cached_recommendations 
                SET hit_count = hit_count + 1 
                WHERE cache_key = ?
            ''', (cache_key,))
            
            conn.commit()
            conn.close()
            
            return {
                'user_context': result[0],
                'user_country': result[1],
                'user_artists': json.loads(result[2]) if result[2] else [],
                'recommendation_data': json.loads(result[3]) if result[3] else {},
                'hit_count': result[4]
            }
        
        conn.close()
        return None
    
    def clear_user_cache(self, user_id: str):
        """Clear cached recommendations for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM cached_recommendations 
            WHERE cache_key LIKE ?
        ''', (f'%{user_id}%',))
        
        conn.commit()
        conn.close()
    
    def update_user_taste_analytics(self, user_id: str, genre: str, artist_count: int = 1, 
                                  track_count: int = 0, playtime: float = 0.0):
        """Update user taste analytics for a genre"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if genre already exists for this user
        cursor.execute('''
            SELECT artist_count, track_count, total_playtime
            FROM user_taste_analytics 
            WHERE user_id = ? AND genre = ?
        ''', (user_id, genre))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record by adding new values
            new_artist_count = existing[0] + artist_count
            new_track_count = existing[1] + track_count
            new_playtime = existing[2] + playtime
            
            cursor.execute('''
                UPDATE user_taste_analytics 
                SET artist_count = ?, track_count = ?, total_playtime = ?, last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND genre = ?
            ''', (new_artist_count, new_track_count, new_playtime, user_id, genre))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO user_taste_analytics 
                (user_id, genre, artist_count, track_count, total_playtime, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, genre, artist_count, track_count, playtime))
        
        conn.commit()
        conn.close()
    
    def get_user_taste_analytics(self, user_id: str) -> Dict:
        """Get user taste analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get genre analytics
        cursor.execute('''
            SELECT genre, artist_count, track_count, total_playtime, last_updated
            FROM user_taste_analytics 
            WHERE user_id = ?
            ORDER BY artist_count DESC
        ''', (user_id,))
        
        genres = []
        for row in cursor.fetchall():
            genres.append({
                'genre': row[0],
                'artist_count': row[1],
                'track_count': row[2],
                'total_playtime': row[3],
                'last_updated': row[4]
            })
        
        # Get mood preferences
        cursor.execute('''
            SELECT mood, preference_score, context_count, last_updated
            FROM user_mood_preferences 
            WHERE user_id = ?
            ORDER BY preference_score DESC
        ''', (user_id,))
        
        moods = []
        for row in cursor.fetchall():
            moods.append({
                'mood': row[0],
                'preference_score': row[1],
                'context_count': row[2],
                'last_updated': row[3]
            })
        
        # Get timeline data (new artists over time)
        cursor.execute('''
            SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as new_artists
            FROM new_artists_discovered 
            WHERE user_id = ?
            GROUP BY strftime('%Y-%m', created_at)
            ORDER BY month DESC
            LIMIT 12
        ''', (user_id,))
        
        timeline = []
        for row in cursor.fetchall():
            timeline.append({
                'month': row[0],
                'new_artists': row[1]
            })
        
        conn.close()
        
        # If no data exists, provide sample data for demonstration
        if not genres and not moods and not timeline:
            import random
            from datetime import datetime, timedelta
            
            # Sample genres
            sample_genres = [
                {'genre': 'pop', 'artist_count': 15, 'track_count': 45, 'total_playtime': 180.5, 'last_updated': datetime.now().isoformat()},
                {'genre': 'rock', 'artist_count': 12, 'track_count': 38, 'total_playtime': 145.2, 'last_updated': datetime.now().isoformat()},
                {'genre': 'electronic', 'artist_count': 8, 'track_count': 25, 'total_playtime': 95.8, 'last_updated': datetime.now().isoformat()},
                {'genre': 'indie', 'artist_count': 10, 'track_count': 32, 'total_playtime': 120.3, 'last_updated': datetime.now().isoformat()},
                {'genre': 'hip-hop', 'artist_count': 6, 'track_count': 18, 'total_playtime': 75.4, 'last_updated': datetime.now().isoformat()}
            ]
            
            # Sample moods
            sample_moods = [
                {'mood': 'energetic', 'preference_score': 0.85, 'context_count': 12, 'last_updated': datetime.now().isoformat()},
                {'mood': 'chill', 'preference_score': 0.72, 'context_count': 8, 'last_updated': datetime.now().isoformat()},
                {'mood': 'happy', 'preference_score': 0.68, 'context_count': 6, 'last_updated': datetime.now().isoformat()},
                {'mood': 'melancholic', 'preference_score': 0.45, 'context_count': 4, 'last_updated': datetime.now().isoformat()},
                {'mood': 'romantic', 'preference_score': 0.38, 'context_count': 3, 'last_updated': datetime.now().isoformat()}
            ]
            
            # Sample timeline (last 6 months)
            sample_timeline = []
            for i in range(6):
                date = datetime.now() - timedelta(days=30*i)
                month = date.strftime('%Y-%m')
                sample_timeline.append({
                    'month': month,
                    'new_artists': random.randint(3, 12)
                })
            sample_timeline.reverse()
            
            return {
                'genres': sample_genres,
                'moods': sample_moods,
                'timeline': sample_timeline
            }
        
        return {
            'genres': genres,
            'moods': moods,
            'timeline': timeline
        }
    
    def populate_sample_analytics(self, user_id: str):
        """Populate sample analytics data for demonstration"""
        try:
            print(f"[DATABASE] Populating sample analytics for user {user_id}")
            
            # Sample genres with realistic data
            sample_genres = [
                ('pop', 15, 45, 180.5),
                ('rock', 12, 38, 145.2),
                ('electronic', 8, 25, 95.8),
                ('indie', 10, 32, 120.3),
                ('hip-hop', 6, 18, 75.4),
                ('jazz', 4, 12, 60.2),
                ('classical', 3, 8, 45.1)
            ]
            
            for genre, artists, tracks, playtime in sample_genres:
                self.update_user_taste_analytics(user_id, genre, artists, tracks, playtime)
            
            # Sample moods
            sample_moods = [
                ('energetic', 8.5, 12),
                ('chill', 7.2, 8),
                ('happy', 6.8, 6),
                ('melancholic', 4.5, 4),
                ('romantic', 3.8, 3)
            ]
            
            for mood, score, count in sample_moods:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_mood_preferences 
                    (user_id, mood, preference_score, context_count, last_updated)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (user_id, mood, score, count))
                conn.commit()
                conn.close()
            
            # Sample timeline data
            import random
            from datetime import datetime, timedelta
            
            for i in range(6):
                date = datetime.now() - timedelta(days=30*i)
                month = date.strftime('%Y-%m')
                artist_count = random.randint(3, 12)
                
                for j in range(artist_count):
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR IGNORE INTO new_artists_discovered 
                        (user_id, artist_name, created_at)
                        VALUES (?, ?, ?)
                    ''', (user_id, f"Sample Artist {j+1}", date))
                    conn.commit()
                    conn.close()
            
            print(f"[DATABASE] Successfully populated sample analytics for user {user_id}")
            
        except Exception as e:
            print(f"[DATABASE] Error populating sample analytics: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_user_analytics(self, user_id: str):
        """Clear all analytics data for a user"""
        try:
            print(f"[DATABASE] Clearing analytics for user {user_id}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM user_taste_analytics WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM user_mood_preferences WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM new_artists_discovered WHERE user_id = ?', (user_id,))
            
            conn.commit()
            conn.close()
            print(f"[DATABASE] Successfully cleared analytics for user {user_id}")
            
        except Exception as e:
            print(f"[DATABASE] Error clearing analytics: {e}")
    
    def _update_analytics_from_recommendation(self, user_id: str, generated_tags: List[str], qloo_artists: List[Dict], user_context: str):
        """Update analytics based on a recommendation"""
        try:
            print(f"[DATABASE] Updating analytics for user {user_id}")
            print(f"[DATABASE] Artists: {len(qloo_artists)}, Tags: {len(generated_tags)}")
            
            # Extract genres from artists and count them
            genre_counts = {}
            for artist in qloo_artists:
                if isinstance(artist, dict):
                    genre = artist.get('genre', '').lower()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Update genre analytics with proper counts
            for genre, count in genre_counts.items():
                if genre:
                    # Estimate track count based on artist count (assuming 3-5 tracks per artist)
                    estimated_tracks = count * 4
                    self.update_user_taste_analytics(user_id, genre, count, estimated_tracks, 0.0)
                    print(f"[DATABASE] Updated genre {genre}: {count} artists, {estimated_tracks} tracks")
            
            # Extract moods from tags and context with better matching
            mood_keywords = {
                'happy': ['happy', 'joy', 'upbeat', 'cheerful', 'positive', 'fun', 'exciting'],
                'sad': ['sad', 'melancholic', 'depressing', 'sorrow', 'emotional', 'heartbreak'],
                'energetic': ['energetic', 'high-energy', 'powerful', 'intense', 'dynamic', 'vibrant'],
                'calm': ['calm', 'peaceful', 'relaxing', 'soothing', 'gentle', 'serene'],
                'romantic': ['romantic', 'love', 'passionate', 'intimate', 'sweet', 'tender'],
                'chill': ['chill', 'laid-back', 'easy-going', 'mellow', 'smooth', 'groovy']
            }
            
            context_lower = user_context.lower() if user_context else ""
            tags_lower = [tag.lower() for tag in generated_tags]
            
            # Check for mood matches
            matched_moods = set()
            for mood, keywords in mood_keywords.items():
                for keyword in keywords:
                    if keyword in context_lower or any(keyword in tag for tag in tags_lower):
                        matched_moods.add(mood)
                        break
            
            # Update mood preferences
            for mood in matched_moods:
                self.update_mood_preferences(user_id, mood, 1.0)
                print(f"[DATABASE] Updated mood {mood}")
                        
        except Exception as e:
            print(f"[DATABASE] Error updating analytics from recommendation: {e}")
            import traceback
            traceback.print_exc()
    
    def update_mood_preferences(self, user_id: str, mood: str, preference_score: float = 1.0):
        """Update user mood preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if mood already exists for this user
        cursor.execute('''
            SELECT preference_score, context_count
            FROM user_mood_preferences 
            WHERE user_id = ? AND mood = ?
        ''', (user_id, mood))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record by accumulating values
            new_preference_score = existing[0] + preference_score
            new_context_count = existing[1] + 1
            
            cursor.execute('''
                UPDATE user_mood_preferences 
                SET preference_score = ?, context_count = ?, last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ? AND mood = ?
            ''', (new_preference_score, new_context_count, user_id, mood))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO user_mood_preferences 
                (user_id, mood, preference_score, context_count, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, mood, preference_score, 1))
        
        conn.commit()
        conn.close()
    
    def get_user_mood_preferences(self, user_id: str) -> List[Dict]:
        """Get user mood preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT mood, preference_score, context_count, last_updated
            FROM user_mood_preferences 
            WHERE user_id = ?
            ORDER BY preference_score DESC
        ''', (user_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'mood': row[0],
                'preference_score': row[1],
                'context_count': row[2],
                'last_updated': row[3]
            })
        
        conn.close()
        return results
    
    def delete_history_item(self, history_id: int, user_id: str) -> bool:
        """Delete a specific history item for a user"""
        try:
            print(f"[DATABASE] Deleting history item {history_id} for user {user_id}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First check if the history item exists and belongs to the user
            cursor.execute('''
                SELECT id FROM recommendation_history 
                WHERE id = ? AND user_id = ?
            ''', (history_id, user_id))
            
            if not cursor.fetchone():
                print(f"[DATABASE] History item {history_id} not found or doesn't belong to user {user_id}")
                conn.close()
                return False
            
            # Delete the history item
            cursor.execute('''
                DELETE FROM recommendation_history 
                WHERE id = ? AND user_id = ?
            ''', (history_id, user_id))
            
            conn.commit()
            conn.close()
            
            print(f"[DATABASE] Successfully deleted history item {history_id}")
            return True
            
        except Exception as e:
            print(f"[DATABASE] Error deleting history item: {e}")
            return False
    
    def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user"""
        try:
            print(f"[DATABASE] Clearing all history for user {user_id}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete all history items for the user
            cursor.execute('''
                DELETE FROM recommendation_history 
                WHERE user_id = ?
            ''', (user_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"[DATABASE] Successfully cleared {deleted_count} history items for user {user_id}")
            return True
            
        except Exception as e:
            print(f"[DATABASE] Error clearing user history: {e}")
            return False 