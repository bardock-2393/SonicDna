# 🎵 SoniqueDNA - Your Personal Music Curator

> **A Next-Generation AI-Powered Music Discovery Platform**  
> *Built for the Qloo Global Hackathon - Exploring the Intersection of LLMs and Cultural Intelligence*

[![React](https://img.shields.io/badge/React-18.3.1-blue.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5.3-blue.svg)](https://www.typescriptlang.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![Qloo API](https://img.shields.io/badge/Qloo%20API-Taste%20AI-orange.svg)](https://qloo.com/)
[![Gemini AI](https://img.shields.io/badge/Gemini%20AI-2.0%20Flash-purple.svg)](https://ai.google.dev/gemini-api)

## 🌟 Project Overview

SoniqueDNA is an innovative music discovery platform that leverages the power of **Large Language Models (LLMs)** and **Qloo's Taste AI™ API** to create deeply personalized music experiences. By combining cultural intelligence with AI-driven context analysis, we deliver recommendations that understand not just what you listen to, but why you listen to it.

### 🎯 Key Features

- **🎭 Cultural Intelligence Integration**: Seamlessly combines Qloo's cross-domain cultural insights with music preferences
- **🧠 AI-Powered Context Analysis**: Uses Google's Gemini 2.0 Flash to understand user context and mood
- **🌍 Cross-Domain Recommendations**: Discovers music through cultural affinities across movies, TV shows, books, and more
- **🎨 Personalized User Experience**: Beautiful, responsive UI with real-time chat interface
- **📊 Taste Analytics**: Visual representation of user music preferences and cultural affinities
- **🔗 Spotify Integration**: Direct playlist creation and user profile analysis
- **⚡ Real-time Processing**: Live progress tracking and intelligent caching system

## 🏗️ Architecture

![SoniqueDNA System Architecture](https://i.ibb.co/Y7TZd9xX/diagram-export-8-1-2025-1-31-50-PM.png)

*System Architecture showing the integration of User Interface, Spotify Integration, AI-Powered Analysis, and Cultural Intelligence modules*

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   External      │
│   (React/TS)    │◄──►│   (Flask/Python)│◄──►│     APIs        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                      │                      │
├─ Music Dashboard     ├─ Spotify Service     ├─ Spotify Web API
├─ Chat Interface      ├─ Qloo Service        ├─ Qloo Taste AI™
├─ User Analytics      ├─ Gemini Service      ├─ Gemini 2.0 Flash
├─ Playlist Manager    ├─ Database Service    └─ SQLite Database
└─ Real-time Updates   └─ Cache Management
```

## 🚀 Technology Stack

### Frontend
- **React 18.3.1** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Shadcn/ui** for beautiful components
- **React Query** for state management
- **React Router** for navigation

### Backend
- **Flask 2.3.3** with Python
- **Flask-CORS** for cross-origin requests
- **SQLite** for data persistence
- **Requests** for API communication

### AI & APIs
- **Google Gemini 2.0 Flash** for context analysis and intelligent tag generation
- **Qloo Taste AI™ API** for cultural intelligence and cross-domain recommendations
- **Spotify Web API** for music data and playlist management

## 🎯 How It Works

### 1. **Context Understanding**
```typescript
// User provides context through natural language
"I'm feeling nostalgic and want to discover new artists like the ones I loved in college"
```

### 2. **AI Analysis**
```python
# Gemini analyzes context and generates cultural tags
{
  "primary_mood": "nostalgic",
  "activity_type": "reflection", 
  "energy_level": "medium",
  "cultural_tags": ["nostalgic", "emotional", "indie", "alternative"]
}
```

### 3. **Cultural Intelligence**
```python
# Qloo API finds cross-domain affinities
qloo_service.get_cross_domain_recommendations(
    tag_ids=["nostalgic", "emotional"],
    domain="artist",
    limit=15
)
```

### 4. **Personalized Results**
- Music recommendations based on cultural affinities
- Cross-domain insights (movies, books, TV shows that match your taste)
- Spotify playlist creation with discovered artists

## 🛠️ Installation & Setup

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+
- Spotify Developer Account
- Qloo API Key
- Google Gemini API Key

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sonique-dna.git
cd sonique-dna
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Copy environment template
cp env.example .env

# Add your API keys
QLOO_API_KEY=your_qloo_api_key
GEMINI_API_KEY=your_gemini_api_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

### 4. Frontend Setup
```bash
cd ../frontend
npm install
```

### 5. Start Development Servers
```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

## 📱 Usage

### Getting Started
1. **Connect Spotify**: Click "Connect Spotify" to authenticate with your account
   - 📹 **[Watch Demo: How to Connect Spotify](https://youtu.be/0leb4gCI39c)**
2. **Describe Your Mood**: Use natural language to describe what you're looking for
3. **Discover Music**: Get personalized recommendations based on cultural intelligence
4. **Create Playlists**: Save discoveries directly to your Spotify account
5. **Explore Insights**: View your taste analytics and cultural affinities

### Example Interactions
```
User: "I'm in the mood for something energetic for my workout"
AI: Analyzes context → Generates tags → Finds cultural affinities → Recommends artists

User: "Show me artists similar to the ones I loved in college"
AI: Analyzes nostalgia context → Cross-references with cultural data → Discovers new artists
```

## 🎨 Key Components

### Frontend Components
- **MusicDashboard**: Main application interface
- **ChatInput**: Natural language input with intelligent suggestions
- **UserTasteGraph**: Visual analytics of user preferences
- **CrossDomainRecommendations**: Cultural affinity insights
- **PlaylistResponse**: Spotify playlist creation and management

### Backend Services
- **SpotifyService**: Handles Spotify API integration
- **QlooService**: Manages cultural intelligence queries
- **GeminiService**: AI-powered context analysis
- **DatabaseService**: User data persistence

## 🔧 API Integration Details

### Qloo Taste AI™ Integration
```python
class QlooService:
    def get_cross_domain_recommendations(self, tag_ids, domain, limit=10):
        """Get recommendations across cultural domains"""
        
    def get_cultural_insights(self, location, domain="music"):
        """Extract cultural context from location data"""
        
    def get_enhanced_recommendations(self, tag_ids, user_artists, user_tracks):
        """Combine user signals with cultural intelligence"""
```

### Gemini AI Integration
```python
class GeminiService:
    def analyze_context_fast(self, user_context):
        """Analyze user context for mood and activity"""
        
    def generate_optimized_tags(self, context, user_country):
        """Generate cultural tags for Qloo API"""
        
    def ai_sort_by_relevance(self, entities, user_context):
        """Intelligently rank recommendations"""
```

## 📊 Features in Detail

### Cultural Intelligence Engine
- **Cross-Domain Affinities**: Discover music through cultural connections
- **Geographic Context**: Location-based cultural insights
- **Temporal Analysis**: Seasonal and trend-based recommendations
- **Privacy-First**: No personal data required for cultural insights

### AI-Powered Context Analysis
- **Natural Language Processing**: Understand user intent and mood
- **Cultural Tag Generation**: Intelligent tag creation for Qloo API
- **Relevance Ranking**: AI-driven recommendation sorting
- **Contextual Adaptation**: Dynamic response to user feedback

### User Experience
- **Real-time Chat Interface**: Natural conversation with AI
- **Progress Tracking**: Live updates during recommendation generation
- **Visual Analytics**: Interactive taste graphs and insights
- **Seamless Spotify Integration**: Direct playlist creation

## 🎯 Hackathon Alignment

### Qloo API Integration
✅ **Cultural Intelligence**: Deep integration with Qloo's Taste AI™ for cross-domain insights  
✅ **Privacy-First Approach**: Leverages Qloo's privacy-first cultural data  
✅ **Cross-Domain Affinities**: Connects music with movies, books, TV shows, and more  
✅ **Geographic Context**: Location-based cultural recommendations  

### LLM Integration
✅ **Gemini 2.0 Flash**: Advanced context analysis and natural language understanding  
✅ **Intelligent Tag Generation**: AI-powered cultural tag creation  
✅ **Contextual Recommendations**: Dynamic response to user mood and activity  
✅ **Natural Language Interface**: Conversational music discovery experience  

### Technical Excellence
✅ **Modern Tech Stack**: React, TypeScript, Flask, Python  
✅ **Scalable Architecture**: Modular service-based design  
✅ **Real-time Processing**: Live progress tracking and updates  
✅ **Production Ready**: Error handling, caching, and optimization  

## 🏆 Potential Impact

### For Users
- **Deeper Discovery**: Find music through cultural connections, not just genres
- **Personalized Experience**: AI understands context and mood
- **Cultural Exploration**: Discover new artists through cross-domain affinities
- **Privacy Protection**: No personal data required for cultural insights

### For the Industry
- **New Discovery Paradigm**: Cultural intelligence over traditional algorithms
- **Cross-Domain Integration**: Music discovery through lifestyle and interests
- **AI-Enhanced UX**: Natural language interface for music discovery
- **Privacy-First Innovation**: Cultural insights without personal data

## 🚀 Future Enhancements

- **Multi-Platform Support**: Mobile apps and smart speaker integration
- **Social Features**: Share cultural discoveries with friends
- **Advanced Analytics**: Deep cultural trend analysis
- **API Marketplace**: Allow third-party integrations
- **Cultural Events**: Real-time recommendations based on local events

## 📝 License

This project is created for the Qloo Global Hackathon and is licensed under the MIT License.

## 🤝 Contributing

This project was developed for the Qloo Global Hackathon. For questions or collaboration opportunities, please reach out to the development team.

---

**Built with ❤️ for the Qloo Global Hackathon**  
*Exploring the intersection of LLMs and cultural intelligence* 