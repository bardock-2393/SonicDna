"use client";
import React, { useState, useEffect } from "react";
import { StickyScroll } from "@/components/ui/sticky-scroll-reveal";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Plus, Music, CheckCircle, AlertCircle, Sparkles } from "lucide-react";
import { ComicText } from "./magicui/comic-text";
import CacheIndicator from "./CacheIndicator";
import CrossDomainRecommendations from "./CrossDomainRecommendations";

// Qloo Power Showcase Component
const QlooPowerShowcase = ({ data }: { data: PlaylistData }) => {
  if (!data.qloo_power_showcase) return null;
  
  const showcase = data.qloo_power_showcase;
  
  return (
    <div className="bg-gradient-to-r from-yellow-200 via-orange-200 to-red-200 text-black p-2 sm:p-4 rounded-xl mb-2 sm:mb-4 comic-shadow comic-border">
      <div className="flex items-center justify-center mb-2 sm:mb-3">
        <div className="w-5 h-5 sm:w-6 sm:h-6 bg-black/20 rounded-full flex items-center justify-center mr-2">
          <Music className="w-3 h-3 sm:w-4 sm:h-4" />
        </div>
        <h3 className="text-base sm:text-lg font-bold text-center">
          🎵 Qloo AI Power Showcase
          {showcase.location_awareness && <span className="text-sm ml-2 text-blue-600">📍</span>}
        </h3>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-1.5 sm:gap-2 mb-2 sm:mb-3 justify-items-center">
        <div className={`flex items-center bg-white/80 rounded-lg p-2 sm:p-3 comic-border ${showcase.cultural_intelligence ? 'ring-2 ring-green-400 shadow-lg' : ''}`}>
          <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2 text-green-600" />
          <span className="text-xs sm:text-sm font-bold">
            Cultural Intelligence
            {showcase.cultural_intelligence && <span className="text-xs ml-1 text-green-600">🌍</span>}
          </span>
        </div>
        <div className={`flex items-center bg-white/80 rounded-lg p-2 sm:p-3 comic-border ${showcase.location_awareness ? 'ring-2 ring-blue-400 shadow-lg' : ''}`}>
          <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2 text-green-600" />
          <span className="text-xs sm:text-sm font-bold">
            Location Awareness
            {showcase.location_awareness && <span className="text-xs ml-1 text-blue-600">📍</span>}
          </span>
        </div>
        <div className="flex items-center bg-white/80 rounded-lg p-2 sm:p-3 comic-border">
          <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2 text-green-600" />
          <span className="text-xs sm:text-sm font-bold">Multi-Strategy Analysis</span>
        </div>
        <div className="flex items-center bg-white/80 rounded-lg p-2 sm:p-3 comic-border">
          <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2 text-green-600" />
          <span className="text-xs sm:text-sm font-bold">Cross-Domain Insights</span>
        </div>
        <div className="flex items-center bg-white/80 rounded-lg p-2 sm:p-3 comic-border">
          <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2 text-green-600" />
          <span className="text-xs sm:text-sm font-bold">Affinity Scoring</span>
        </div>
        <div className="flex items-center bg-white/80 rounded-lg p-2 sm:p-3 comic-border">
          <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2 text-green-600" />
          <span className="text-xs sm:text-sm font-bold">Enhanced Gemini</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 sm:gap-2 text-center justify-items-center">
        <div className="bg-white/80 rounded-lg p-1.5 sm:p-2 comic-border">
          <div className="text-base sm:text-xl font-bold text-yellow-600">{showcase.total_recommendations}</div>
          <div className="text-xs font-bold">Total Recommendations</div>
        </div>
        <div className={`bg-white/80 rounded-lg p-1.5 sm:p-2 comic-border ${showcase.cultural_tags_count > 0 ? 'ring-2 ring-green-400 shadow-lg' : ''}`}>
          <div className={`text-base sm:text-xl font-bold ${showcase.cultural_tags_count > 0 ? 'text-green-700' : 'text-green-600'}`}>
            {showcase.cultural_tags_count}
            {showcase.cultural_tags_count > 0 && <span className="text-sm ml-1">🌍</span>}
          </div>
          <div className="text-xs font-bold">Cultural Tags</div>
        </div>
        <div className={`bg-white/80 rounded-lg p-1.5 sm:p-2 comic-border ${showcase.location_based_count > 0 ? 'ring-2 ring-blue-400 shadow-lg' : ''}`}>
          <div className={`text-base sm:text-xl font-bold ${showcase.location_based_count > 0 ? 'text-blue-700' : 'text-blue-600'}`}>
            {showcase.location_based_count}
            {showcase.location_based_count > 0 && <span className="text-sm ml-1">📍</span>}
          </div>
          <div className="text-xs font-bold">Location-Based</div>
        </div>
        <div className="bg-white/80 rounded-lg p-1.5 sm:p-2 comic-border">
          <div className="text-base sm:text-xl font-bold text-red-600">{data.enhanced_features?.length || 0}</div>
          <div className="text-xs font-bold">AI Features</div>
        </div>
      </div>
      
      {data.cultural_insights && data.cultural_insights.length > 0 && (
        <div className="mt-2 sm:mt-3 bg-white/80 rounded-lg p-2 sm:p-3 comic-border text-center">
          <h4 className="font-semibold mb-1.5 text-sm sm:text-base">🌍 Cultural Insights for {data.location_used}</h4>
          <div className="text-xs sm:text-sm font-bold">
            {data.cultural_insights.slice(0, 3).map((insight: any, index: number) => (
              <div key={index} className="mb-0.5">
                • {insight.name || insight.title || 'Cultural trend'}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {data.cultural_context && (
        <div className="mt-2 sm:mt-3 bg-white/80 rounded-lg p-2 sm:p-3 comic-border text-center">
          <h4 className="font-semibold mb-1.5 text-sm sm:text-base">🎭 Cultural Context</h4>
          <div className="text-xs sm:text-sm font-bold">{data.cultural_context}</div>
        </div>
      )}
    </div>
  );
};

interface Track {
  name: string;
  artist: string;
  album_name: string;
  release_year: string;
  album_art_url?: string;
  preview_url?: string;
  url: string;
  context_score: number;
}

interface PlaylistData {
  playlist: Track[];
  tags: string[];
  qloo_artists: Array<{ name: string; [key: string]: any }>;
  context_type: string;
  from_cache?: boolean;
  cache_timestamp?: string;
  generated_timestamp?: string;
  location_used?: string;  // NEW: Show which location was used
  user_country?: string;   // NEW: Show user's country
  // ENHANCED: Qloo power showcase information
  qloo_power_showcase?: {
    enhanced_system: boolean;
    cultural_intelligence: boolean;
    location_awareness: boolean;
    multi_strategy_recommendations: boolean;
    cross_domain_analysis: boolean;
    url_encoding_fixed: boolean;
    total_recommendations: number;
    cultural_tags_count: number;
    location_based_count: number;
    average_affinity_score?: number;
  };
  enhanced_features?: string[];
  cultural_insights?: any[];
  cultural_context?: string;
}

interface PlaylistResponseProps {
  data: PlaylistData;
}

// Comic-style gradient palette - consistent with comic theme
const comicBackgrounds = [
  "linear-gradient(135deg, #ffeb3b 0%, #ffc107 100%)", // yellow
  "linear-gradient(135deg, #4caf50 0%, #8bc34a 100%)", // green
  "linear-gradient(135deg, #2196f3 0%, #03a9f4 100%)", // blue
  "linear-gradient(135deg, #ff9800 0%, #ff5722 100%)", // orange
  "linear-gradient(135deg, #e91e63 0%, #f44336 100%)", // pink-red
  "linear-gradient(135deg, #9c27b0 0%, #673ab7 100%)", // purple
  "linear-gradient(135deg, #00bcd4 0%, #009688 100%)", // cyan-teal
];

// Cat meme mapping for moods/emotions
const catEmotions = [
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_1.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Fierce and feisty, ready to take on the world!",
    tags: ["angry", "fierce", "mad", "rage", "furious"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_2.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Feeling blue and in need of a comforting tune.",
    tags: ["sad", "blue", "cry", "teary", "heartbroken", "depressed"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_3.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Mischievous mood, up to something fun.",
    tags: ["mischievous", "playful", "sneaky", "fun", "cheeky"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_4.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Perplexed and pondering life’s mysteries.",
    tags: ["confused", "perplexed", "lost", "puzzled", "uncertain"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_5.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Sophisticated and classy, craving elegance.",
    tags: ["classy", "sophisticated", "elegant", "fancy", "refined"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_6.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Celebratory and festive, let’s get the party started!",
    tags: ["party", "celebrate", "festive", "excited", "happy"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_7.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Heartbroken or touched, feeling all the feels.",
    tags: ["crying", "emotional", "heartbroken", "touched", "sentimental"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_16.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Amazed and inspired, ready for something epic.",
    tags: ["amazed", "inspired", "epic", "wow", "impressed"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_9.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Playful and meme-loving, in a silly mood.",
    tags: ["playful", "meme", "silly", "funny", "goofy"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_10.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Grumpy and unimpressed, need a pick-me-up.",
    tags: ["grumpy", "unimpressed", "bored", "meh", "tired"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_11.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Awkward and shy, looking for comfort.",
    tags: ["awkward", "shy", "nervous", "anxious", "uncomfortable"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_12.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Hungry or craving something delicious.",
    tags: ["hungry", "craving", "food", "snack", "yum"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_13.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Feeling bold and ready for action.",
    tags: ["bold", "brave", "action", "ready", "confident"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_14.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Joyful and energetic, time to dance!",
    tags: ["joyful", "energetic", "dance", "happy", "celebrate"]
  },
  {
    url: "https://s3.getstickerpack.com/storage/uploads/sticker-pack/memes-19/sticker_15.png?78062e5195189616ab0fc020690f6b1f&d=200x200",
    description: "Hopeful and longing, wishing for something.",
    tags: ["hopeful", "longing", "wishful", "dreamy", "optimistic"]
  }
];

function findMatchingCat(context, tags) {
  const all = [context, ...(tags || [])].join(' ').toLowerCase();
  for (const cat of catEmotions) {
    if (cat.tags.some(tag => all.includes(tag))) {
      return cat;
    }
  }
  // fallback: random cat
  return catEmotions[Math.floor(Math.random() * catEmotions.length)];
}

export default function PlaylistResponse({ data }: PlaylistResponseProps) {
  const [playlistName, setPlaylistName] = useState("");
  const [selectedTracks, setSelectedTracks] = useState<Set<number>>(new Set());
  const [isCreating, setIsCreating] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showCrossDomain, setShowCrossDomain] = useState(false);
  const [spotifyToken, setSpotifyToken] = useState("");

  // Get Spotify token from localStorage
  useEffect(() => {
    const storedToken = localStorage.getItem('spotify_token');
    if (storedToken) {
      setSpotifyToken(storedToken);
    }
  }, []);

  // Use the context type from the music recommendation data as user context
  const userContext = data.context_type || "music recommendations";

  // Safety check: ensure playlist data exists and has tracks
  if (!data || !data.playlist || !Array.isArray(data.playlist) || data.playlist.length === 0) {
    return (
      <div className="bg-yellow-50 border-2 border-black rounded-lg p-4 comic-shadow">
        <div className="flex items-center gap-3 mb-3">
          <img src="/cat/404caty.jpg" alt="Error Cat" className="w-8 h-8 rounded-full border-2 border-black" />
          <span className="font-bold text-black font-comic">No playlist data available</span>
        </div>
        <div className="text-gray-600 text-sm font-comic">
          😼 "Sorry, I couldn't generate a playlist for you. Please try again!"
        </div>
      </div>
    );
  }

  // Sort tracks by relevance score (highest first)
  const sortedPlaylist = [...data.playlist].sort((a, b) => b.context_score - a.context_score);

  // Location Information Display
  const locationInfo = data.location_used && (
    <div className="flex items-center justify-center gap-2 text-sm text-gray-600 mb-2 p-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border-2 border-blue-300 shadow-sm">
      <span className="text-blue-600 text-base">📍</span>
      <span className="font-semibold">Location-based recommendations from: <strong className="text-blue-700">{data.location_used}</strong></span>
      {data.user_country && (
        <span className="text-xs bg-blue-200 px-1.5 py-0.5 rounded-full font-medium text-blue-800">
          {data.user_country}
        </span>
      )}
      {data.qloo_power_showcase?.location_based_count > 0 && (
        <span className="text-xs bg-green-200 px-1.5 py-0.5 rounded-full font-medium text-green-800">
          {data.qloo_power_showcase.location_based_count} location-aware
        </span>
      )}
    </div>
  );

  const content = sortedPlaylist.map((track, idx) => ({
    title: track?.name || 'Unknown Track',
    description: `${track?.artist || 'Unknown Artist'} • ${track?.album_name || 'Unknown Album'} (${track?.release_year || 'Unknown Year'})`,
    content: (
      <div
        className="flex flex-col h-full w-full items-start justify-start text-black p-4 sm:p-6 border-4 border-black rounded-xl shadow-lg comic-shadow min-h-[650px] min-w-[450px]"
        style={{
          background: comicBackgrounds[idx % comicBackgrounds.length],
        }}
      >
        <img
          src={track?.album_art_url || '/placeholder.svg'}
          alt={track?.album_name || 'Album Cover'}
          className="w-44 h-44 sm:w-52 sm:h-52 rounded-xl border-2 border-black mb-4 sm:mb-6 object-cover shadow-lg"
          onError={(e) => {
            e.currentTarget.src = '/placeholder.svg';
          }}
        />
        <a
          href={track?.url || '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-3 px-3 py-2 sm:px-4 sm:py-2.5 bg-green-500 hover:bg-green-600 text-white font-bold rounded-lg border-2 border-black transition comic-shadow text-xs sm:text-sm flex items-center justify-center gap-2 w-auto text-center min-h-[40px]"
        >
          <img 
            src="/icon/spotify.png" 
            alt="Spotify" 
            className="w-3 h-3 sm:w-4 sm:h-4"
          />
          <span className="whitespace-nowrap">Listen on Spotify</span>
        </a>
        <div className="mt-3 text-xs sm:text-sm text-black font-bold bg-white px-3 py-1.5 rounded comic-shadow border border-black">
          ⭐ Score: {(track?.context_score || 0).toFixed(2)}
        </div>
      </div>
    ),
  }));

  const handleTrackToggle = (trackIndex: number) => {
    const newSelected = new Set(selectedTracks);
    if (newSelected.has(trackIndex)) {
      newSelected.delete(trackIndex);
    } else {
      newSelected.add(trackIndex);
    }
    setSelectedTracks(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedTracks.size === sortedPlaylist.length) {
      setSelectedTracks(new Set());
    } else {
      setSelectedTracks(new Set(sortedPlaylist.map((_, index) => index)));
    }
  };

  const extractTrackId = (spotifyUrl: string) => {
    try {
      // Handle different Spotify URL formats
      const url = new URL(spotifyUrl);
      const pathParts = url.pathname.split('/');
      const trackId = pathParts[pathParts.length - 1];
      return trackId;
    } catch (error) {
      console.error("Error extracting track ID:", error);
      return null;
    }
  };

  const handleCreatePlaylist = async () => {
    if (!playlistName.trim()) {
      setError("Please enter a playlist name.");
      return;
    }

    if (selectedTracks.size === 0) {
      setError("Please select at least one track.");
      return;
    }

    setIsCreating(true);
    setError("");
    setSuccess("");

    try {
      const selectedTrackUrls = Array.from(selectedTracks).map(index => 
        sortedPlaylist[index].url
      );

      const response = await fetch('http://localhost:5500/create-playlist', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: playlistName,
          description: `Playlist created by SoniqueDNA AI - ${data.context_type} vibes`,
          track_uris: selectedTrackUrls,
          spotify_token: localStorage.getItem('spotify_token'),
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        setSuccess(`Playlist "${playlistName}" created successfully on Spotify!`);
        setTimeout(() => {
          setIsOpen(false);
          setPlaylistName("");
          setSelectedTracks(new Set());
          setSuccess("");
        }, 2000);
      } else {
        if (result.error && result.error.includes('permissions')) {
          setError("Permission denied. Please reconnect to Spotify and ensure you grant playlist creation permissions.");
        } else {
          setError(`Error creating playlist: ${result.error}`);
        }
      }
    } catch (error) {
      setError("Error creating playlist. Please try again.");
      console.error("Error creating playlist:", error);
    } finally {
      setIsCreating(false);
    }
  };

  const handleDialogOpen = (open: boolean) => {
    setIsOpen(open);
    if (!open) {
      setPlaylistName("");
      setSelectedTracks(new Set());
      setError("");
      setSuccess("");
    }
  };

  // Extract top 5 artists based on context scores
  const getTopScoredArtists = () => {
    if (!data.playlist || data.playlist.length === 0) return [];
    
    // Group tracks by artist and calculate average context score
    const artistScores = data.playlist.reduce((acc, track) => {
      if (!acc[track.artist]) {
        acc[track.artist] = {
          artist: track.artist,
          tracks: [],
          totalScore: 0,
          count: 0
        };
      }
      acc[track.artist].tracks.push(track);
      acc[track.artist].totalScore += track.context_score;
      acc[track.artist].count += 1;
      return acc;
    }, {} as Record<string, { artist: string; tracks: Track[]; totalScore: number; count: number }>);

    // Calculate average scores and sort by highest score
    const artistAverages = Object.values(artistScores).map(artistData => ({
      artist: artistData.artist,
      averageScore: artistData.totalScore / artistData.count,
      trackCount: artistData.count,
      tracks: artistData.tracks
    }));

    // Sort by average score (highest first) and take top 5
    return artistAverages
      .sort((a, b) => b.averageScore - a.averageScore)
      .slice(0, 5)
      .map(item => item.artist);
  };

  const topScoredArtists = getTopScoredArtists();

  const cat = findMatchingCat(data.context_type, data.tags);

  return (
    <div className="w-full max-w-6xl mx-auto py-1 sm:py-4 px-2 sm:px-4 overflow-x-hidden">
      {/* ENHANCED: Show Qloo power showcase */}
      <QlooPowerShowcase data={data} />
      
      <div className="mb-2 sm:mb-4 text-center">
        <div className="w-full text-center break-words">
          <ComicText
            fontSize={3.5}
            className="text-[1.8rem] sm:text-[2.2rem] md:text-[3.5rem]"
            style={{
              wordBreak: "break-word",
              overflowWrap: "break-word",
              whiteSpace: "normal",
            }}
          >
            {`Perfect vibes for ${data.context_type}`}
          </ComicText>
        </div>
        
        {/* Cache Indicator */}
        <div className="flex justify-center mt-2 mb-1">
          <CacheIndicator 
            fromCache={data.from_cache || false} 
            timestamp={data.cache_timestamp || data.generated_timestamp}
          />
        </div>
        
        {/* Location Information */}
        {locationInfo}
        
        {/* Cat meme for mood - moved here */}
        <div className="flex flex-col items-center justify-center mt-2 sm:mt-3 mb-1">
          <img src={cat.url} alt={cat.description} className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 rounded-xl border-2 border-black mb-1 object-cover shadow-lg" />
          <div className="text-sm sm:text-base font-bold text-black text-center comic-shadow max-w-xs px-2">{cat.description}</div>
        </div>
        <p className="text-sm sm:text-base md:text-lg text-black font-bold mb-2 sm:mb-3 comic-shadow px-2 sm:px-0">
          {sortedPlaylist.length} tracks curated by AI • Sorted by relevance score
        </p>
        

        
        {/* Action Buttons */}
        <div className="mb-2 sm:mb-3 flex justify-center gap-2 sm:gap-3">
          <Dialog open={isOpen} onOpenChange={handleDialogOpen}>
            <DialogTrigger asChild>
              <Button className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 sm:py-3 px-4 sm:px-6 rounded-lg border-2 border-black comic-shadow flex items-center gap-2 text-sm sm:text-base transition-colors">
                <Plus className="w-4 h-4 sm:w-5 sm:h-5" />
                Add to Spotify Playlist
              </Button>
            </DialogTrigger>
            <DialogContent 
              className="bg-yellow-100 border-4 border-black comic-shadow max-w-[95vw] sm:max-w-md mx-auto p-4 sm:p-6"
              aria-describedby="playlist-description"
            >
              <DialogHeader>
                <DialogTitle className="text-lg sm:text-xl font-bold text-black comic-shadow text-center">
                  Create Spotify Playlist
                </DialogTitle>
                <p id="playlist-description" className="text-xs text-gray-600 text-center mt-2">
                  Note: You need to grant playlist creation permissions when connecting to Spotify
                </p>
              </DialogHeader>
              
              <div className="space-y-3 sm:space-y-4">
                {/* Playlist Name Input */}
                <div>
                  <Label htmlFor="playlist-name" className="text-black font-bold text-sm">
                    Playlist Name
                  </Label>
                  <Input
                    id="playlist-name"
                    value={playlistName}
                    onChange={(e) => setPlaylistName(e.target.value)}
                    placeholder="Enter playlist name..."
                    className="border-2 border-black comic-shadow bg-white"
                  />
                </div>

                {/* Select All Button */}
                <div className="flex justify-between items-center">
                  <Label className="text-black font-bold text-sm">
                    Select Tracks ({selectedTracks.size}/{sortedPlaylist.length})
                  </Label>
                  <Button
                    onClick={handleSelectAll}
                    variant="outline"
                    className="text-xs bg-yellow-200 border-2 border-black comic-shadow hover:bg-yellow-300"
                  >
                    {selectedTracks.size === sortedPlaylist.length ? 'Deselect All' : 'Select All'}
                  </Button>
                </div>

                {/* Track Selection */}
                <div className="max-h-48 sm:max-h-60 overflow-y-auto space-y-2">
                  {sortedPlaylist.map((track, index) => (
                    <div key={index} className="flex items-center space-x-2 sm:space-x-3 p-2 sm:p-3 bg-white rounded-lg border-2 border-black hover:bg-yellow-50 transition-colors">
                      <Checkbox
                        id={`track-${index}`}
                        checked={selectedTracks.has(index)}
                        onCheckedChange={() => handleTrackToggle(index)}
                        className="border-2 border-black data-[state=checked]:bg-green-500"
                      />
                      <div className="flex-1 min-w-0">
                        <Label htmlFor={`track-${index}`} className="text-xs sm:text-sm font-bold text-black truncate cursor-pointer">
                          {track.name}
                        </Label>
                        <p className="text-xs text-gray-600 truncate">
                          {track.artist} • ⭐ {track.context_score.toFixed(2)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Error/Success Messages */}
                {error && (
                  <div className="flex items-center gap-2 p-3 bg-red-100 border-2 border-red-500 rounded-lg">
                    <AlertCircle className="w-4 h-4 text-red-500" />
                    <span className="text-sm text-red-700 font-bold">{error}</span>
                    {error.includes('permissions') && (
                      <Button
                        onClick={() => {
                          localStorage.removeItem('spotify_token');
                          window.location.href = '/';
                        }}
                        className="ml-2 bg-blue-500 hover:bg-blue-600 text-white text-xs px-2 py-1 rounded border border-black"
                      >
                        Reconnect Spotify
                      </Button>
                    )}
                  </div>
                )}

                {success && (
                  <div className="flex items-center gap-2 p-3 bg-green-100 border-2 border-green-500 rounded-lg">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-green-700 font-bold">{success}</span>
                  </div>
                )}

                {/* Create Button */}
                <Button
                  onClick={handleCreatePlaylist}
                  disabled={isCreating || !playlistName.trim() || selectedTracks.size === 0}
                  className="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-4 rounded-lg border-2 border-black comic-shadow disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {isCreating ? (
                    <div className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Creating...
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <Music className="w-4 h-4" />
                      Create Playlist
                    </div>
                  )}
                </Button>
              </div>
            </DialogContent>
          </Dialog>
          
          <Button 
            onClick={() => setShowCrossDomain(true)}
            className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 sm:py-3 px-4 sm:px-6 rounded-lg border-2 border-black comic-shadow flex items-center gap-2 text-sm sm:text-base transition-colors"
            disabled={!spotifyToken}
          >
            <Sparkles className="w-4 h-4 sm:w-5 sm:h-5" />
            Discover More
          </Button>
        </div>

        <div className="flex flex-wrap justify-center gap-1 sm:gap-1.5 lg:gap-2 mb-1 px-1 sm:px-2 lg:px-0">
          {data.tags.map((tag, idx) => (
            <span key={idx} className="bg-yellow-200 border-2 border-black rounded-full px-1.5 py-0.5 sm:px-2 sm:py-0.5 lg:px-3 lg:py-1 text-xs font-bold text-black comic-shadow">
              {tag}
            </span>
          ))}
        </div>
        {data.qloo_artists.length > 0 && (
          <div className="flex flex-wrap justify-center gap-1 sm:gap-1.5 lg:gap-2 items-center mt-1 px-1 sm:px-2 lg:px-0">
            <span className="text-xs font-bold text-black mr-1 sm:mr-2">Similar artists:</span>
            {data.qloo_artists.map((artist, idx) => (
              <span key={idx} className="bg-blue-200 border-2 border-black rounded-full px-1 py-0.5 sm:px-1.5 sm:py-0.5 lg:px-2 text-xs font-bold text-black comic-shadow">
                {typeof artist === 'string' ? artist : artist.name}
              </span>
            ))}
          </div>
        )}
        
        {/* Top Scored Artists Display */}
        {topScoredArtists.length > 0 && (
          <div className="flex flex-wrap justify-center gap-1 sm:gap-1.5 lg:gap-2 items-center mt-1 px-1 sm:px-2 lg:px-0">
            <span className="text-xs font-bold text-black mr-1 sm:mr-2">Top scored artists:</span>
            {topScoredArtists.map((artist, idx) => (
              <span key={idx} className="bg-green-200 border-2 border-black rounded-full px-1 py-0.5 sm:px-1.5 sm:py-0.5 lg:px-2 text-xs font-bold text-black comic-shadow">
                {artist}
              </span>
            ))}
          </div>
        )}
        

      </div>
      
      {/* Mobile Layout - Compact Scrollable List */}
      <div className="lg:hidden space-y-1 max-h-[40vh] sm:max-h-[50vh] overflow-y-auto px-1">
        {sortedPlaylist.map((track, idx) => (
          <div key={track.name + idx} className="w-full">
                    {/* Mobile Compact Card */}
        <div
          style={{ background: comicBackgrounds[idx % comicBackgrounds.length] }}
          className="w-full h-14 sm:h-18 md:h-22 rounded-xl border-4 border-black shadow-lg comic-shadow flex items-center justify-center overflow-hidden p-1.5 sm:p-2"
        >
              {/* Compact Mobile Content */}
              <div className="flex flex-row h-full w-full items-center justify-center text-black gap-2 sm:gap-3">
                {/* Small Album Art */}
                <div className="flex-shrink-0">
                  <img
                    src={track.album_art_url || '/placeholder.svg'}
                    alt={track.album_name}
                    className="w-8 h-8 sm:w-10 sm:h-10 md:w-14 md:h-14 rounded-lg border-2 border-black object-cover shadow"
                    onError={(e) => {
                      e.currentTarget.src = '/placeholder.svg';
                    }}
                  />
                </div>
                
                {/* Track Info */}
                <div className="flex-1 min-w-0 text-center">
                  <h3 className="text-xs sm:text-sm md:text-base font-extrabold text-black truncate">
                    {track.name}
                  </h3>
                  <p className="text-xs text-black font-bold truncate">
                    {track.artist}
                  </p>
                </div>
                
                {/* Compact Score */}
                <div className="flex-shrink-0">
                  <div className="text-xs text-black font-bold bg-white/80 px-1 sm:px-1.5 py-0.5 rounded comic-shadow border border-black">
                    ⭐ {track.context_score.toFixed(1)}
                  </div>
                </div>
                
                {/* Spotify Link */}
                <div className="flex-shrink-0">
                  <a
                    href={track.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-1 sm:px-1.5 py-0.5 bg-yellow-200 text-black font-bold rounded border-2 border-black text-xs comic-shadow flex items-center justify-center"
                  >
                    <img 
                      src="/icon/spotify.png" 
                      alt="Spotify" 
                      className="w-3 h-3 sm:w-4 sm:h-4"
                    />
                  </a>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Desktop Layout - StickyScroll */}
      <div className="hidden lg:block">
      <StickyScroll content={content} />
      </div>

      {/* Cross-Domain Recommendations */}
      {showCrossDomain && (
        <Dialog open={showCrossDomain} onOpenChange={setShowCrossDomain}>
          <DialogContent 
            className="bg-yellow-100 border-4 border-black comic-shadow max-w-[95vw] sm:max-w-6xl mx-auto p-4 sm:p-6 max-h-[90vh] overflow-y-auto"
            aria-describedby="cross-domain-description"
          >
            <DialogHeader>
              <DialogTitle className="text-lg sm:text-xl font-bold text-black comic-shadow text-center">
                Cross-Domain Recommendations
              </DialogTitle>
              <p id="cross-domain-description" className="text-xs text-gray-600 text-center mt-2">
                Discover movies, books, TV shows, and podcasts based on your music taste
              </p>
            </DialogHeader>
            <CrossDomainRecommendations
              userContext={userContext}
              musicArtists={data.qloo_artists.slice(0, 5).map(artist => typeof artist === 'string' ? artist : artist.name)}
              topScoredArtists={topScoredArtists.slice(0, 5)}
              userTags={data.tags}
              spotifyToken={spotifyToken}
              onClose={() => setShowCrossDomain(false)}
            />
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}