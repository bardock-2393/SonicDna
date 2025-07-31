import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { Loader2, History, Play, Calendar, Clock, Tag, Music, Trash2, Trash, CheckCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface HistoryItem {
  id: number;
  recommendation_type: string;
  user_context: string;
  generated_tags: string[];
  qloo_artists: Array<{ name: string; [key: string]: any }>;
  playlist_data: any[];
  response_time: number;
  created_at: string;
}

interface ArtistDetails {
  id: string;
  name: string;
  image: string | null;
  genres: string[];
  popularity: number;
  followers: number;
  spotify_url: string;
  uri: string;
}

interface RecommendationHistoryProps {
  userId: string;
  spotifyToken: string;
  onReplayRecommendation?: (historyItem: HistoryItem) => void;
}

const RecommendationHistory: React.FC<RecommendationHistoryProps> = ({
  userId,
  spotifyToken,
  onReplayRecommendation
}) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [deletingItem, setDeletingItem] = useState<number | null>(null);
  const [artistDetails, setArtistDetails] = useState<Record<string, ArtistDetails>>({});
  const [loadingArtists, setLoadingArtists] = useState<Set<string>>(new Set());

  const loadHistory = async () => {
    if (!userId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:5500/recommendations/history/${userId}?limit=20`);
      if (!response.ok) {
        throw new Error('Failed to load history');
      }
      
      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const fetchArtistDetails = async (artistName: string) => {
    if (!spotifyToken || artistDetails[artistName] || loadingArtists.has(artistName)) {
      return;
    }

    console.log(`Fetching details for artist: ${artistName}`);
    setLoadingArtists(prev => new Set(prev).add(artistName));

    try {
      const response = await fetch('http://localhost:5500/recommendations/artist-details', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spotify_token: spotifyToken,
          artist_name: artistName
        })
      });

      if (response.ok) {
        const data = await response.json();
        console.log(`Artist details received for ${artistName}:`, data.artist);
        setArtistDetails(prev => ({
          ...prev,
          [artistName]: data.artist
        }));
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.warn(`Failed to fetch details for ${artistName}: ${response.status}`, errorData);
      }
    } catch (err) {
      console.error(`Failed to fetch details for ${artistName}:`, err);
    } finally {
      setLoadingArtists(prev => {
        const newSet = new Set(prev);
        newSet.delete(artistName);
        return newSet;
      });
    }
  };

  const handleArtistClick = (artistName: string) => {
    const details = artistDetails[artistName];
    if (details?.spotify_url) {
      window.open(details.spotify_url, '_blank');
    }
  };

  const replayRecommendation = async (historyItem: HistoryItem) => {
    if (!spotifyToken) {
      setError('Spotify token required to replay recommendation');
      return;
    }

    try {
      const response = await fetch(`http://localhost:5500/recommendations/history/${userId}/${historyItem.id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spotify_token: spotifyToken
        })
      });

      if (!response.ok) {
        throw new Error('Failed to replay recommendation');
      }

      const data = await response.json();
      
      if (onReplayRecommendation) {
        onReplayRecommendation(historyItem);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to replay recommendation');
    }
  };

  const deleteHistoryItem = async (historyItem: HistoryItem) => {
    setError(null);
    setSuccessMessage(null);
    setDeletingItem(historyItem.id);

    try {
      const response = await fetch(`http://localhost:5500/user-history/${userId}/${historyItem.id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error('Failed to delete recommendation');
      }

      // Remove the item from the local state
      setHistory(prevHistory => prevHistory.filter(item => item.id !== historyItem.id));
      setSuccessMessage('Recommendation deleted successfully!');
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete recommendation');
    } finally {
      setDeletingItem(null);
    }
  };

  const clearAllHistory = async () => {
    setError(null);
    setSuccessMessage(null);

    try {
      const response = await fetch(`http://localhost:5500/user-history/${userId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error('Failed to clear history');
      }

      // Clear the local state
      setHistory([]);
      setSuccessMessage('All history cleared successfully!');
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear history');
    }
  };

  useEffect(() => {
    loadHistory();
  }, [userId]);

  useEffect(() => {
    // Fetch artist details for all artists in history
    if (history.length > 0 && spotifyToken) {
      const allArtists = new Set<string>();
      history.forEach(item => {
        if (item.qloo_artists) {
          item.qloo_artists.forEach(artist => {
            const artistName = typeof artist === 'string' ? artist : artist.name;
            if (artistName && !artistDetails[artistName]) {
              allArtists.add(artistName);
            }
          });
        }
      });

      allArtists.forEach(artistName => {
        fetchArtistDetails(artistName);
      });
    }
  }, [history, spotifyToken, artistDetails]);

  const getRecommendationTypeIcon = (type: string) => {
    switch (type) {
      case 'music':
        return <Music className="h-4 w-4" />;
      case 'crossdomain':
        return <Tag className="h-4 w-4" />;
      default:
        return <History className="h-4 w-4" />;
    }
  };

  const getRecommendationTypeColor = (type: string) => {
    switch (type) {
      case 'music':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'crossdomain':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  if (loading) {
    return (
      <Card className="w-full border-2 border-black comic-shadow">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Recommendation History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="ml-2">Loading history...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full border-2 border-black comic-shadow">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Recommendation History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center p-4">
            <p className="text-red-600 mb-4">{error}</p>
            <Button onClick={loadHistory} variant="outline" className="border-2 border-black comic-shadow">
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (history.length === 0) {
    return (
      <Card className="w-full border-2 border-black comic-shadow">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Recommendation History
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-gray-500">
            <History className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No recommendation history yet</p>
            <p className="text-sm">Your past recommendations will appear here</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full border-2 border-black comic-shadow">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Recommendation History
            </CardTitle>
            <CardDescription>
              Your past recommendations and discoveries
            </CardDescription>
          </div>
          {history.length > 0 && (
            <Button
              onClick={clearAllHistory}
              size="sm"
              variant="destructive"
              className="flex items-center gap-2 border-2 border-black comic-shadow"
            >
              <Trash className="h-4 w-4" />
              Clear All
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {successMessage && (
          <div className="mb-4 p-3 bg-green-50 border-2 border-green-500 rounded-lg flex items-center gap-2 comic-shadow">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <span className="text-green-800 text-sm font-bold">{successMessage}</span>
          </div>
        )}
        <ScrollArea className="h-[600px] pr-4">
          <div className="space-y-4 pb-4">
            {history.map((item) => (
              <Card key={item.id} className="hover:shadow-lg transition-shadow border-2 border-black comic-shadow">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-2">
                      {getRecommendationTypeIcon(item.recommendation_type)}
                      <Badge className={`${getRecommendationTypeColor(item.recommendation_type)} border-2 border-black font-bold`}>
                        {item.recommendation_type}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Calendar className="h-3 w-3" />
                      {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
                    </div>
                  </div>

                  {item.user_context && (
                    <div className="mb-4">
                      <p className="text-sm font-bold text-gray-700 mb-1">Context:</p>
                      <p className="text-sm text-gray-600 italic bg-gray-50 p-2 rounded border border-gray-200">"{item.user_context}"</p>
                    </div>
                  )}

                  {item.generated_tags && item.generated_tags.length > 0 && (
                    <div className="mb-4">
                      <p className="text-sm font-bold text-gray-700 mb-2">Tags:</p>
                      <div className="flex flex-wrap gap-2">
                        {item.generated_tags.slice(0, 5).map((tag, index) => (
                          <Badge key={index} variant="secondary" className="text-xs bg-yellow-200 border-2 border-black font-bold">
                            {tag}
                          </Badge>
                        ))}
                        {item.generated_tags.length > 5 && (
                          <Badge variant="outline" className="text-xs border-2 border-black font-bold">
                            +{item.generated_tags.length - 5} more
                          </Badge>
                        )}
                      </div>
                    </div>
                  )}

                  {item.qloo_artists && item.qloo_artists.length > 0 && (
                    <div className="mb-4">
                      <p className="text-sm font-bold text-gray-700 mb-2">Artists:</p>
                      <div className="flex flex-wrap gap-3">
                        {item.qloo_artists.slice(0, 5).map((artist, index) => {
                          const artistName = typeof artist === 'string' ? artist : artist.name;
                          const details = artistDetails[artistName];
                          const isLoading = loadingArtists.has(artistName);
                          
                          return (
                            <div
                              key={index}
                              onClick={() => handleArtistClick(artistName)}
                              className={`flex items-center gap-3 p-3 rounded-lg border-2 border-black comic-shadow cursor-pointer transition-all hover:bg-blue-50 hover:scale-105 ${
                                details?.spotify_url ? 'hover:shadow-lg' : ''
                              }`}
                            >
                              {/* Artist Image */}
                              <div className="flex-shrink-0">
                                {details?.image ? (
                                  <img
                                    src={details.image}
                                    alt={artistName}
                                    className="w-10 h-10 rounded-full object-cover border-2 border-gray-300 shadow-sm"
                                    onError={(e) => {
                                      e.currentTarget.style.display = 'none';
                                      e.currentTarget.nextElementSibling?.classList.remove('hidden');
                                    }}
                                  />
                                ) : null}
                                {(!details?.image || isLoading) && (
                                  <div className={`w-10 h-10 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center border-2 border-gray-300 ${details?.image ? 'hidden' : ''}`}>
                                    {isLoading ? (
                                      <Loader2 className="h-5 w-5 animate-spin text-white" />
                                    ) : (
                                      <span className="text-white font-bold text-sm">
                                        {artistName.charAt(0).toUpperCase()}
                                      </span>
                                    )}
                                  </div>
                                )}
                              </div>
                              
                                                              {/* Artist Info */}
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center gap-2 mb-2">
                                    <span className="text-base font-bold text-gray-900 truncate">
                                      {artistName}
                                    </span>
                                    {details && details.popularity >= 80 && (
                                      <div className="flex items-center gap-1 text-xs text-blue-600 font-semibold bg-blue-50 px-1.5 py-0.5 rounded-full border border-blue-200">
                                        <span>✓</span>
                                        <span>Verified</span>
                                      </div>
                                    )}
                                  </div>
                                  
                                  {/* Spotify Metadata */}
                                  {details && (
                                    <div className="space-y-1">
                                      {/* Followers */}
                                      <div className="text-xs text-gray-600">
                                        {details.followers.toLocaleString()} followers
                                      </div>
                                      
                                      {/* Popularity */}
                                      <div className="flex items-center gap-2">
                                        <div className="w-14 h-2 bg-gray-200 rounded-full overflow-hidden">
                                          <div 
                                            className="h-full bg-gradient-to-r from-green-400 to-blue-500 rounded-full transition-all duration-300"
                                            style={{ width: `${details.popularity}%` }}
                                          ></div>
                                        </div>
                                        <span className="text-xs text-gray-600 font-medium">{details.popularity}/100</span>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Loading State */}
                                  {isLoading && (
                                    <div className="text-xs text-gray-400">
                                      Loading metadata...
                                    </div>
                                  )}
                                </div>
                              
                              {/* Arrow Indicator */}
                              {details?.spotify_url && (
                                <div className="flex-shrink-0">
                                  <span className="text-sm text-blue-600 font-semibold">→</span>
                                </div>
                              )}
                            </div>
                          );
                        })}
                        {item.qloo_artists.length > 5 && (
                          <div className="flex items-center gap-2 p-2 rounded-lg border-2 border-black comic-shadow bg-gray-50">
                            <span className="text-sm font-medium text-gray-600">
                              +{item.qloo_artists.length - 5} more
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  <div className="flex items-center gap-4 text-sm text-gray-500 mt-4">
                    <div className="flex items-center gap-1 bg-gray-100 px-2 py-1 rounded border border-gray-200">
                      <Clock className="h-3 w-3" />
                      {item.response_time.toFixed(2)}s
                    </div>
                    {item.playlist_data && (
                      <div className="flex items-center gap-1 bg-gray-100 px-2 py-1 rounded border border-gray-200">
                        <Music className="h-3 w-3" />
                        {item.playlist_data.length} tracks
                      </div>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2 mt-3">
                    <button
                      onClick={() => replayRecommendation(item)}
                      className="bg-green-500 hover:bg-green-600 text-white font-bold border-2 border-black comic-shadow flex items-center gap-2 transition-colors px-3 py-2 min-w-[80px] shadow-lg rounded-lg"
                    >
                      <Play className="h-3 w-3" />
                      Replay
                    </button>
                    
                    <button
                      onClick={() => deleteHistoryItem(item)}
                      disabled={deletingItem === item.id}
                      className="bg-red-500 hover:bg-red-600 text-white font-bold border-2 border-black comic-shadow flex items-center gap-2 transition-colors px-3 py-2 min-w-[80px] shadow-lg rounded-lg disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                      {deletingItem === item.id ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Trash2 className="h-3 w-3" />
                      )}
                      {deletingItem === item.id ? 'Deleting...' : 'Delete'}
                    </button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default RecommendationHistory; 