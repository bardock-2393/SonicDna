import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  Film, 
  Tv, 
  Headphones, 
  BookOpen, 
  Music, 
  Loader2,
  ExternalLink,
  Star,
  ArrowLeft,
  Calendar,
  Clock,
  User,
  Globe,
  Award
} from 'lucide-react';
import { ComicText } from './magicui/comic-text';

interface CrossDomainRecommendationsProps {
  userContext: string;
  musicArtists: string[];
  topScoredArtists?: string[];
  userTags?: string[];
  spotifyToken: string;
  onClose?: () => void;
}

interface RecommendationItem {
  name: string;
  id?: string;
  description?: string;
  url?: string;
  image_url?: string;
  popularity?: number;
  affinity_score?: number;
  cultural_relevance?: number;
  tags?: string[];
  properties?: {
    image?: { url?: string } | null;
    description?: string;
    year?: number;
    genre?: string;
    director?: string;
    author?: string;
    publisher?: string;
    runtime?: string;
    rating?: string;
    language?: string;
    country?: string;
    url?: string;
    external_urls?: Record<string, string>;
    host?: string;
    episodes?: number;
    seasons?: number;
    followers?: number;
    albums?: number;
    spotify_url?: string;
  };
  selected_tag?: string;
  source_artist?: string;
}

interface CrossDomainData {
  recommendations_by_domain: {
    movie: RecommendationItem[];
    'TV show': RecommendationItem[];
    podcast: RecommendationItem[];
    book: RecommendationItem[];
    'music artist': RecommendationItem[];
  };
  top_artists: string[];
  top_artists_with_images: Array<{
    id: string;
    name: string;
    image: string | null;
    genres: string[];
    popularity: number;
    followers: number;
  }>;
  detailed_results: Record<string, any>;
  total_domains: number;
  recommendations_per_domain: number;
  user_context: string;
  music_artists: string[];
  cultural_context?: string;
  location_used?: string;
  user_country?: string;
  generated_timestamp?: string;
  from_cache?: boolean;
}

const domainConfig = {
  movie: {
    title: 'Movies',
    icon: Film,
    color: 'bg-red-500',
    hoverColor: 'hover:bg-red-600',
    description: 'Discover films that match your vibe'
  },
  'TV show': {
    title: 'TV Shows',
    icon: Tv,
    color: 'bg-blue-500',
    hoverColor: 'hover:bg-blue-600',
    description: 'Series that complement your taste'
  },
  podcast: {
    title: 'Podcasts',
    icon: Headphones,
    color: 'bg-purple-500',
    hoverColor: 'hover:bg-purple-600',
    description: 'Audio content that resonates with you'
  },
  book: {
    title: 'Books',
    icon: BookOpen,
    color: 'bg-green-500',
    hoverColor: 'hover:bg-green-600',
    description: 'Stories that align with your interests'
  },
  'music artist': {
    title: 'Music Artists',
    icon: Music,
    color: 'bg-orange-500',
    hoverColor: 'hover:bg-orange-600',
    description: 'Artists similar to your favorites'
  }
};

export default function CrossDomainRecommendations({ 
  userContext, 
  musicArtists, 
  topScoredArtists,
  userTags,
  spotifyToken,
  onClose 
}: CrossDomainRecommendationsProps) {
  const [recommendations, setRecommendations] = useState<CrossDomainData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [selectedDomain, setSelectedDomain] = useState<string>('');
  const [selectedItem, setSelectedItem] = useState<RecommendationItem | null>(null);
  const [progressInterval, setProgressInterval] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    fetchCrossDomainRecommendations();
  }, []);

  useEffect(() => {
    return () => {
      if (progressInterval) {
        clearInterval(progressInterval);
      }
    };
  }, [progressInterval]);

  // Handle keyboard events for popup
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && selectedItem) {
        setSelectedItem(null);
      }
    };

    if (selectedItem) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [selectedItem]);

  const clearCache = async () => {
    try {
      await fetch('http://127.0.0.1:5500/clear-cache', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.error('Failed to clear cache:', error);
    }
  };

  const fetchCrossDomainRecommendations = async (clearCacheFirst = false) => {
    setLoading(true);
    setError(null);
    setProgress(0);

    // Clear cache first if requested
    if (clearCacheFirst) {
      await clearCache();
    }

    // Start progress simulation
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev;
        return prev + Math.random() * 10;
      });
    }, 500);
    setProgressInterval(interval);

    try {
      const response = await fetch('http://127.0.0.1:5500/crossdomain-recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          spotify_token: spotifyToken,
          user_context: userContext,
          music_artists: musicArtists,
          top_scored_artists: topScoredArtists || [],
          user_tags: userTags || []
        }),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to fetch recommendations';
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
          
          // Check if it's a token-related error
          if (response.status === 401 || errorMessage.toLowerCase().includes('token') || errorMessage.toLowerCase().includes('unauthorized')) {
            errorMessage = 'Your Spotify connection has expired. Please reconnect your Spotify account to continue.';
          }
        } catch (jsonError) {
          if (response.status === 401) {
            errorMessage = 'Your Spotify connection has expired. Please reconnect your Spotify account to continue.';
          } else {
            errorMessage = `HTTP ${response.status}: ${response.statusText}`;
          }
        }
        throw new Error(errorMessage);
      }

      let data;
      try {
        data = await response.json();
      } catch (jsonError) {
        throw new Error('Invalid JSON response from server');
      }
      
      setRecommendations(data);
      
      // Debug: Log domain counts
      console.log('Cross-domain recommendations received:', data);
      const domains = ['movie', 'TV show', 'podcast', 'book', 'music artist'];
      domains.forEach(domain => {
        const count = data.recommendations_by_domain[domain]?.length || 0;
        console.log(`${domain}: ${count} items`);
      });
      
      // Auto-select the first domain that has data
      for (const domain of domains) {
        if (data.recommendations_by_domain[domain]?.length > 0) {
          setSelectedDomain(domain);
          break;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
      setProgress(100);
      if (progressInterval) {
        clearInterval(progressInterval);
        setProgressInterval(null);
      }
    }
  };

  const handleDomainClick = (domain: string) => {
    setSelectedDomain(domain);
    if (!recommendations) {
      fetchCrossDomainRecommendations();
    }
  };

  const renderRecommendationCard = (item: RecommendationItem, index: number) => (
    <Card 
      key={index} 
      className="border-2 border-black comic-shadow bg-white hover:shadow-lg transition-all duration-200 cursor-pointer"
      onClick={() => setSelectedItem(item)}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          {(item.properties?.image?.url || item.image_url) && (
            <img 
              src={item.properties?.image?.url || item.image_url} 
              alt={item.name}
              className="w-16 h-16 rounded-lg border-2 border-black object-cover"
              onError={(e) => {
                e.currentTarget.style.display = 'none';
              }}
            />
          )}
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-black text-sm truncate">{item.name}</h3>
            {(item.properties?.description || item.description) && (
              <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                {item.properties?.description || item.description}
              </p>
            )}
            {item.properties?.year && (
              <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {item.properties.year}
              </p>
            )}
            {item.properties?.genre && (
              <p className="text-xs text-blue-600 mt-1 font-medium">{item.properties.genre}</p>
            )}
            <div className="flex items-center gap-2 mt-2">
              {item.affinity_score && (
                <Badge variant="outline" className="text-xs">
                  <Star className="w-3 h-3 mr-1" />
                  {item.affinity_score}
                </Badge>
              )}
              {item.popularity && (
                <Badge variant="outline" className="text-xs">
                  <Award className="w-3 h-3 mr-1" />
                  {item.popularity}
                </Badge>
              )}
            </div>
            {item.source_artist && (
              <p className="text-xs text-gray-500 mt-1">
                Based on: <span className="font-medium">{item.source_artist}</span>
              </p>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderDetailedPopup = () => {
    if (!selectedItem) return null;

    return (
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        role="dialog"
        aria-modal="true"
        aria-labelledby="detail-title"
        aria-describedby="detail-description"
      >
        <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto border-4 border-black comic-shadow">
          <div className="p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <Button
                onClick={() => setSelectedItem(null)}
                variant="outline"
                className="border-2 border-black"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Discover
              </Button>
              <Button
                onClick={() => setSelectedItem(null)}
                variant="outline"
                className="border-2 border-black"
                aria-label="Close detailed view"
              >
                ✕
              </Button>
            </div>

            {/* Content */}
            <div className="space-y-4">
              {/* Image and Title */}
              <div className="flex items-start gap-4">
                {(selectedItem.properties?.image?.url || selectedItem.image_url) && (
                  <img 
                    src={selectedItem.properties?.image?.url || selectedItem.image_url} 
                    alt={selectedItem.name}
                    className="w-24 h-24 rounded-lg border-2 border-black object-cover"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                    }}
                  />
                )}
                <div className="flex-1">
                  <h2 id="detail-title" className="text-2xl font-bold text-black mb-2">{selectedItem.name}</h2>
                  {selectedItem.properties?.description && (
                    <p id="detail-description" className="text-gray-600 mb-2">{selectedItem.properties.description}</p>
                  )}
                </div>
              </div>

              {/* Details Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selectedItem.properties?.year && (
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Year: {selectedItem.properties.year}</span>
                  </div>
                )}
                {selectedItem.properties?.genre && (
                  <div className="flex items-center gap-2">
                    <Award className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Genre: {selectedItem.properties.genre}</span>
                  </div>
                )}
                {selectedItem.properties?.director && (
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Director: {selectedItem.properties.director}</span>
                  </div>
                )}
                {selectedItem.properties?.author && (
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Author: {selectedItem.properties.author}</span>
                  </div>
                )}
                {selectedItem.properties?.host && (
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Host: {selectedItem.properties.host}</span>
                  </div>
                )}
                {selectedItem.properties?.runtime && (
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Runtime: {selectedItem.properties.runtime}</span>
                  </div>
                )}
                {selectedItem.properties?.episodes && (
                  <div className="flex items-center gap-2">
                    <Tv className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Episodes: {selectedItem.properties.episodes}</span>
                  </div>
                )}
                {selectedItem.properties?.seasons && (
                  <div className="flex items-center gap-2">
                    <Tv className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Seasons: {selectedItem.properties.seasons}</span>
                  </div>
                )}
                {selectedItem.properties?.rating && (
                  <div className="flex items-center gap-2">
                    <Star className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Rating: {selectedItem.properties.rating}</span>
                  </div>
                )}
                {selectedItem.properties?.language && (
                  <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Language: {selectedItem.properties.language}</span>
                  </div>
                )}
                {selectedItem.properties?.publisher && (
                  <div className="flex items-center gap-2">
                    <BookOpen className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Publisher: {selectedItem.properties.publisher}</span>
                  </div>
                )}
                {selectedItem.properties?.followers && (
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Followers: {selectedItem.properties.followers.toLocaleString()}</span>
                  </div>
                )}
                {selectedItem.properties?.albums && (
                  <div className="flex items-center gap-2">
                    <Music className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">Albums: {selectedItem.properties.albums}</span>
                  </div>
                )}
              </div>

              {/* Scores */}
              <div className="flex gap-4">
                {selectedItem.affinity_score && (
                  <Badge className="bg-blue-500 text-white">
                    <Star className="w-3 h-3 mr-1" />
                    Affinity: {selectedItem.affinity_score}
                  </Badge>
                )}
                {selectedItem.popularity && (
                  <Badge className="bg-green-500 text-white">
                    <Award className="w-3 h-3 mr-1" />
                    Popularity: {selectedItem.popularity}
                  </Badge>
                )}
                {selectedItem.cultural_relevance && (
                  <Badge className="bg-purple-500 text-white">
                    <Globe className="w-3 h-3 mr-1" />
                    Cultural: {selectedItem.cultural_relevance}
                  </Badge>
                )}
              </div>

              {/* Tags */}
              {selectedItem.tags && selectedItem.tags.length > 0 && (
                <div>
                  <h3 className="font-bold text-sm mb-2">Tags:</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedItem.tags.map((tag, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Source Artist */}
              {selectedItem.source_artist && (
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Recommended based on:</span> {selectedItem.source_artist}
                  </p>
                </div>
              )}

              {/* External Links */}
              {(selectedItem.properties?.url || selectedItem.properties?.spotify_url || selectedItem.url) && (
                <div className="flex gap-2">
                  {selectedItem.properties?.url && (
                    <Button asChild className="bg-blue-500 hover:bg-blue-600 text-white">
                      <a href={selectedItem.properties.url} target="_blank" rel="noopener noreferrer">
                        <ExternalLink className="w-4 h-4 mr-2" />
                        Visit Website
                      </a>
                    </Button>
                  )}
                  {selectedItem.properties?.spotify_url && (
                    <Button asChild className="bg-green-500 hover:bg-green-600 text-white">
                      <a href={selectedItem.properties.spotify_url} target="_blank" rel="noopener noreferrer">
                        <Music className="w-4 h-4 mr-2" />
                        Open in Spotify
                      </a>
                    </Button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto py-4 px-4">
      {/* Header */}
      <div className="text-center mb-6">
        <ComicText
          fontSize={2.5}
          className="text-[1.5rem] sm:text-[2rem] md:text-[2.5rem] mb-2"
        >
          Discover More Based on Your Taste
        </ComicText>
        <p className="text-sm text-gray-600 font-comic">
          Explore movies, TV shows, podcasts, books, and more artists that match your vibe
        </p>
        <Button 
          onClick={() => fetchCrossDomainRecommendations(true)}
          className="mt-3 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg border-2 border-black comic-shadow"
          disabled={loading}
        >
          🔄 Refresh (Clear Cache)
        </Button>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="text-center py-8">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-sm text-gray-600 font-comic">
            Discovering amazing content for you...
          </p>
          <Progress value={progress} className="w-full max-w-md mx-auto mt-4" />
          <p className="text-xs text-gray-500 mt-2">{Math.round(progress)}%</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="text-center py-8">
          <div className="bg-red-50 border-2 border-red-500 rounded-lg p-4 max-w-md mx-auto">
            <p className="text-red-700 font-bold text-sm">{error}</p>
            <div className="flex gap-2 mt-3 justify-center">
              <Button 
                onClick={() => fetchCrossDomainRecommendations()}
                className="bg-red-500 hover:bg-red-600 text-white"
              >
                Try Again
              </Button>
              {error.includes('Spotify connection has expired') && (
                <Button 
                  onClick={() => {
                    // Clear the expired token and redirect to reconnect
                    localStorage.removeItem('spotify_token');
                    window.location.href = '/';
                  }}
                  className="bg-green-500 hover:bg-green-600 text-white"
                >
                  Reconnect Spotify
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Recommendations Display */}
      {recommendations && !loading && !error && (
        <div className="space-y-8">
          {/* Debug Info */}
          <div className="text-center p-4 bg-gray-100 rounded-lg">
            <p className="text-sm text-gray-600">
              Debug: Found {Object.keys(recommendations.recommendations_by_domain).length} domains
            </p>
            {Object.entries(recommendations.recommendations_by_domain).map(([domain, items]) => (
              <p key={domain} className="text-xs text-gray-500">
                {domain}: {Array.isArray(items) ? items.length : 0} items
              </p>
            ))}
          </div>

          {/* Show all domains with data */}
          {Object.entries(domainConfig).map(([domain, config]) => {
            const domainRecommendations = recommendations.recommendations_by_domain[domain as keyof typeof recommendations.recommendations_by_domain];
            
            if (!domainRecommendations || domainRecommendations.length === 0) {
              return null;
            }

            const IconComponent = config.icon;
            
            return (
              <div key={domain} className="space-y-4">
                {/* Domain Header */}
                <div className="text-center">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <IconComponent className="w-6 h-6" />
                    <h2 className="text-xl font-bold text-black font-comic">
                      {config.title}
                    </h2>
                  </div>
                  <p className="text-sm text-gray-600 font-comic">
                    {config.description}
                  </p>
                </div>

                {/* Recommendations Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {domainRecommendations.map((item, index) => 
                    renderRecommendationCard(item, index)
                  )}
                </div>
              </div>
            );
          })}

          {/* No Recommendations Found */}
          {(() => {
            const hasAnyRecommendations = Object.entries(domainConfig).some(([domain, config]) => {
              const domainRecommendations = recommendations.recommendations_by_domain[domain as keyof typeof recommendations.recommendations_by_domain];
              return domainRecommendations && domainRecommendations.length > 0;
            });

            if (!hasAnyRecommendations) {
              return (
                <div className="text-center py-8">
                  <div className="bg-yellow-50 border-2 border-yellow-500 rounded-lg p-6 max-w-md mx-auto">
                    <p className="text-yellow-700 font-bold text-lg mb-2">No Recommendations Found</p>
                    <p className="text-yellow-600 text-sm mb-4">
                      We couldn't find any cross-domain recommendations for your current taste profile.
                    </p>
                    <Button 
                      onClick={() => fetchCrossDomainRecommendations(true)}
                      className="bg-yellow-500 hover:bg-yellow-600 text-white"
                    >
                      Try Again with Fresh Data
                    </Button>
                  </div>
                </div>
              );
            }
            return null;
          })()}

          {/* Cache Info */}
          {recommendations.from_cache && (
            <div className="text-center mt-4">
              <Badge variant="outline" className="text-xs">
                From Cache • {new Date(recommendations.generated_timestamp || '').toLocaleString()}
              </Badge>
            </div>
          )}
        </div>
      )}

      {/* Close Button */}
      {onClose && (
        <div className="text-center mt-6">
          <Button 
            onClick={onClose}
            variant="outline"
            className="border-2 border-black comic-shadow"
          >
            Close
          </Button>
        </div>
      )}

      {/* Detailed Popup */}
      {renderDetailedPopup()}
    </div>
  );
} 