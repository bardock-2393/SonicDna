import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Loader2, BarChart3, PieChart, TrendingUp, Music, Heart, Calendar } from 'lucide-react';

interface GenreData {
  genre: string;
  artist_count: number;
  track_count: number;
  total_playtime: number;
}

interface MoodData {
  mood: string;
  preference_score: number;
  context_count: number;
}

interface TimelineData {
  month: string;
  new_artists: number;
}

interface TasteAnalytics {
  genres: GenreData[];
  moods: MoodData[];
  timeline: TimelineData[];
}

interface UserTasteGraphProps {
  userId: string;
}

const UserTasteGraph: React.FC<UserTasteGraphProps> = ({ userId }) => {
  const [analytics, setAnalytics] = useState<TasteAnalytics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('genres');

  const loadAnalytics = async () => {
    if (!userId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:5500/recommendations/taste-analytics/${userId}`);
      if (!response.ok) {
        throw new Error('Failed to load taste analytics');
      }
      
      const data = await response.json();
      setAnalytics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load taste analytics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, [userId]);

  const getGenreColor = (genre: string) => {
    const genreColors: { [key: string]: string } = {
      'pop': 'bg-pink-500',
      'rock': 'bg-red-500',
      'hip-hop': 'bg-purple-500',
      'electronic': 'bg-blue-500',
      'jazz': 'bg-indigo-500',
      'classical': 'bg-gray-500',
      'country': 'bg-orange-500',
      'r&b': 'bg-purple-500',
      'indie': 'bg-green-500',
      'latin': 'bg-yellow-500',
    };
    
    return genreColors[genre.toLowerCase()] || 'bg-gray-500';
  };

  const getMoodColor = (mood: string) => {
    const moodColors: { [key: string]: string } = {
      'happy': 'bg-yellow-500',
      'sad': 'bg-blue-500',
      'energetic': 'bg-red-500',
      'calm': 'bg-green-500',
      'romantic': 'bg-pink-500',
      'melancholic': 'bg-purple-500',
      'upbeat': 'bg-orange-500',
      'chill': 'bg-teal-500',
    };
    
    return moodColors[mood.toLowerCase()] || 'bg-gray-500';
  };

  const renderGenreChart = () => {
    if (!analytics?.genres || analytics.genres.length === 0) {
      return (
        <div className="text-center p-8 text-gray-500">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="font-comic font-bold">No genre data available</p>
          <p className="text-sm font-comic">Start exploring music to see your genre preferences!</p>
        </div>
      );
    }

    const maxCount = Math.max(...analytics.genres.map(g => g.artist_count));

    return (
      <div className="space-y-4">
        {analytics.genres.slice(0, 10).map((genre, index) => (
          <div key={genre.genre} className="space-y-2 p-3 bg-gray-50 border-2 border-black comic-shadow rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${getGenreColor(genre.genre)} border border-gray-300`} />
                <span className="font-medium text-sm font-comic font-bold">{genre.genre.toUpperCase()}</span>
                <Badge variant="secondary" className="text-xs border-2 border-black font-comic font-bold">
                  {genre.artist_count} artists
                </Badge>
              </div>
              <span className="text-sm text-gray-500 font-comic">{genre.track_count} tracks</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 border border-gray-300">
              <div
                className={`h-3 rounded-full ${getGenreColor(genre.genre)} transition-all duration-300 comic-shadow`}
                style={{ width: `${(genre.artist_count / maxCount) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderMoodChart = () => {
    if (!analytics?.moods || analytics.moods.length === 0) {
      return (
        <div className="text-center p-8 text-gray-500">
          <Heart className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="font-comic font-bold">No mood data available</p>
          <p className="text-sm font-comic">Start exploring music to see your mood preferences!</p>
        </div>
      );
    }

    return (
      <div className="grid grid-cols-2 gap-4">
        {analytics.moods.map((mood) => (
          <Card key={mood.mood} className="p-4 border-2 border-black comic-shadow">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${getMoodColor(mood.mood)} border border-gray-300`} />
                <span className="font-medium text-sm capitalize font-comic font-bold">{mood.mood}</span>
              </div>
              <Badge variant="secondary" className="text-xs border-2 border-black font-comic font-bold">
                {mood.context_count} times
              </Badge>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 border border-gray-300">
              <div
                className={`h-3 rounded-full ${getMoodColor(mood.mood)} transition-all duration-300 comic-shadow`}
                style={{ width: `${(mood.preference_score / 10) * 100}%` }}
              />
            </div>
            <div className="text-xs text-gray-500 mt-1 font-comic">
              Score: {mood.preference_score.toFixed(1)}/10
            </div>
          </Card>
        ))}
      </div>
    );
  };

  const renderTimelineChart = () => {
    if (!analytics?.timeline || analytics.timeline.length === 0) {
      return (
        <div className="text-center p-8 text-gray-500">
          <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="font-comic font-bold">No timeline data available</p>
          <p className="text-sm font-comic">Start exploring music to see your discovery timeline!</p>
        </div>
      );
    }

    const maxArtists = Math.max(...analytics.timeline.map(t => t.new_artists));

    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg border-2 border-black comic-shadow">
            <div className="text-2xl font-bold text-blue-600 font-comic">
              {analytics.timeline.reduce((sum, t) => sum + t.new_artists, 0)}
            </div>
            <div className="text-sm text-blue-600 font-comic font-bold">Total New Artists</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg border-2 border-black comic-shadow">
            <div className="text-2xl font-bold text-green-600 font-comic">
              {analytics.timeline.length}
            </div>
            <div className="text-sm text-green-600 font-comic font-bold">Active Months</div>
          </div>
        </div>
        
        <div className="space-y-3">
          {analytics.timeline.map((timeline, index) => (
            <div key={timeline.month} className="space-y-2 p-3 bg-gray-50 border-2 border-black comic-shadow rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-gray-500" />
                  <span className="font-medium text-sm font-comic font-bold">{timeline.month}</span>
                  <Badge variant="secondary" className="text-xs border-2 border-black font-comic font-bold">
                    {timeline.new_artists} artists
                  </Badge>
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 border border-gray-300">
                <div
                  className="h-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300 comic-shadow"
                  style={{ width: `${(timeline.new_artists / maxArtists) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <Card className="w-full border-2 border-black comic-shadow">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Your Taste Analytics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="ml-2 font-comic font-bold">Analyzing your taste...</span>
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
            <BarChart3 className="h-5 w-5" />
            Your Taste Analytics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center p-4">
            <p className="text-red-600 mb-4 font-comic font-bold">{error}</p>
            <Button onClick={loadAnalytics} variant="outline" className="border-2 border-black comic-shadow font-comic font-bold">
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!analytics) {
    return (
      <Card className="w-full border-2 border-black comic-shadow">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Your Taste Analytics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-gray-500">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="font-comic font-bold">No analytics data available</p>
            <p className="text-sm font-comic">Start exploring music to see your taste insights!</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full border-2 border-black comic-shadow">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Your Taste Analytics
        </CardTitle>
        <CardDescription>
          Insights into your music preferences and discovery patterns
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Debug buttons for testing */}
        <div className="flex gap-2 mb-4">
          <Button 
            onClick={async () => {
              try {
                await fetch(`http://localhost:5500/recommendations/analytics/clear/${userId}`, { method: 'POST' });
                loadAnalytics();
              } catch (err) {
                console.error('Failed to clear analytics:', err);
              }
            }}
            size="sm"
            variant="outline"
            className="border-2 border-black comic-shadow font-comic font-bold text-xs"
          >
            Clear Analytics
          </Button>
          <Button 
            onClick={async () => {
              try {
                await fetch(`http://localhost:5500/recommendations/analytics/populate-sample/${userId}`, { method: 'POST' });
                loadAnalytics();
              } catch (err) {
                console.error('Failed to populate sample analytics:', err);
              }
            }}
            size="sm"
            variant="outline"
            className="border-2 border-black comic-shadow font-comic font-bold text-xs"
          >
            Load Sample Data
          </Button>
        </div>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-white border-2 border-black comic-shadow mb-4">
            <TabsTrigger value="genres" className="flex items-center gap-2 font-comic font-bold text-xs data-[state=active]:bg-black data-[state=active]:text-white transition-colors">
              <Music className="h-4 w-4" />
              Genres
            </TabsTrigger>
            <TabsTrigger value="moods" className="flex items-center gap-2 font-comic font-bold text-xs data-[state=active]:bg-black data-[state=active]:text-white transition-colors">
              <Heart className="h-4 w-4" />
              Moods
            </TabsTrigger>
            <TabsTrigger value="timeline" className="flex items-center gap-2 font-comic font-bold text-xs data-[state=active]:bg-black data-[state=active]:text-white transition-colors">
              <TrendingUp className="h-4 w-4" />
              Timeline
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="genres" className="mt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold font-comic">Genre Preferences</h3>
                <Badge variant="secondary" className="border-2 border-black font-comic font-bold">
                  {analytics.genres.length} genres
                </Badge>
              </div>
              {renderGenreChart()}
            </div>
          </TabsContent>
          
          <TabsContent value="moods" className="mt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold font-comic">Mood Preferences</h3>
                <Badge variant="secondary" className="border-2 border-black font-comic font-bold">
                  {analytics.moods.length} moods
                </Badge>
              </div>
              {renderMoodChart()}
            </div>
          </TabsContent>
          
          <TabsContent value="timeline" className="mt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold font-comic">Discovery Timeline</h3>
                <Badge variant="secondary" className="border-2 border-black font-comic font-bold">
                  {analytics.timeline.length} months
                </Badge>
              </div>
              {renderTimelineChart()}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default UserTasteGraph; 