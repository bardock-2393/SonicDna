import React, { useState, useEffect, useRef } from 'react';
import ChatInput from './ChatInput';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import PlaylistResponse from './PlaylistResponse';
import { PlaceholdersAndVanishInput } from './ui/placeholders-and-vanish-input';
import { MultiStepLoader } from './ui/multi-step-loader';
import { Music } from 'lucide-react';
import heroImage from '@/assets/hero-music.jpg';
// Fix imports to use local UI components
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { AuroraText } from "@/components/magicui/aurora-text";
import { ComicText } from "@/components/magicui/comic-text";
import { useLocation } from 'react-router-dom';
import { toast } from "@/components/ui/use-toast";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  playlistData?: any;
}

const MusicDashboard: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [spotifyToken, setSpotifyToken] = useState<string | null>(null);
  const [spotifyUserId, setSpotifyUserId] = useState<string | null>(null);
  const [checkingToken, setCheckingToken] = useState(false);
  const [spotifyUserProfile, setSpotifyUserProfile] = useState<{ name: string; avatar: string | null } | null>(null);
  const [tokenExpiryTime, setTokenExpiryTime] = useState<Date | null>(null);
  const [showCredsModal, setShowCredsModal] = useState(false);
  const [clientId, setClientId] = useState(localStorage.getItem('spotify_client_id') || '');
  const [clientSecret, setClientSecret] = useState(localStorage.getItem('spotify_client_secret') || '');

  // Keep caching simple - no complex UI for users

  // Refresh recommendations function
  const refreshCrossDomainRecs = async () => {
    if (!spotifyToken) return;
    if (crossDomainLoading) return; // Prevent multiple simultaneous calls

    setCrossDomainLoading(true);
    setCrossDomainProgress(0);
    setCrossDomainStatus('starting');
    setCrossDomainCurrentArtist('');
    setCrossDomainCurrentDomain('');
    
    // Start fallback progress immediately in case backend progress fails
    let immediateFallbackProgress = 0;
    const fallbackInterval = setInterval(() => {
      immediateFallbackProgress = Math.min(immediateFallbackProgress + 1, 85);
      setCrossDomainProgress(immediateFallbackProgress);
    }, 1000);
    
    // Real progress tracking from backend with fallback
    let fallbackProgress = 0;
    let pollCount = 0;
    const maxPolls = 300; // Maximum 5 minutes of polling (300 * 1000ms)
    
    const progressInterval = setInterval(async () => {
      pollCount++;
      
      // Stop polling if we've exceeded the maximum time
      if (pollCount > maxPolls) {
        console.log(`[FRONTEND] Stopping progress polling - exceeded maximum time`);
        clearInterval(progressInterval);
        clearInterval(fallbackInterval);
        intervalRefs.current.progress = undefined;
        intervalRefs.current.fallback = undefined;
        setCrossDomainLoading(false);
        setCrossDomainProgress(0);
        setCrossDomainStatus('');
        return;
      }
      try {
        if (spotifyUserId) {
          console.log(`[FRONTEND] Polling progress for user: ${spotifyUserId}`);
          const progressRes = await fetch(`http://localhost:5500/crossdomain-progress/${spotifyUserId}`);
          console.log(`[FRONTEND] Progress response status: ${progressRes.status}`);
          
          if (progressRes.ok) {
            const progressData = await progressRes.json();
            console.log(`[FRONTEND] Progress data:`, progressData);
            
            setCrossDomainProgress(progressData.percentage);
            setCrossDomainStatus(progressData.status);
            setCrossDomainCurrentArtist(progressData.current_artist);
            setCrossDomainCurrentDomain(progressData.current_domain);
            
            // Stop polling if completed or error
            if (progressData.status === 'completed' || progressData.status === 'error') {
              console.log(`[FRONTEND] Stopping progress polling - status: ${progressData.status}`);
              clearInterval(progressInterval);
              clearInterval(fallbackInterval);
              intervalRefs.current.progress = undefined;
              intervalRefs.current.fallback = undefined;
              return; // Exit early to prevent further processing
            }
          } else {
            console.log(`[FRONTEND] Progress response not ok, using fallback`);
            // Fallback progress if backend progress fails
            fallbackProgress = Math.min(fallbackProgress + 2, 85);
            setCrossDomainProgress(fallbackProgress);
            setCrossDomainStatus('processing');
          }
        } else {
          console.log(`[FRONTEND] No spotifyUserId available`);
        }
      } catch (error) {
        console.error('[FRONTEND] Error fetching progress:', error);
        // Fallback progress if network error
        fallbackProgress = Math.min(fallbackProgress + 2, 85);
        setCrossDomainProgress(fallbackProgress);
        setCrossDomainStatus('processing');
      }
    }, 1000); // Poll every 1000ms instead of 500ms to reduce load // Poll every 500ms
    
    try {
      // Clear cache for this user first
      await fetch('http://localhost:5500/clear-cache', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: spotifyUserId })
      });

      // Fetch fresh recommendations
      const res = await fetch('http://localhost:5500/crossdomain-recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spotify_token: spotifyToken,
          limit: 5,  // Reduced from 10 to 5 for faster response
        }),
      });

      const data = await res.json();
      
      // Wait for progress to complete
      setTimeout(() => {
        if (data.error) {
          console.error('Cross-domain recommendations error:', data.error);
          toast({
            title: "Refresh Failed",
            description: "Couldn't get fresh recommendations. Please try again.",
            duration: 3000,
          });
        } else {
          setCrossDomainRecs(data);
          toast({
            title: "Recommendations Refreshed!",
            description: "🎉 Got fresh recommendations based on your latest music taste!",
            duration: 3000,
          });
        }
        setCrossDomainLoading(false);
        setCrossDomainProgress(0);
        setCrossDomainStatus('');
        setCrossDomainCurrentArtist('');
        setCrossDomainCurrentDomain('');
        clearInterval(fallbackInterval);
      }, 1000);
      
    } catch (error) {
      console.error('Failed to refresh recommendations:', error);
      toast({
        title: "Refresh Failed",
        description: "Something went wrong. Please try again.",
        duration: 3000,
      });
      clearInterval(progressInterval);
      clearInterval(fallbackInterval);
      intervalRefs.current.progress = undefined;
      intervalRefs.current.fallback = undefined;
      setCrossDomainLoading(false);
      setCrossDomainProgress(0);
      setCrossDomainStatus('');
      setCrossDomainCurrentArtist('');
      setCrossDomainCurrentDomain('');
    }
  };

  const loadingStates = [{
    text: "Analyzing your mood...",
    percentage: 5
  }, {
    text: "Searching music database...",
    percentage: 15
  }, {
    text: "Curating perfect tracks...",
    percentage: 25
  }, {
    text: "Finding similar artists...",
    percentage: 35
  }, {
    text: "Adding emotional tags...",
    percentage: 45
  }, {
    text: "Creating your playlist...",
    percentage: 55
  }, {
    text: "Optimizing recommendations...",
    percentage: 65
  }, {
    text: "Applying AI filters...",
    percentage: 75
  }, {
    text: "Finalizing your mix...",
    percentage: 85
  }, {
    text: "Almost ready...",
    percentage: 95,
    showCat: true
  }];

  const placeholders = ["What's your vibe today?", "Feeling romantic and loveable...", "Late night drive energy", "Study session background", "Party time vibes", "Feeling a bit blue", "Need some motivation", "Chill weekend mood"];

  // Mock playlist data for demonstration
  const generateMockPlaylist = (mood: string) => {
    const mockTracks = [{
      name: "All of Me",
      artist: "John Legend",
      album_name: "Love in the Future",
      release_year: "2013",
      album_art_url: "https://i.scdn.co/image/ab67616d0000b273c8c4bb14fd21f320a8e2f8fd",
      preview_url: "https://p.scdn.co/mp3-preview/fe1ca9de6f4ee5c09dc0a3572f12e86d3b89525e",
      url: "https://open.spotify.com/track/449d94cKjBnEPy4dAOUrXr",
      context_score: 1.32
    }, {
      name: "Perfect",
      artist: "Ed Sheeran",
      album_name: "÷ (Deluxe)",
      release_year: "2017",
      album_art_url: "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",
      preview_url: "https://p.scdn.co/mp3-preview/a15fb7d62654ac1efae1e2c2da5f0b49f4000002",
      url: "https://open.spotify.com/track/0tgVpDi06FyKpA1z0VMD4v",
      context_score: 1.29
    }, {
      name: "Thinking Out Loud",
      artist: "Ed Sheeran",
      album_name: "x (Deluxe Edition)",
      release_year: "2014",
      album_art_url: "https://i.scdn.co/image/ab67616d0000b273e9c32adf59db4e54c26a0b69",
      preview_url: "https://p.scdn.co/mp3-preview/1a1a8ed20d29ac5b916c6f95eed0b726f0e1aa40",
      url: "https://open.spotify.com/track/1GlA7CJhYdO6YQKEHwWfKF",
      context_score: 1.25
    }];
    const tags = mood.toLowerCase() === 'loveable' ? ["Romantic", "Heartfelt", "Ballad", "Love Song", "Mellow"] : ["Upbeat", "Energetic", "Feel-good", "Popular", "Mainstream"];
    const artists = mood.toLowerCase() === 'loveable' ? ["Adele", "Sam Smith", "Norah Jones", "Ed Sheeran", "John Legend"] : ["Taylor Swift", "Bruno Mars", "The Weeknd", "Dua Lipa", "Post Malone"];
    return {
      playlist: mockTracks,
      tags,
      qloo_artists: artists,
      context_type: mood
    };
  };

  const handleSendMessage = async (message: string) => {
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString() + '_user',
      text: message,
      isUser: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const res = await fetch('http://localhost:5500/musicrecommandation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_context: message,
          spotify_token: spotifyToken,
        })
      });
      const data = await res.json();

      // Show different loading time based on cache status
      if (data.from_cache) {
        // Faster loading for cached results
        await new Promise(resolve => setTimeout(resolve, 1000));
      } else {
        // Wait longer at the last loading step for fresh results
        await new Promise(resolve => setTimeout(resolve, 3500));
      }

      const responseText = data.from_cache ?
        `⚡ I remember your "${message}" vibe! Here are those perfect tracks I found for you before.` :
        `I've curated a perfect playlist for your "${message}" mood! Here are some tracks that capture that vibe perfectly.`;

      const aiMessage: Message = {
        id: Date.now().toString() + '_ai',
        text: responseText,
        isUser: false,
        timestamp: new Date(),
        playlistData: data
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now().toString() + '_ai',
        text: 'Sorry, there was an error getting your playlist.',
        isUser: false,
        timestamp: new Date()
      }]);
    }
    setIsLoading(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Optional: handle input changes if needed
  };

  const handleInputSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const message = formData.get('message') as string;
    if (message.trim()) {
      handleSendMessage(message.trim());
    }
  };

  // Helper: Check if Spotify token is valid
  const checkSpotifyTokenValidity = async (token: string | null) => {
    if (!token) return false;
    setCheckingToken(true);
    try {
      // Fetch profile from backend instead of Spotify API directly
      const res = await fetch('http://localhost:5500/spotify-profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spotify_token: token })
      });
      if (res.status === 401 || res.status === 400) {
        // Token is expired or invalid, clear it
        localStorage.removeItem('spotify_token');
        setSpotifyToken(null);
        setSpotifyUserProfile(null);
        toast({
          title: "Spotify Token Expired",
          description: "Spotify tokens expire after 8 hours. Please reconnect your account to continue.",
          duration: 5000,
        });
        return false;
      }
      const data = await res.json();
      // Cat fallback logic with localStorage persistence
      const catImages = [
        '/cat/cat1.jpg',
        '/cat/cat2.jpg',
        '/cat/cat3.jpg',
        '/cat/cat4.jpg',
      ];
      function getOrSetCat(userId) {
        const key = `cat_avatar_${userId}`;
        let cat = localStorage.getItem(key);
        if (!cat) {
          cat = catImages[Math.floor(Math.random() * catImages.length)];
          localStorage.setItem(key, cat);
        }
        return cat;
      }
      if (data && data.id) {
        setSpotifyUserId(data.id);
        setSpotifyUserProfile({
          name: data.display_name || 'Spotify User',
          avatar: data.images && data.images.length > 0 && data.images[0].url ? data.images[0].url : getOrSetCat(data.id)
        });
        return true;
      }
      setSpotifyUserProfile(null);
      return false;
    } catch (e) {
      console.error('Error checking token validity:', e);
      setSpotifyUserProfile(null);
      return false;
    } finally {
      setCheckingToken(false);
    }
  };

  const handleSaveCreds = () => {
    localStorage.setItem('spotify_client_id', clientId);
    localStorage.setItem('spotify_client_secret', clientSecret);
    setShowCredsModal(false);
    toast({
      title: "Spotify Credentials Saved",
      description: "Your Spotify Client ID and Secret have been saved.",
      duration: 3000,
    });
  };

  // On mount, check for Spotify token in localStorage or exchange code for token
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    if (code) {
      // Exchange code for access token
      fetch('http://localhost:5500/exchange-token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      })
        .then(res => res.json())
        .then(data => {
          if (data.access_token) {
            localStorage.setItem('spotify_token', data.access_token);
            setSpotifyToken(data.access_token);
            
            // Calculate and store token expiry time (8 hours from now)
            const expiresIn = data.expires_in || 28800; // 8 hours in seconds
            const expiryTime = new Date(Date.now() + expiresIn * 1000);
            setTokenExpiryTime(expiryTime);
            localStorage.setItem('spotify_token_expiry', expiryTime.toISOString());
            
            window.history.replaceState({}, document.title, window.location.pathname);
          } else {
            console.error('Failed to exchange code for token:', data.error);
          }
        })
        .catch(error => {
          console.error('Error exchanging code for token:', error);
        });
      return;
    }
    const token = localStorage.getItem('spotify_token');
    const expiryStr = localStorage.getItem('spotify_token_expiry');
    if (token) {
      setSpotifyToken(token);
      if (expiryStr) {
        const expiry = new Date(expiryStr);
        setTokenExpiryTime(expiry);
        
        // Check if token is expired
        if (expiry < new Date()) {
          localStorage.removeItem('spotify_token');
          localStorage.removeItem('spotify_token_expiry');
          setSpotifyToken(null);
          setTokenExpiryTime(null);
          toast({
            title: "Spotify Token Expired",
            description: "Your Spotify connection has expired (8-hour limit). Please reconnect.",
            duration: 5000,
          });
        }
      }
    }
  }, []);

  // When token changes, check validity
  useEffect(() => {
    if (!spotifyToken) {
      setSpotifyUserProfile(null);
      return;
    }
    let isMounted = true;
    checkSpotifyTokenValidity(spotifyToken).then(valid => {
      if (!valid && isMounted) {
        localStorage.removeItem('spotify_token');
        setSpotifyToken(null);
        setSpotifyUserId(null);
        setSpotifyUserProfile(null);
      }
    });
    return () => { isMounted = false; };
  }, [spotifyToken]);

  // REMOVED: recent-recommendations API call - no backend endpoint exists

  // Add state for cross-domain recommendations
  const [crossDomainRecs, setCrossDomainRecs] = useState<{
    top_artists: string[],
    top_artists_with_images: Array<{
      id: string;
      name: string;
      image: string | null;
      genres: string[];
      popularity: number;
      followers: number;
    }>,
    recommendations_by_domain: Record<string, any[]>,
    total_domains: number
  } | null>(null);
  const [crossDomainLoading, setCrossDomainLoading] = useState(false);
  const [crossDomainProgress, setCrossDomainProgress] = useState(0);
  const [crossDomainStatus, setCrossDomainStatus] = useState('');
  const [crossDomainCurrentArtist, setCrossDomainCurrentArtist] = useState('');
  const [crossDomainCurrentDomain, setCrossDomainCurrentDomain] = useState('');

  // Fetch cross-domain recommendations when spotifyToken is set
  const fetchRef = useRef(false);
  const intervalRefs = useRef<{progress?: NodeJS.Timeout, fallback?: NodeJS.Timeout}>({});
  
  useEffect(() => {
    if (spotifyToken && !fetchRef.current && !crossDomainLoading) {
      fetchRef.current = true;
      setCrossDomainLoading(true);
      setCrossDomainProgress(0);
      setCrossDomainStatus('starting');
      setCrossDomainCurrentArtist('');
      setCrossDomainCurrentDomain('');
      
                // Start fallback progress immediately in case backend progress fails
    let immediateFallbackProgress = 0;
    const fallbackInterval = setInterval(() => {
      immediateFallbackProgress = Math.min(immediateFallbackProgress + 1, 85);
      setCrossDomainProgress(immediateFallbackProgress);
        }, 1000);
    intervalRefs.current.fallback = fallbackInterval;
      
      // Real progress tracking for initial load with fallback
      let fallbackProgress = 0;
      let pollCount = 0;
      const maxPolls = 300; // Maximum 5 minutes of polling (300 * 1000ms)
      
      const progressInterval = setInterval(async () => {
        intervalRefs.current.progress = progressInterval;
      pollCount++;
      
      // Stop polling if we've exceeded the maximum time
      if (pollCount > maxPolls) {
        console.log(`[FRONTEND] Initial load - Stopping progress polling - exceeded maximum time`);
        clearInterval(progressInterval);
        clearInterval(fallbackInterval);
        intervalRefs.current.progress = undefined;
        intervalRefs.current.fallback = undefined;
        setCrossDomainLoading(false);
        setCrossDomainProgress(0);
        setCrossDomainStatus('');
        return;
      }
        try {
          if (spotifyUserId) {
            console.log(`[FRONTEND] Initial load - Polling progress for user: ${spotifyUserId}`);
            const progressRes = await fetch(`http://localhost:5500/crossdomain-progress/${spotifyUserId}`);
            console.log(`[FRONTEND] Initial load - Progress response status: ${progressRes.status}`);
            
            if (progressRes.ok) {
              const progressData = await progressRes.json();
              console.log(`[FRONTEND] Initial load - Progress data:`, progressData);
              
              setCrossDomainProgress(progressData.percentage);
              setCrossDomainStatus(progressData.status);
              setCrossDomainCurrentArtist(progressData.current_artist);
              setCrossDomainCurrentDomain(progressData.current_domain);
              
              // Stop polling if completed or error
              if (progressData.status === 'completed' || progressData.status === 'error') {
                console.log(`[FRONTEND] Initial load - Stopping progress polling - status: ${progressData.status}`);
                clearInterval(progressInterval);
                clearInterval(fallbackInterval);
                intervalRefs.current.progress = undefined;
                intervalRefs.current.fallback = undefined;
                return; // Exit early to prevent further processing
              }
            } else {
              console.log(`[FRONTEND] Initial load - Progress response not ok, using fallback`);
              // Fallback progress if backend progress fails
              fallbackProgress = Math.min(fallbackProgress + 2, 85);
              setCrossDomainProgress(fallbackProgress);
              setCrossDomainStatus('processing');
            }
          } else {
            console.log(`[FRONTEND] Initial load - No spotifyUserId available`);
          }
        } catch (error) {
          console.error('[FRONTEND] Initial load - Error fetching progress:', error);
          // Fallback progress if network error
          fallbackProgress = Math.min(fallbackProgress + 2, 85);
          setCrossDomainProgress(fallbackProgress);
          setCrossDomainStatus('processing');
        }
      }, 1000); // Poll every 1000ms instead of 500ms to reduce load
      
      fetch('http://localhost:5500/crossdomain-recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          spotify_token: spotifyToken,
          limit: 5, // Get 5 recommendations per domain for faster response
        }),
      })
        .then(res => res.json())
        .then(data => {
          // Complete the progress
                  clearInterval(progressInterval);
        intervalRefs.current.progress = undefined;
        setCrossDomainProgress(100);
        
        // Small delay to show 100% completion
        setTimeout(() => {
          if (data.error) {
            console.error('Cross-domain recommendations error:', data.error);
            setCrossDomainRecs(null);
          } else {
            setCrossDomainRecs(data);
          }
          setCrossDomainLoading(false);
          setCrossDomainProgress(0);
          clearInterval(fallbackInterval);
          intervalRefs.current.fallback = undefined;
        }, 500);
        })
        .catch(error => {
          console.error('Failed to fetch cross-domain recommendations:', error);
          setCrossDomainRecs(null);
          clearInterval(progressInterval);
          clearInterval(fallbackInterval);
          intervalRefs.current.progress = undefined;
          intervalRefs.current.fallback = undefined;
          setCrossDomainLoading(false);
          setCrossDomainProgress(0);
        });
    }
    
    // Cleanup function to clear intervals when component unmounts
    return () => {
      if (intervalRefs.current.progress) {
        clearInterval(intervalRefs.current.progress);
      }
      if (intervalRefs.current.fallback) {
        clearInterval(intervalRefs.current.fallback);
      }
      intervalRefs.current = {};
    };
  }, [spotifyToken]);

  // Add state for details modal
  const [selectedRec, setSelectedRec] = useState<any | null>(null);
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  
  // Add state for showing more recommendations
  const [showAllRecommendations, setShowAllRecommendations] = useState<{[key: string]: boolean}>({});
  
  // Add refs for horizontal scrolling
  const scrollRefs = useRef<{[key: string]: HTMLDivElement | null}>({});
  
  // Scroll functions for horizontal navigation
  const scrollLeft = (domain: string) => {
    const container = scrollRefs.current[domain];
    if (container) {
      container.scrollBy({ left: -300, behavior: 'smooth' });
    }
  };
  
  const scrollRight = (domain: string) => {
    const container = scrollRefs.current[domain];
    if (container) {
      container.scrollBy({ left: 300, behavior: 'smooth' });
    }
  };

  // CLEAN RETURN BLOCK
  const location = useLocation();
  // Handler for input send that checks Spotify connection
  const handleInputSendWithSpotifyCheck = (value: string) => {
    if (!spotifyToken) {
      toast({
        title: "Not Connected to Spotify",
        description: (
          <div className="flex items-center gap-3">
            <img src="/cat/404caty.jpg" alt="Not Connected" className="w-12 h-12 rounded-full border-2 border-black" />
            <span className="font-comic text-lg text-black">😼 Caty says: You want music? Connect Spotify, human. I'm a cat, not a magician.</span>
          </div>
        ),
        duration: 4000,
      });
      return;
    }
    handleSendMessage(value);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Only show on home page */}
      {location.pathname === '/' && messages.length === 0 && spotifyToken && spotifyUserProfile && (
        <div className="fixed top-2 left-2 sm:top-4 sm:left-4 z-50 flex items-center gap-2 bg-white border-2 border-black px-3 py-1.5 sm:px-4 sm:py-2 rounded-full font-comic font-bold shadow comic-shadow text-sm sm:text-base">
          <Avatar className="w-8 h-8 border-2 border-black">
            {spotifyUserProfile.avatar ? (
              <AvatarImage src={spotifyUserProfile.avatar} alt={spotifyUserProfile.name} />
            ) : (
              <span className="w-8 h-8 flex items-center justify-center bg-gray-200 rounded-full">?</span>
            )}
          </Avatar>
          <span className="truncate max-w-[120px] sm:max-w-[180px]">{spotifyUserProfile.name}</span>
        </div>
      )}

      {location.pathname === '/' && (
        <>
          <div className="fixed top-2 right-2 sm:top-4 sm:right-4 z-50 flex flex-col sm:flex-row gap-2">
            {/* Only show Connect/Disconnect button if credentials are set */}
            {clientId && clientSecret ? (
              <button
                className={`border-2 border-black px-3 py-1.5 sm:px-4 sm:py-2 rounded-full font-comic font-bold shadow comic-shadow text-sm sm:text-base transition ${spotifyToken ? 'bg-green-200 text-green-900 hover:bg-red-400 hover:text-white' : 'bg-black text-white hover:bg-green-700'}`}
                onClick={async () => {
                  if (spotifyToken) {
                    // Disconnect - Full logout with token revocation
                    try {
                      // Call backend to revoke the token
                      const logoutResponse = await fetch('http://localhost:5500/logout', {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                          access_token: spotifyToken,
                          client_id: clientId,
                          client_secret: clientSecret
                        })
                      });
                      
                      const logoutData = await logoutResponse.json();
                      console.log('Logout response:', logoutData);
                    } catch (error) {
                      console.log('Token revocation failed, but proceeding with logout:', error);
                    }
                    
                    // Clear ALL browser storage to force re-authentication
                    localStorage.clear();
                    sessionStorage.clear();
                    
                    // Clear cookies related to Spotify
                    document.cookie.split(";").forEach(function(c) { 
                      document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); 
                    });
                    
                    // Clear local state
                    setSpotifyToken(null);
                    setSpotifyUserId(null);
                    setSpotifyUserProfile(null);
                    setCrossDomainRecs(null);
                    setCrossDomainLoading(false);
                    setCrossDomainProgress(0);
                    setCrossDomainStatus('');
                    setCrossDomainCurrentArtist('');
                    setCrossDomainCurrentDomain('');
                    setMessages([]);
                    setIsLoading(false);
                    
                    // Clear any cached data
                    try {
                      await fetch('http://localhost:5500/clear-cache', {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json',
                        }
                      });
                    } catch (error) {
                      console.log('Cache clearing failed:', error);
                    }
                    
                    // Force a complete page reload to clear any remaining state
                    window.location.reload();
                  } else {
                    // Connect with standard redirect URI but force dialog
                    const res = await fetch('http://localhost:5500/spotify-auth-url?redirect_uri=http://127.0.0.1:8080/callback');
                    const data = await res.json();
                    window.location.href = data.auth_url;
                  }
                }}
                disabled={checkingToken}
              >
                {checkingToken
                  ? 'Checking...'
                  : spotifyToken
                    ? 'Disconnect Spotify'
                    : 'Connect to Spotify'}
              </button>
            ) : null}
            
            {/* Token expiry warning */}
            {spotifyToken && tokenExpiryTime && (
              <div className="text-xs text-gray-600 bg-yellow-100 px-2 py-1 rounded border border-yellow-300">
                ⏰ Token expires: {tokenExpiryTime.toLocaleTimeString()} ({Math.round((tokenExpiryTime.getTime() - Date.now()) / (1000 * 60))} min left)
              </div>
            )}
            
            {/* Always show credentials button */}
            <button
              className="border-2 border-black px-3 py-1.5 sm:px-4 sm:py-2 rounded-full font-comic font-bold shadow comic-shadow text-sm sm:text-base bg-yellow-200 text-black hover:bg-yellow-300"
              onClick={() => setShowCredsModal(true)}
            >
              Set Spotify Credentials
            </button>



          </div>

          {/* Modal for entering credentials */}
          <Dialog open={showCredsModal} onOpenChange={setShowCredsModal}>
            <DialogContent className="bg-white border-4 border-black comic-shadow max-w-md mx-auto p-6">
              <DialogHeader>
                <DialogTitle className="text-xl font-bold text-black comic-shadow text-center">
                  Enter Spotify Client Credentials
                </DialogTitle>
              </DialogHeader>
              <div className="flex flex-col gap-4 mt-4 max-h-[60vh] overflow-y-auto pr-2">
                <label className="font-bold text-black">Client ID</label>
                <Input
                  value={clientId}
                  onChange={e => setClientId(e.target.value)}
                  placeholder="Enter your Spotify Client ID"
                  className="border-2 border-black"
                />
                <label className="font-bold text-black">Client Secret</label>
                <Input
                  value={clientSecret}
                  onChange={e => setClientSecret(e.target.value)}
                  placeholder="Enter your Spotify Client Secret"
                  type="password"
                  className="border-2 border-black"
                />
                {/* Improved testing credentials section with cat and copy buttons */}
                <div className="flex flex-col items-center bg-yellow-50 border-2 border-dashed border-black rounded-lg p-4 mt-4">
                  <img src="/cat/404caty.jpg" alt="Testing Cat" className="w-16 h-16 rounded-full border-2 border-black mb-2" />
                  <div className="text-center font-comic text-lg font-bold text-black mb-1">
                    Need test credentials?
                  </div>
                  <div className="text-center font-comic text-base text-black mb-2">
                    😼 Cat says:<br />
                    "Don't have your own Spotify keys? No problem!<br />
                    Copy these to try the app right <b>meow</b>."
                  </div>
                  <div className="w-full flex flex-col gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-black">Client ID:</span>
                      <span className="font-mono text-xs bg-white border border-black rounded px-2 py-1 select-all">
                        5b5e4ceb834347e6a6c3b998cfaf0088
                      </span>
                      <button
                        className="bg-yellow-200 border-2 border-black rounded px-2 py-1 text-xs font-bold hover:bg-yellow-300"
                        onClick={() => navigator.clipboard.writeText('5b5e4ceb834347e6a6c3b998cfaf0088')}
                      >
                        Copy
                      </button>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-black">Client Secret:</span>
                      <span className="font-mono text-xs bg-white border border-black rounded px-2 py-1 select-all">
                        9c9aadd2b18e49859df887e5e9cc6ede
                      </span>
                      <button
                        className="bg-yellow-200 border-2 border-black rounded px-2 py-1 text-xs font-bold hover:bg-yellow-300"
                        onClick={() => navigator.clipboard.writeText('9c9aadd2b18e49859df887e5e9cc6ede')}
                      >
                        Copy
                      </button>
                    </div>
                  </div>
                  <div className="text-xs text-gray-600 mt-2 text-center">
                    (For your own playlists, use your own Spotify keys!)
                  </div>
                </div>

                <div className="flex justify-end gap-2 mt-4">
                  <Button onClick={() => setShowCredsModal(false)} variant="secondary" className="border-2 border-black">Cancel</Button>
                  <Button onClick={handleSaveCreds} className="bg-green-500 hover:bg-green-600 text-white font-bold border-2 border-black">Save</Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>


        </>
      )}

      <div className="w-full px-2 sm:px-6 py-4 sm:py-8 mt-20 sm:mt-24 overflow-x-hidden">
        {messages.length === 0 ? (
          <>
            {/* Hero Section - Enhanced Comic Theme */}
            <div className="max-w-6xl mx-auto text-center py-3 sm:py-4 lg:py-8 xl:py-12 font-comic">
              <ComicText fontSize={4.5} className="mb-3 sm:mb-4 lg:mb-6">
                What's your vibe today?
              </ComicText>
              <div className="comic-speech-bubble p-3 sm:p-4 lg:p-6 bg-yellow-50 mb-6 sm:mb-8 lg:mb-10 xl:mb-12">
                <p className="text-text-secondary text-sm sm:text-base lg:text-lg xl:text-xl font-semibold leading-relaxed font-comic">
                  Tell me your mood, activity, or the feeling you want to capture, and I'll create the perfect playlist for you.
                </p>
              </div>

              {/* Suggestion Buttons - Enhanced Comic Styling */}
              <div className="flex flex-wrap justify-center gap-1.5 sm:gap-2 lg:gap-3 xl:gap-4 mb-4 sm:mb-6 lg:mb-8 xl:mb-10">
                {['lovable', 'late night drive', 'study session', 'feeling blue', 'party time'].map((suggestion, index) => (
                  <button
                    key={suggestion}
                    onClick={() => handleInputSendWithSpotifyCheck(suggestion)}
                    className="comic-button px-3 py-2 sm:px-4 sm:py-2 lg:px-6 lg:py-3 xl:px-8 xl:py-3 text-sm sm:text-base lg:text-lg xl:text-xl uppercase tracking-wide"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Area - Consistent max-width */}
            <div className="max-w-6xl mx-auto mb-4 sm:mb-6 lg:mb-8 xl:mb-12">
              <PlaceholdersAndVanishInput
                placeholders={placeholders}
                onChange={handleInputChange}
                onSendMessage={handleInputSendWithSpotifyCheck}
                loading={isLoading}
              />
            </div>

            {/* Caty Recommendations - Enhanced Comic Theme */}
            <div className="max-w-6xl mx-auto font-comic">
              <div className="flex flex-col items-center mb-4 sm:mb-6 lg:mb-8 xl:mb-12">
                <div className="relative mb-3 sm:mb-4">
                  <img
                    src="/cat/catyrecommdation.png"
                    alt="Caty Recommendation"
                    className="w-20 h-20 sm:w-24 sm:h-24 lg:w-32 lg:h-32 rounded-full comic-border-lg comic-shadow-lg mb-2 sm:mb-3 comic-pulse"
                  />
                  <div className="absolute -top-1 -right-1 sm:-top-2 sm:-right-2 w-6 h-6 sm:w-8 sm:h-8 bg-yellow-400 comic-border rounded-full flex items-center justify-center font-comic font-bold text-xs sm:text-sm comic-shadow">
                    ⭐
                  </div>
                </div>
                <ComicText fontSize={3.2} className="text-center mb-3">
                  Caty Know Everything: Trust Caty!
                </ComicText>
                <div className="comic-speech-bubble p-3 sm:p-4 bg-yellow-50 comic-border">
                  <p className="text-sm sm:text-base lg:text-lg font-comic font-bold text-black">
                    😼 "I've analyzed your music taste and found the PERFECT recommendations for you!"
                  </p>
                </div>
              </div>

                            {/* Comic Strip Section - Enhanced Comic Design - Only show when NOT connected to Spotify */}
              {!spotifyToken && (
                <div className="comic-recommendation-card mb-6 sm:mb-8">
                  <div className="comic-section-header mb-4 sm:mb-6">
                    <div className="flex items-center justify-center gap-3 sm:gap-4">
                      <img src="/cat/404caty.jpg" alt="Comic Cat" className="w-10 h-10 sm:w-12 sm:h-12 rounded-full comic-border comic-shadow" />
                      <div className="text-center">
                        <ComicText fontSize={2.4} className="uppercase tracking-wider text-base sm:text-lg lg:text-xl">
                          Caty's Comic Adventure
                        </ComicText>
                        <div className="font-comic font-bold text-base sm:text-lg lg:text-xl text-black mt-2">
                          😼 "Check out my comic strip!"
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Comic Strip Grid - Traditional Comic Page Layout */}
                  <div className="space-y-4 sm:space-y-6 lg:space-y-8 max-w-7xl mx-auto">
                    {/* Row 1: Panels 1 & 2 */}
                    <div className="grid grid-cols-2 gap-4 sm:gap-6 lg:gap-8">
                      <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                        <img 
                          src="/comic/1.png" 
                          alt="Comic Panel 1" 
                          className="w-full h-56 sm:h-64 lg:h-[600px] object-cover"
                          onError={(e) => {
                            e.currentTarget.src = '/cat/404caty.jpg';
                          }}
                        />
                      </div>
                      <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                        <img 
                          src="/comic/2.png" 
                          alt="Comic Panel 2" 
                          className="w-full h-56 sm:h-64 lg:h-[600px] object-cover"
                          onError={(e) => {
                            e.currentTarget.src = '/cat/404caty.jpg';
                          }}
                        />
                      </div>
                    </div>

                    {/* Row 2: Panel 3 (Landscape - Full Width) */}
                    <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                      <img 
                        src="/comic/3.png" 
                        alt="Comic Panel 3" 
                        className="w-full h-40 sm:h-48 lg:h-[40rem] object-cover"
                        onError={(e) => {
                          e.currentTarget.src = '/cat/404caty.jpg';
                        }}
                      />
                    </div>

                    {/* Row 3: Panels 4, 5, 6 */}
                    <div className="grid grid-cols-3 gap-4 sm:gap-6 lg:gap-8">
                      <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                        <img 
                          src="/comic/4.png" 
                          alt="Comic Panel 4" 
                          className="w-full h-40 sm:h-48 lg:h-96 object-cover"
                          onError={(e) => {
                            e.currentTarget.src = '/cat/404caty.jpg';
                          }}
                        />
                      </div>
                      <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                        <img 
                          src="/comic/5.png" 
                          alt="Comic Panel 5" 
                          className="w-full h-40 sm:h-48 lg:h-96 object-cover"
                          onError={(e) => {
                            e.currentTarget.src = '/cat/404caty.jpg';
                          }}
                        />
                      </div>
                      <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                        <img 
                          src="/comic/6.png" 
                          alt="Comic Panel 6" 
                          className="w-full h-40 sm:h-48 lg:h-96 object-cover"
                          onError={(e) => {
                            e.currentTarget.src = '/cat/404caty.jpg';
                          }}
                        />
                      </div>
                    </div>

                    {/* Row 4: Panels 7, 8 */}
                    <div className="grid grid-cols-2 gap-4 sm:gap-6 lg:gap-8">
                      <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                        <img 
                          src="/comic/7.png" 
                          alt="Comic Panel 7" 
                          className="w-full h-56 sm:h-64 lg:h-[600px] object-cover"
                          onError={(e) => {
                            e.currentTarget.src = '/cat/404caty.jpg';
                          }}
                        />
                      </div>
                      <div className="comic-panel bg-white border-4 border-black comic-shadow-lg overflow-hidden">
                        <img 
                          src="/comic/8.png" 
                          alt="Comic Panel 8" 
                          className="w-full h-56 sm:h-64 lg:h-[600px] object-cover"
                          onError={(e) => {
                            e.currentTarget.src = '/cat/404caty.jpg';
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Comic Strip Footer */}
                  <div className="mt-4 sm:mt-6 text-center">
                    <div className="comic-speech-bubble p-3 sm:p-4 bg-yellow-50 comic-border inline-block">
                      <p className="text-sm sm:text-base lg:text-lg font-comic font-bold text-black">
                        😼 "Each panel tells a story! Connect to Spotify to see your personalized recommendations!"
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Loading State - Enhanced Comic Styling with Progress Bar */}
              {crossDomainLoading ? (
                <div className="comic-recommendation-card mb-6 sm:mb-8">
                  {/* Mobile Layout - Vertical */}
                  <div className="sm:hidden flex flex-col items-center text-center gap-3 mb-4">
                    <div className="relative">
                      <img src="/cat/404caty.jpg" alt="Loading Cat" className="w-16 h-16 rounded-full comic-border animate-pulse" />
                      <div className="absolute -top-1 -right-1 w-6 h-6 bg-blue-400 comic-border rounded-full flex items-center justify-center">
                        <div className="w-3 h-3 bg-white rounded-full animate-ping"></div>
                      </div>
                    </div>
                    <div className="w-full">
                      <span className="font-bold text-black font-comic text-lg">Caty is thinking...</span>
                      <div className="text-gray-600 text-sm font-comic mt-1">
                        😼 "Hold on, human! I'm analyzing your music taste to find movies, books, and shows you'll love..."
                      </div>
                    </div>
                    <div className="text-center">
                      <span className="font-bold text-black font-comic text-lg">
                        {crossDomainProgress > 100 ? '102.475135245687%' : `${crossDomainProgress}%`}
                      </span>
                      {crossDomainProgress > 100 && (
                        <div className="text-xs text-red-500 font-comic">(Caty's math skills! 😸)</div>
                      )}
                    </div>
                  </div>

                  {/* Desktop Layout - Horizontal */}
                  <div className="hidden sm:flex items-center gap-4 mb-4">
                    <div className="relative">
                      <img src="/cat/404caty.jpg" alt="Loading Cat" className="w-12 h-12 rounded-full comic-border animate-pulse" />
                      <div className="absolute -top-1 -right-1 w-6 h-6 bg-blue-400 comic-border rounded-full flex items-center justify-center">
                        <div className="w-3 h-3 bg-white rounded-full animate-ping"></div>
                      </div>
                    </div>
                    <div className="flex-1">
                      <span className="font-bold text-black font-comic text-lg">Caty is thinking...</span>
                      <div className="text-gray-600 text-sm font-comic mt-1">
                        😼 "Hold on, human! I'm analyzing your music taste to find movies, books, and shows you'll love..."
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-black font-comic text-lg">
                        {crossDomainProgress > 100 ? '102.475135245687%' : `${crossDomainProgress}%`}
                      </span>
                      {crossDomainProgress > 100 && (
                        <div className="text-xs text-red-500 font-comic">(Caty's math skills! 😸)</div>
                      )}
                    </div>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="mb-4">
                    <div className="w-full bg-gray-200 rounded-full h-3 sm:h-4 border-2 border-black comic-shadow overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-yellow-400 via-orange-400 to-red-400 transition-all duration-500 ease-out"
                        style={{ width: `${crossDomainProgress}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Real Progress Steps - Caty's Backend Adventure */}
                  <div className="space-y-2">
                    {/* Mobile Layout - Vertical with better text wrapping */}
                    <div className="sm:hidden space-y-3">
                      <div className={`flex flex-col gap-1 text-xs font-comic ${crossDomainStatus.includes('starting') || crossDomainStatus.includes('profile_loaded') || crossDomainStatus.includes('preferences_loaded') || crossDomainStatus.includes('artists_analyzed') ? 'text-blue-600' : crossDomainProgress >= 20 ? 'text-green-600' : 'text-gray-500'}`}>
                        <div className="flex items-center gap-2">
                          <span>{crossDomainStatus.includes('starting') || crossDomainStatus.includes('profile_loaded') || crossDomainStatus.includes('preferences_loaded') || crossDomainStatus.includes('artists_analyzed') ? '🔄' : crossDomainProgress >= 20 ? '✅' : '⏳'}</span>
                          <span className="font-bold">Starting Caty's analysis...</span>
                        </div>
                        {(crossDomainStatus.includes('starting') || crossDomainStatus.includes('profile_loaded') || crossDomainStatus.includes('preferences_loaded') || crossDomainStatus.includes('artists_analyzed')) && (
                          <div className="text-xs ml-6">(Caty is warming up! 😸)</div>
                        )}
                      </div>
                      
                      <div className={`flex flex-col gap-1 text-xs font-comic ${crossDomainCurrentArtist ? 'text-blue-600' : crossDomainProgress >= 40 ? 'text-green-600' : 'text-gray-500'}`}>
                        <div className="flex items-center gap-2">
                          <span>{crossDomainCurrentArtist ? '🎵' : crossDomainProgress >= 40 ? '✅' : '⏳'}</span>
                          <span className="font-bold">
                            {crossDomainCurrentArtist ? 
                              `Processing ${crossDomainCurrentArtist}...` : 
                              'Processing artists...'
                            }
                          </span>
                        </div>
                        {crossDomainCurrentArtist && (
                          <div className="text-xs ml-6">(Caty loves {crossDomainCurrentArtist}! 🎤)</div>
                        )}
                      </div>
                      
                      <div className={`flex flex-col gap-1 text-xs font-comic ${crossDomainCurrentDomain ? 'text-blue-600' : crossDomainProgress >= 60 ? 'text-green-600' : 'text-gray-500'}`}>
                        <div className="flex items-center gap-2">
                          <span>{crossDomainCurrentDomain ? '🎬' : crossDomainProgress >= 60 ? '✅' : '⏳'}</span>
                          <span className="font-bold">
                            {crossDomainCurrentDomain ? 
                              `Finding ${crossDomainCurrentDomain} recommendations...` : 
                              'Finding recommendations...'
                            }
                          </span>
                        </div>
                        {crossDomainCurrentDomain && (
                          <div className="text-xs ml-6">(Caty's {crossDomainCurrentDomain} math: {crossDomainProgress}% = {crossDomainProgress + 2.475135245687}%! 🧮)</div>
                        )}
                      </div>
                      
                      <div className={`flex flex-col gap-1 text-xs font-comic ${crossDomainStatus.includes('aggregating_results') || crossDomainStatus.includes('sorting_results') ? 'text-blue-600' : crossDomainProgress >= 80 ? 'text-green-600' : 'text-gray-500'}`}>
                        <div className="flex items-center gap-2">
                          <span>{crossDomainStatus.includes('aggregating_results') || crossDomainStatus.includes('sorting_results') ? '📊' : crossDomainProgress >= 80 ? '✅' : '⏳'}</span>
                          <span className="font-bold">Aggregating results...</span>
                        </div>
                        {(crossDomainStatus.includes('aggregating_results') || crossDomainStatus.includes('sorting_results')) && (
                          <div className="text-xs ml-6">(Almost there! Caty is organizing everything! 📊)</div>
                        )}
                      </div>
                      
                      <div className={`flex flex-col gap-1 text-xs font-comic ${crossDomainStatus.includes('finalizing') || crossDomainStatus.includes('completed') ? 'text-green-600' : 'text-gray-500'}`}>
                        <div className="flex items-center gap-2">
                          <span>{crossDomainStatus.includes('finalizing') || crossDomainStatus.includes('completed') ? '✅' : '⏳'}</span>
                          <span className="font-bold">Finalizing your recommendations...</span>
                        </div>
                        {(crossDomainStatus.includes('finalizing') || crossDomainStatus.includes('completed')) && (
                          <div className="text-xs ml-6">(100% = 102.475135245687% in Caty math! 🎯)</div>
                        )}
                      </div>
                      
                      <div className={`flex flex-col gap-1 text-xs font-comic ${crossDomainProgress >= 90 ? 'text-green-600' : 'text-gray-500'}`}>
                        <div className="flex items-center gap-2">
                          <span>{crossDomainProgress >= 90 ? '🎉' : '⏳'}</span>
                          <span className="font-bold">Preparing your personalized mix...</span>
                        </div>
                        {crossDomainProgress >= 90 && (
                          <div className="text-xs ml-6">(Caty's masterpiece is almost ready! 🎨)</div>
                        )}
                      </div>
                    </div>

                    {/* Desktop Layout - Horizontal */}
                    <div className="hidden sm:block space-y-2">
                      <div className={`flex items-center gap-2 text-sm font-comic ${crossDomainStatus.includes('starting') || crossDomainStatus.includes('profile_loaded') || crossDomainStatus.includes('preferences_loaded') || crossDomainStatus.includes('artists_analyzed') ? 'text-blue-600' : crossDomainProgress >= 20 ? 'text-green-600' : 'text-gray-500'}`}>
                        <span>{crossDomainStatus.includes('starting') || crossDomainStatus.includes('profile_loaded') || crossDomainStatus.includes('preferences_loaded') || crossDomainStatus.includes('artists_analyzed') ? '🔄' : crossDomainProgress >= 20 ? '✅' : '⏳'}</span>
                        <span>Starting Caty's analysis... {(crossDomainStatus.includes('starting') || crossDomainStatus.includes('profile_loaded') || crossDomainStatus.includes('preferences_loaded') || crossDomainStatus.includes('artists_analyzed')) && '(Caty is warming up! 😸)'}</span>
                      </div>
                      <div className={`flex items-center gap-2 text-sm font-comic ${crossDomainCurrentArtist ? 'text-blue-600' : crossDomainProgress >= 40 ? 'text-green-600' : 'text-gray-500'}`}>
                        <span>{crossDomainCurrentArtist ? '🎵' : crossDomainProgress >= 40 ? '✅' : '⏳'}</span>
                        <span>
                          {crossDomainCurrentArtist ? 
                            `Processing ${crossDomainCurrentArtist}...` : 
                            'Processing artists...'
                          }
                          {crossDomainCurrentArtist && ` (Caty loves ${crossDomainCurrentArtist}! 🎤)`}
                        </span>
                      </div>
                      <div className={`flex items-center gap-2 text-sm font-comic ${crossDomainCurrentDomain ? 'text-blue-600' : crossDomainProgress >= 60 ? 'text-green-600' : 'text-gray-500'}`}>
                        <span>{crossDomainCurrentDomain ? '🎬' : crossDomainProgress >= 60 ? '✅' : '⏳'}</span>
                        <span>
                          {crossDomainCurrentDomain ? 
                            `Finding ${crossDomainCurrentDomain} recommendations...` : 
                            'Finding recommendations...'
                          }
                          {crossDomainCurrentDomain && ` (Caty's ${crossDomainCurrentDomain} math: {crossDomainProgress}% = {crossDomainProgress + 2.475135245687}%! 🧮)`}
                        </span>
                      </div>
                      <div className={`flex items-center gap-2 text-sm font-comic ${crossDomainStatus.includes('aggregating_results') || crossDomainStatus.includes('sorting_results') ? 'text-blue-600' : crossDomainProgress >= 80 ? 'text-green-600' : 'text-gray-500'}`}>
                        <span>{crossDomainStatus.includes('aggregating_results') || crossDomainStatus.includes('sorting_results') ? '📊' : crossDomainProgress >= 80 ? '✅' : '⏳'}</span>
                        <span>Aggregating results... {(crossDomainStatus.includes('aggregating_results') || crossDomainStatus.includes('sorting_results')) && '(Almost there! Caty is organizing everything! 📊)'}</span>
                      </div>
                      <div className={`flex items-center gap-2 text-sm font-comic ${crossDomainStatus.includes('finalizing') || crossDomainStatus.includes('completed') ? 'text-green-600' : 'text-gray-500'}`}>
                        <span>{crossDomainStatus.includes('finalizing') || crossDomainStatus.includes('completed') ? '✅' : '⏳'}</span>
                        <span>Finalizing your recommendations... {(crossDomainStatus.includes('finalizing') || crossDomainStatus.includes('completed')) && '(100% = 102.475135245687% in Caty math! 🎯)'}</span>
                      </div>
                      <div className={`flex items-center gap-2 text-sm font-comic ${crossDomainProgress >= 90 ? 'text-green-600' : 'text-gray-500'}`}>
                        <span>{crossDomainProgress >= 90 ? '🎉' : '⏳'}</span>
                        <span>Preparing your personalized mix... {crossDomainProgress >= 90 && "(Caty's masterpiece is almost ready! 🎨)"}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : crossDomainRecs ? (
                <div className="comic-recommendation-card mb-8">
                                      {/* Recent Artists Section - Enhanced Comic Design */}
                    <div className="mb-6 sm:mb-8">
                      <div className="comic-section-header mb-4 sm:mb-6">
                        {/* Mobile Layout - Vertical */}
                        <div className="sm:hidden flex flex-col items-center text-center gap-3">
                          <img src="/cat/404caty.jpg" alt="Cat Recs" className="w-16 h-16 rounded-full comic-border" />
                          <div>
                                                      <h3 className="font-comic font-bold text-xl sm:text-2xl lg:text-3xl text-black">Your Recent Artists</h3>
                          <p className="text-base sm:text-lg lg:text-xl text-black font-comic">Artists you've been listening to</p>
                          </div>
                          <button
                            onClick={refreshCrossDomainRecs}
                            disabled={crossDomainLoading}
                            className="comic-button p-3 disabled:opacity-50 disabled:cursor-not-allowed"
                            title={crossDomainLoading ? "Getting fresh recommendations..." : "Refresh recommendations"}
                          >
                            <span className={`text-xl ${crossDomainLoading ? "animate-spin" : ""}`}>🔄</span>
                          </button>
                        </div>

                        {/* Desktop Layout - Horizontal */}
                        <div className="hidden sm:flex items-center justify-center gap-4">
                          <img src="/cat/404caty.jpg" alt="Cat Recs" className="w-12 h-12 rounded-full comic-border" />
                          <div>
                            <h3 className="font-comic font-bold text-2xl sm:text-3xl lg:text-4xl text-black">Your Recent Artists</h3>
                            <p className="text-lg sm:text-xl lg:text-2xl text-black font-comic">Artists you've been listening to</p>
                          </div>
                          <button
                            onClick={refreshCrossDomainRecs}
                            disabled={crossDomainLoading}
                            className="comic-button p-2 disabled:opacity-50 disabled:cursor-not-allowed"
                            title={crossDomainLoading ? "Getting fresh recommendations..." : "Refresh recommendations"}
                          >
                            <span className={`text-lg ${crossDomainLoading ? "animate-spin" : ""}`}>🔄</span>
                          </button>
                        </div>
                      </div>

                    {/* Artists Grid - Enhanced Comic Design */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
                      {crossDomainRecs.top_artists_with_images && crossDomainRecs.top_artists_with_images.slice(0, 6).map((artist, index) => (
                        <div key={artist.id || artist.name} className="comic-artist-card group">
                          {/* Mobile Layout - Vertical */}
                          <div className="sm:hidden flex flex-col items-center text-center">
                            <div className="relative mb-2">
                              {artist.image ? (
                                <img
                                  src={artist.image}
                                  alt={artist.name}
                                  className="w-16 h-16 rounded-full comic-border object-cover"
                                  onError={(e) => {
                                    e.currentTarget.src = '/cat/404caty.jpg';
                                  }}
                                />
                              ) : (
                                <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full comic-border flex items-center justify-center text-white font-comic font-bold text-lg">
                                  {artist.name.charAt(0)}
                                </div>
                              )}
                              <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-gradient-to-br from-yellow-400 to-orange-400 rounded-full comic-border flex items-center justify-center text-black font-comic font-bold text-xs comic-shadow">
                                {index + 1}
                              </div>
                            </div>
                            <div className="w-full">
                              <div className="font-comic font-bold text-black text-sm truncate group-hover:text-blue-600 transition-colors mb-1">
                                {artist.name}
                              </div>
                              <div className="text-xs text-gray-500 font-comic flex items-center justify-center gap-1 mb-2">
                                <span>🎵</span>
                                <span>Top artist</span>
                                {artist.followers > 0 && (
                                  <span>• {(artist.followers / 1000000).toFixed(1)}M followers</span>
                                )}
                              </div>
                              {artist.genres && artist.genres.length > 0 && (
                                <div className="flex flex-wrap justify-center gap-1">
                                  {artist.genres.slice(0, 2).map(genre => (
                                    <span key={genre} className="bg-yellow-200 text-black px-2 py-1 rounded-full text-xs font-comic font-bold comic-border">
                                      {genre}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Desktop Layout - Horizontal */}
                          <div className="hidden sm:flex items-center gap-3">
                            <div className="relative">
                              {artist.image ? (
                                <img
                                  src={artist.image}
                                  alt={artist.name}
                                  className="w-12 h-12 rounded-full comic-border object-cover"
                                  onError={(e) => {
                                    e.currentTarget.src = '/cat/404caty.jpg';
                                  }}
                                />
                              ) : (
                                <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full comic-border flex items-center justify-center text-white font-comic font-bold text-sm">
                                  {artist.name.charAt(0)}
                                </div>
                              )}
                              <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-gradient-to-br from-yellow-400 to-orange-400 rounded-full comic-border flex items-center justify-center text-black font-comic font-bold text-xs comic-shadow">
                                {index + 1}
                              </div>
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="font-comic font-bold text-black text-sm truncate group-hover:text-blue-600 transition-colors">
                                {artist.name}
                              </div>
                              <div className="text-xs text-gray-500 font-comic flex items-center gap-1">
                                <span>🎵</span>
                                <span>Top artist</span>
                                {artist.followers > 0 && (
                                  <span>• {(artist.followers / 1000000).toFixed(1)}M followers</span>
                                )}
                              </div>
                              {artist.genres && artist.genres.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {artist.genres.slice(0, 2).map(genre => (
                                    <span key={genre} className="bg-yellow-200 text-black px-2 py-0.5 rounded-full text-xs font-comic font-bold comic-border">
                                      {genre}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recommendations Grid - Consistent layout */}
                  {crossDomainRecs.recommendations_by_domain && Object.entries(crossDomainRecs.recommendations_by_domain).map(([domain, items]) => (
                    items && items.length > 0 && (
                      <div key={domain} className="mb-8 sm:mb-12">
                        {/* Section Header - Enhanced Comic Design */}
                        <div className="relative mb-6 sm:mb-8 lg:mb-12">
                          <div className="comic-section-header max-w-2xl mx-auto">
                            <div className="flex items-center justify-center gap-3 sm:gap-4">
                              <img src="/cat/404caty.jpg" alt={`Cat ${domain}`} className="w-10 h-10 sm:w-12 sm:h-12 rounded-full comic-border comic-shadow" />
                              <div className="text-center">
                                <ComicText fontSize={1.8} className="uppercase tracking-wider text-sm sm:text-base">
                                  {`${domain.charAt(0).toUpperCase()}${domain.slice(1)}s`}
                                </ComicText>
                                <div className="font-comic font-bold text-sm sm:text-base lg:text-lg text-black mt-1">
                                  😼 Caty's Picks!
                                </div>
                              </div>
                              <div className="absolute -top-1 -right-1 sm:-top-2 sm:-right-2 bg-white comic-border rounded-full w-6 h-6 sm:w-8 sm:h-8 flex items-center justify-center font-comic font-black text-sm sm:text-lg comic-shadow">
                                {items.length}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Mobile Grid Layout / Desktop Horizontal Scrollable Layout */}
                        <div className="relative overflow-x-hidden">
                          {/* Scroll indicator - hidden on mobile */}
                          <div className="hidden sm:block comic-scroll-indicator absolute top-2 right-2 z-10">
                            ← Scroll → {items.length} items
                          </div>
                          
                          {/* Scroll buttons - hidden on mobile */}
                          <button
                            onClick={() => scrollLeft(domain)}
                            className="hidden sm:flex absolute left-2 top-1/2 transform -translate-y-1/2 bg-white comic-border rounded-full w-10 h-10 items-center justify-center comic-shadow hover:bg-yellow-100 transition-all z-10 font-comic font-bold"
                          >
                            ←
                          </button>
                          <button
                            onClick={() => scrollRight(domain)}
                            className="hidden sm:flex absolute right-2 top-1/2 transform -translate-y-1/2 bg-white comic-border rounded-full w-10 h-10 items-center justify-center comic-shadow hover:bg-yellow-100 transition-all z-10 font-comic font-bold"
                          >
                            →
                          </button>
                          
                          {/* Mobile Grid Layout */}
                          <div className="sm:hidden grid grid-cols-1 gap-4 mb-4">
                            {items
                              .filter((item, idx, arr) => {
                                return arr.findIndex(i => i.name === item.name && (i.id === item.id || (!i.id && !item.id))) === idx;
                              })
                              .slice(0, showAllRecommendations[domain] ? undefined : 6) // Show 6 on mobile initially
                              .map((item, idx) => (
                                <div key={idx}
                                  className="comic-recommendation-card overflow-hidden cursor-pointer group relative"
                                  onClick={() => {
                                    setSelectedRec(item);
                                    setSelectedDomain(domain);
                                    setDetailsOpen(true);
                                  }}>

                                  {/* Mobile Card Layout */}
                                  <div className="flex h-24 sm:h-28">
                                    {/* Image Section - Smaller on mobile */}
                                    <div className="relative w-20 sm:w-24 h-full bg-gradient-to-br from-yellow-200 via-orange-200 to-red-200 border-r-4 border-black overflow-hidden flex-shrink-0">
                                      {item.properties?.image?.url ? (
                                        <img
                                          src={item.properties.image.url}
                                          alt={item.name}
                                          className="w-full h-full object-contain bg-white"
                                          onError={(e) => {
                                            e.currentTarget.src = '/cat/404caty.jpg';
                                          }}
                                        />
                                      ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                          <span className="text-2xl sm:text-3xl">
                                            {domain === 'book' ? '📚' :
                                              domain === 'movie' ? '🎬' :
                                                domain === 'TV show' ? '📺' :
                                                  domain === 'podcast' ? '🎧' :
                                                    domain === 'music artist' ? '🎵' : '🎭'}
                                          </span>
                                        </div>
                                      )}

                                      {/* Relevance score badge */}
                                      {item.relevance_score !== undefined && item.relevance_score > 0 && (
                                        <div className="absolute top-1 left-1 bg-blue-400 border-2 border-black rounded-full px-1.5 py-0.5 text-xs font-comic font-black comic-shadow">
                                          ⭐ {item.relevance_score.toFixed(1)}
                                        </div>
                                      )}

                                      {(item.properties?.release_year || item.properties?.publication_year) && (
                                        <div className="absolute top-1 right-1 bg-blue-400 border-2 border-black rounded-full px-1.5 py-0.5 text-xs font-comic font-black text-white comic-shadow">
                                          {item.properties.release_year || item.properties.publication_year}
                                        </div>
                                      )}
                                    </div>

                                    {/* Content Section - Compact mobile layout */}
                                    <div className="flex-1 p-3 flex flex-col justify-between">
                                      <div className="font-comic font-black text-black text-sm leading-tight mb-1 line-clamp-2">
                                        {item.name}
                                      </div>

                                      <div className="flex-1 flex flex-col justify-center gap-1">
                                        {item.selected_tag && (
                                          <div className="bg-purple-300 border-2 border-black rounded px-1.5 py-0.5 text-xs font-comic font-black text-center comic-shadow">
                                            #{item.selected_tag}
                                          </div>
                                        )}

                                        {item.source_artist && (
                                          <div className="text-xs text-gray-700 flex items-center gap-1 font-comic font-bold">
                                            <span>🎵</span>
                                            <span className="truncate">From: {item.source_artist}</span>
                                          </div>
                                        )}

                                        {/* Domain-specific info */}
                                        <div className="text-xs text-gray-700 flex items-center gap-1 font-comic font-bold">
                                          {domain === 'movie' && item.properties?.duration && (
                                            <>
                                              <span>⏰</span>
                                              <span>{Math.floor(item.properties.duration / 60)}h {item.properties.duration % 60}m</span>
                                            </>
                                          )}
                                          {domain === 'book' && item.properties?.page_count && (
                                            <>
                                              <span>📖</span>
                                              <span>{item.properties.page_count} pages</span>
                                            </>
                                          )}
                                          {domain === 'TV show' && (item.properties?.release_year && item.properties?.finale_year) && (
                                            <>
                                              <span>📅</span>
                                              <span>{item.properties.release_year} - {item.properties.finale_year}</span>
                                            </>
                                          )}
                                          {domain === 'podcast' && item.properties?.episode_count && (
                                            <>
                                              <span>🎙️</span>
                                              <span>{item.properties.episode_count} eps</span>
                                            </>
                                          )}
                                        </div>
                                      </div>

                                      {/* Bottom action */}
                                      <div className="mt-1 text-center">
                                        <div className="bg-yellow-300 border-2 border-black rounded-full px-2 py-0.5 text-xs font-comic font-black comic-shadow group-hover:bg-yellow-400 transition-colors">
                                          TAP TO EXPLORE!
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                          </div>

                          {/* Desktop Horizontal Scrollable Layout */}
                          <div 
                            ref={(el) => scrollRefs.current[domain] = el}
                            className="hidden sm:flex gap-4 sm:gap-6 overflow-x-auto pb-4 scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100 hover:scrollbar-thumb-gray-600 px-12 overflow-hidden sm:overflow-x-auto"
                          >
                            {items
                              .filter((item, idx, arr) => {
                                // Remove duplicates based on name and id
                                return arr.findIndex(i => i.name === item.name && (i.id === item.id || (!i.id && !item.id))) === idx;
                              })
                              .slice(0, showAllRecommendations[domain] ? undefined : 10) // Show 10 initially, all if expanded
                              .map((item, idx) => {
                                // Debug: Log item to check for duplicates
                                console.log(`${domain} item ${idx}:`, item.name, item.id || 'no-id');
                                return (
                                <div key={idx}
                                  className="comic-recommendation-card overflow-hidden cursor-pointer group relative flex-shrink-0 w-56 sm:w-60 lg:w-64"
                                  onClick={() => {
                                    setSelectedRec(item);
                                    setSelectedDomain(domain);
                                    setDetailsOpen(true);
                                  }}>

                                  {/* Card content - maintain existing structure but with consistent spacing */}
                                  <div className="relative h-full">
                                    {/* Image Section - Full visibility with proper aspect ratio */}
                                    <div className="relative h-36 sm:h-40 lg:h-44 bg-gradient-to-br from-yellow-200 via-orange-200 to-red-200 border-b-4 border-black overflow-hidden">
                                      {item.properties?.image?.url ? (
                                        <img
                                          src={item.properties.image.url}
                                          alt={item.name}
                                          className="w-full h-full object-contain bg-white"
                                          onError={(e) => {
                                            e.currentTarget.src = '/cat/404caty.jpg';
                                          }}
                                        />
                                      ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                          <span className="text-4xl sm:text-5xl">
                                            {domain === 'book' ? '📚' :
                                              domain === 'movie' ? '🎬' :
                                                domain === 'TV show' ? '📺' :
                                                  domain === 'podcast' ? '🎧' :
                                                    domain === 'music artist' ? '🎵' : '🎭'}
                                          </span>
                                        </div>
                                      )}

                                      {/* Relevance score badge - shows relevance instead of popularity */}
                                      {item.relevance_score !== undefined && item.relevance_score > 0 && (
                                        <div className="absolute top-2 left-2 bg-blue-400 border-2 border-black rounded-full px-2 py-1 text-xs font-comic font-black comic-shadow">
                                          ⭐ {item.relevance_score.toFixed(1)}
                                        </div>
                                      )}

                                      {(item.properties?.release_year || item.properties?.publication_year) && (
                                        <div className="absolute top-2 right-2 bg-blue-400 border-2 border-black rounded-full px-2 py-1 text-xs font-comic font-black text-white comic-shadow">
                                          {item.properties.release_year || item.properties.publication_year}
                                        </div>
                                      )}

                                      <div className="absolute inset-0 bg-yellow-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300 flex items-center justify-center">
                                        <span className="text-4xl sm:text-5xl animate-pulse">⭐</span>
                                      </div>
                                    </div>

                                    {/* Content Section - Adjusted for larger image */}
                                    <div className="p-3 sm:p-4 h-28 sm:h-32 lg:h-36 flex flex-col justify-between">
                                      <div className="font-comic font-black text-black text-sm sm:text-base leading-tight mb-2 line-clamp-2">
                                        {item.name}
                                      </div>

                                      <div className="flex-1 flex flex-col justify-center gap-2">
                                        {item.selected_tag && (
                                          <div className="bg-purple-300 border-2 border-black rounded-lg px-2 py-1 text-xs font-comic font-black text-center comic-shadow">
                                            #{item.selected_tag}
                                          </div>
                                        )}

                                        {item.source_artist && (
                                          <div className="text-xs text-gray-700 flex items-center justify-center gap-1 font-comic font-bold">
                                            <span>🎵</span>
                                            <span className="truncate">From: {item.source_artist}</span>
                                          </div>
                                        )}

                                        {/* Domain-specific info with consistent styling */}
                                        <div className="text-xs text-gray-700 flex items-center justify-center gap-1 font-comic font-bold">
                                          {domain === 'movie' && item.properties?.duration && (
                                            <>
                                              <span>⏰</span>
                                              <span>{Math.floor(item.properties.duration / 60)}h {item.properties.duration % 60}m</span>
                                            </>
                                          )}
                                          {domain === 'book' && item.properties?.page_count && (
                                            <>
                                              <span>📖</span>
                                              <span>{item.properties.page_count} pages</span>
                                            </>
                                          )}
                                          {domain === 'TV show' && (item.properties?.release_year && item.properties?.finale_year) && (
                                            <>
                                              <span>📅</span>
                                              <span>{item.properties.release_year} - {item.properties.finale_year}</span>
                                            </>
                                          )}
                                          {domain === 'podcast' && item.properties?.episode_count && (
                                            <>
                                              <span>🎙️</span>
                                              <span>{item.properties.episode_count} eps</span>
                                            </>
                                          )}
                                        </div>
                                      </div>

                                      {/* Bottom action - consistent styling */}
                                      <div className="mt-2 text-center">
                                        <div className="bg-yellow-300 border-2 border-black rounded-full px-3 py-1 text-xs font-comic font-black comic-shadow group-hover:bg-yellow-400 transition-colors">
                                          CLICK TO EXPLORE!
                                        </div>
                                      </div>
                                    </div>

                                    {/* Hover effect - consistent positioning */}
                                    <div className="absolute -top-2 -left-2 bg-white border-3 border-black rounded-full w-8 h-8 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300 comic-shadow transform group-hover:scale-110">
                                      <span className="text-lg">💥</span>
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>


                        
                        {/* Section Divider - Enhanced Comic Design */}
                        <div className="mt-6 sm:mt-8 lg:mt-12 flex items-center justify-center">
                          <div className="bg-gradient-to-r from-transparent via-black to-transparent h-1 sm:h-2 w-full max-w-md comic-shadow"></div>
                          <div className="mx-3 sm:mx-4 bg-yellow-400 comic-border rounded-full w-8 h-8 sm:w-10 sm:h-10 flex items-center justify-center font-comic font-black comic-shadow-lg comic-pulse">
                            ⭐
                          </div>
                          <div className="bg-gradient-to-r from-black via-transparent to-transparent h-1 sm:h-2 w-full max-w-md comic-shadow"></div>
                        </div>
                      </div>
                    )
                  ))}

                  {/* Footer - Enhanced Comic Design */}
                  <div className="mt-8 sm:mt-12">
                    {crossDomainRecs.total_domains && crossDomainRecs.total_domains > 0 && (
                      <div className="comic-recommendation-card mb-6">
                        <div className="text-center font-comic">
                          <ComicText fontSize={1.4} className="mb-3">
                            🎉 Caty's Recommendation Summary
                          </ComicText>
                          <div className="text-sm text-gray-700 mb-2">
                            Found <span className="font-bold text-black">{crossDomainRecs.total_domains}</span> categories with{' '}
                            <span className="font-bold text-black">
                              {Object.values(crossDomainRecs.recommendations_by_domain || {}).reduce((total: number, items: any[]) => total + items.length, 0)}
                            </span>{' '}
                            total recommendations
                          </div>
                          <div className="text-xs text-gray-600">
                            Each category shows top 10 most relevant items based on your music taste!
                          </div>
                        </div>
                      </div>
                    )}

                    {(!crossDomainRecs.recommendations_by_domain || Object.keys(crossDomainRecs.recommendations_by_domain).length === 0) && (
                      <div className="comic-speech-bubble p-4 text-center text-gray-600 font-comic">
                        😼 "Hmm, I couldn't find cross-domain recommendations right now. Maybe try connecting to Spotify first?"
                      </div>
                    )}
                  </div>
                </div>
              ) : spotifyToken ? (
                <div className="comic-recommendation-card mb-8">
                  <div className="flex items-center gap-4 mb-4">
                    <img src="/cat/404caty.jpg" alt="Error Cat" className="w-12 h-12 rounded-full comic-border" />
                    <div>
                      <span className="font-bold text-black font-comic text-lg">Caty couldn't find recommendations</span>
                      <div className="text-gray-600 text-sm font-comic mt-1">
                        😼 "Something went wrong while getting your cross-domain recommendations. Maybe try refreshing?"
                      </div>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </>
        ) : (
          // Chat Messages - Consistent max-width with home page
          <div className="max-w-6xl mx-auto space-y-0 mb-4 sm:mb-6 lg:mb-8 overflow-x-hidden">
            <button
              className="fixed left-2 sm:left-4 lg:left-8 top-2 sm:top-4 lg:top-8 px-3 py-1.5 sm:px-4 sm:py-2 lg:px-6 lg:py-2 rounded-xl border-4 border-black bg-white text-black font-comic text-xs sm:text-sm lg:text-base shadow comic-shadow hover:bg-yellow-100 transition-all z-50"
              onClick={() => window.location.href = '/'}
            >
              ← Back to Home
            </button>
            {messages.map(message => (
              <ChatBubble key={message.id} message={message.text} isUser={message.isUser} timestamp={message.timestamp}>
                {message.playlistData && <PlaylistResponse data={message.playlistData} />}
              </ChatBubble>
            ))}
            {isLoading && <TypingIndicator />}
          </div>
        )}

        {/* Modal - Proper scrolling and layout */}
        <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
          <DialogContent className="bg-white border-4 border-black comic-shadow max-w-2xl mx-auto p-0 max-h-[90vh] flex flex-col">
            {selectedRec && (
              <>
                {/* Fixed Header */}
                <div className="p-4 sm:p-6 border-b-2 border-black bg-yellow-50">
                  <div className="flex items-center gap-4">
                    {selectedRec.properties?.image?.url ? (
                      <img
                        src={selectedRec.properties.image.url}
                        alt={selectedRec.name}
                        className="w-16 h-16 sm:w-20 sm:h-20 rounded-lg border-2 border-black object-cover flex-shrink-0"
                        onError={e => { e.currentTarget.src = '/cat/404caty.jpg'; }}
                      />
                    ) : (
                      <div className="w-16 h-16 sm:w-20 sm:h-20 bg-gradient-to-br from-yellow-200 to-orange-300 rounded-lg border-2 border-black flex items-center justify-center flex-shrink-0">
                        <span className="text-2xl sm:text-3xl">
                          {selectedDomain === 'book' ? '📚' :
                            selectedDomain === 'movie' ? '🎬' :
                              selectedDomain === 'TV show' ? '📺' :
                                selectedDomain === 'podcast' ? '🎧' :
                                  selectedDomain === 'music artist' ? '🎵' : '🎭'}
                        </span>
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <ComicText fontSize={1.3} className="mb-2 line-clamp-2">{selectedRec.name}</ComicText>
                      <div className="flex flex-wrap gap-2">
                        {(selectedRec.properties?.release_year || selectedRec.properties?.publication_year) && (
                          <div className="text-xs bg-gray-100 border border-black rounded px-2 py-1 font-bold font-comic">
                            {selectedRec.properties.release_year || selectedRec.properties.publication_year}
                          </div>
                        )}
                        {selectedRec.relevance_score !== undefined && selectedRec.relevance_score > 0 && (
                          <div className="text-xs bg-blue-100 border border-black rounded px-2 py-1 font-bold font-comic">
                            ⭐ {selectedRec.relevance_score.toFixed(1)} relevance
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Scrollable Content */}
                <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
                  {/* Selected Tag and Source Artist */}
                  <div className="flex flex-wrap gap-2">
                    {selectedRec.selected_tag && (
                      <Badge className="text-xs bg-blue-100 text-blue-800 border border-blue-300 font-comic">
                        #{selectedRec.selected_tag}
                      </Badge>
                    )}
                    {selectedRec.source_artist && (
                      <div className="text-xs text-gray-500 flex items-center gap-1 font-comic bg-gray-50 border border-gray-300 rounded px-2 py-1">
                        <span>🎵</span>
                        <span>Based on: {selectedRec.source_artist}</span>
                      </div>
                    )}
                  </div>

                  {/* Description */}
                  {(selectedRec.properties?.description || selectedRec.properties?.short_description) && (
                    <div className="bg-gray-50 border-2 border-gray-200 rounded-lg p-3">
                      <h4 className="font-comic font-bold text-sm mb-2 text-black">Description</h4>
                      <div
                        className="text-sm text-gray-700 font-comic leading-relaxed"
                        dangerouslySetInnerHTML={{
                          __html: (selectedRec.properties?.description || selectedRec.properties?.short_description || '').replace(/\n/g, '<br />')
                        }}
                      />
                    </div>
                  )}

                  {/* Tags */}
                  {selectedRec.tags && selectedRec.tags.length > 0 && (
                    <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-3">
                      <h4 className="font-comic font-bold text-sm mb-2 text-black">Tags</h4>
                      <div className="flex flex-wrap gap-2">
                        {(() => {
                          const genreTags = selectedRec.tags.filter((tag: any) => tag.type && tag.type.includes('genre'));
                          const otherTags = selectedRec.tags.filter((tag: any) => !(tag.type && tag.type.includes('genre')));
                          const shownTags = [...genreTags, ...otherTags].slice(0, 8);
                          return shownTags.map((tag: any) => (
                            <Badge key={tag.id} className="bg-yellow-100 border border-black text-black text-xs font-bold font-comic">
                              {tag.name}
                            </Badge>
                          ));
                        })()}
                      </div>
                    </div>
                  )}

                  {/* Details */}
                  <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3">
                    <h4 className="font-comic font-bold text-sm mb-2 text-black">Details</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-gray-700 font-comic">
                      {selectedDomain === 'movie' && selectedRec.properties?.duration && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <span className="font-bold">Duration:</span> {Math.floor(selectedRec.properties.duration / 60)}h {selectedRec.properties.duration % 60}m
                        </div>
                      )}
                      {selectedDomain === 'book' && selectedRec.properties?.page_count && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <span className="font-bold">Pages:</span> {selectedRec.properties.page_count}
                        </div>
                      )}
                      {selectedDomain === 'book' && selectedRec.properties?.isbn13 && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <span className="font-bold">ISBN:</span> {selectedRec.properties.isbn13}
                        </div>
                      )}
                      {selectedDomain === 'podcast' && selectedRec.properties?.episode_count && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <span className="font-bold">Episodes:</span> {selectedRec.properties.episode_count}
                        </div>
                      )}
                      {selectedDomain === 'TV show' && (selectedRec.properties?.release_year && selectedRec.properties?.finale_year) && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <span className="font-bold">Years:</span> {selectedRec.properties.release_year} - {selectedRec.properties.finale_year}
                        </div>
                      )}
                      {selectedDomain === 'music artist' && selectedRec.properties?.date_of_birth && (
                        <div className="bg-white border border-gray-200 rounded p-2">
                          <span className="font-bold">Born:</span> {selectedRec.properties.date_of_birth}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* External Links */}
                  {selectedRec.external && (
                    <div className="bg-green-50 border-2 border-green-200 rounded-lg p-3">
                      <h4 className="font-comic font-bold text-sm mb-2 text-black">External Links</h4>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(selectedRec.external).map(([key, arr]: [string, any]) => (
                          arr && arr.length > 0 && arr.map((ext: any, i: number) => {
                            let url = '';
                            let icon = '🔗';
                            if (key === 'imdb') { url = `https://www.imdb.com/name/${ext.id}`; icon = '🎬'; }
                            if (key === 'goodreads') { url = `https://www.goodreads.com/book/show/${ext.id}`; icon = '📚'; }
                            if (key === 'spotify') { url = `https://open.spotify.com/artist/${ext.id}`; icon = '🎵'; }
                            if (key === 'wikidata') { url = `https://www.wikidata.org/wiki/${ext.id}`; icon = '📖'; }
                            if (key === 'musicbrainz') { url = `https://musicbrainz.org/artist/${ext.id}`; icon = '🎼'; }
                            if (key === 'lastfm') { url = `https://www.last.fm/music/${ext.id}`; icon = '🎧'; }
                            if (!url) return null;
                            return (
                              <a
                                key={key + i}
                                href={url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 bg-white border-2 border-black rounded-lg px-3 py-2 text-xs font-bold font-comic text-black hover:bg-yellow-100 transition-colors comic-shadow"
                              >
                                <span>{icon}</span>
                                <span>{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                              </a>
                            );
                          })
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Fixed Footer */}
                <div className="border-t-2 border-black p-4 bg-gray-50">
                  <div className="flex justify-center">
                    <Button
                      onClick={() => setDetailsOpen(false)}
                      className="bg-yellow-200 hover:bg-yellow-300 text-black font-bold border-2 border-black font-comic comic-shadow"
                    >
                      Close
                    </Button>
                  </div>
                </div>
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>

      {/* Multi-step loader */}
      <MultiStepLoader loadingStates={loadingStates} loading={isLoading} duration={5000} loop={false} />
    </div>
  );
};

export default MusicDashboard;