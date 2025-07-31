import { useEffect, useState } from 'react';
import { ComicText } from '@/components/magicui/comic-text';
import { useNavigate } from 'react-router-dom';

const SpotifyCallback = () => {
  const navigate = useNavigate();
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const state = params.get('state');
    const error = params.get('error');
    
    if (error) {
      console.error('Spotify authorization error:', error);
      // Handle authorization errors
      setConnected(false);
      return;
    }
    
    if (code) {
      // Exchange code for access token via backend
      fetch('http://localhost:5500/exchange-token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          code,
          redirect_uri: 'http://127.0.0.1:8080/callback',
          state: state // Pass state for verification if needed
        })
      })
        .then(res => res.json())
        .then(data => {
          if (data.access_token) {
            // Clear any existing tokens first
            localStorage.removeItem('spotify_token');
            localStorage.removeItem('spotify_refresh_token');
            
            // Set new tokens
            localStorage.setItem('spotify_token', data.access_token);
            if (data.refresh_token) {
              localStorage.setItem('spotify_refresh_token', data.refresh_token);
            }
            
            setConnected(true);
            window.history.replaceState({}, document.title, window.location.pathname);
          } else {
            console.error('Failed to exchange code for token:', data.error);
            setConnected(false);
          }
        })
        .catch(error => {
          console.error('Error exchanging code for token:', error);
          setConnected(false);
        });
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-yellow-50 font-comic px-4 sm:px-0">
      <div className="rounded-2xl border-4 border-black bg-yellow-100 p-6 sm:p-10 shadow-xl comic-shadow flex flex-col items-center max-w-sm sm:max-w-md w-full">
        <ComicText fontSize={2.5} className="sm:text-3xl mb-3 sm:mb-4 text-center">
          {connected ? 'Spotify Connected!' : 'Connecting to Spotify...'}
        </ComicText>
        <p className="font-comic text-base sm:text-lg mb-6 sm:mb-8 text-green-900 font-bold text-center">
          {connected ? 'You can now return to the app.' : 'Please wait...'}
        </p>
        {connected && (
          <button
            className="px-6 sm:px-8 py-2.5 sm:py-3 bg-pink-200 hover:bg-pink-300 border-2 border-black rounded-full font-comic font-bold text-base sm:text-lg shadow comic-shadow transition"
            onClick={() => navigate('/')}
          >
            Return to Home
          </button>
        )}
      </div>
    </div>
  );
};

export default SpotifyCallback; 