// Environment configuration for the frontend
export const config = {
  // Spotify Configuration
  spotify: {
    clientId: import.meta.env.VITE_SPOTIFY_CLIENT_ID || '5b5e4ceb834347e6a6c3b998cfaf0088',
    redirectUri: import.meta.env.VITE_SPOTIFY_REDIRECT_URI || 'http://localhost:5500/callback',
  },
  
  // Backend Configuration
  backend: {
    url: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8080',
  },
  
  // App Configuration
  app: {
    env: import.meta.env.VITE_APP_ENV || 'development',
  },
} as const;

// Type-safe environment variable access
export const getSpotifyClientId = () => config.spotify.clientId;
export const getSpotifyRedirectUri = () => config.spotify.redirectUri;
export const getBackendUrl = () => config.backend.url;
export const getAppEnv = () => config.app.env; 