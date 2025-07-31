// Environment configuration for the frontend
export const config = {
  // Spotify Configuration
  spotify: {
    clientId: import.meta.env.VITE_SPOTIFY_CLIENT_ID || '5b5e4ceb834347e6a6c3b998cfaf0088',
    redirectUri: import.meta.env.VITE_SPOTIFY_REDIRECT_URI || 'https://soniquedna.deepsantoshwar.xyz/callback',
  },
  
  // Backend Configuration
  backend: {
    url: import.meta.env.VITE_BACKEND_URL || 'https://soniquedna.deepsantoshwar.xyz/api',
  },
  
  // App Configuration
  app: {
    env: import.meta.env.VITE_APP_ENV || 'production',
  },
} as const;

// Type-safe environment variable access
export const getSpotifyClientId = () => config.spotify.clientId;
export const getSpotifyRedirectUri = () => config.spotify.redirectUri;
export const getBackendUrl = () => config.backend.url;
export const getAppEnv = () => config.app.env; 