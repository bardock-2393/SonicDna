// Cache Management Utility for SoniqueDNA
// Provides centralized caching functionality for API responses

export interface CacheEntry<T> {
  data: T;
  timestamp: number;
  requestHash: string;
  expiresAt: number;
}

export interface CacheConfig {
  key: string;
  duration: number; // in milliseconds
  maxSize?: number; // maximum number of entries
}

export class CacheManager {
  private static instance: CacheManager;
  private cacheConfigs: Map<string, CacheConfig> = new Map();

  private constructor() {}

  static getInstance(): CacheManager {
    if (!CacheManager.instance) {
      CacheManager.instance = new CacheManager();
    }
    return CacheManager.instance;
  }

  // Register a cache configuration
  registerCache(config: CacheConfig): void {
    this.cacheConfigs.set(config.key, config);
  }

  // Generate a hash for request parameters
  generateRequestHash(params: Record<string, any>): string {
    const sortedParams = Object.keys(params)
      .sort()
      .reduce((result, key) => {
        const value = params[key];
        if (Array.isArray(value)) {
          result[key] = value.sort();
        } else {
          result[key] = value;
        }
        return result;
      }, {} as Record<string, any>);
    
    return btoa(JSON.stringify(sortedParams));
  }

  // Get cached data
  getCachedData<T>(cacheKey: string, requestHash: string): T | null {
    try {
      const config = this.cacheConfigs.get(cacheKey);
      if (!config) {
        console.warn(`Cache config not found for key: ${cacheKey}`);
        return null;
      }

      const cached = localStorage.getItem(cacheKey);
      if (!cached) return null;

      const cacheEntry: CacheEntry<T> = JSON.parse(cached);
      const now = Date.now();

      // Check if cache is expired
      if (now > cacheEntry.expiresAt) {
        localStorage.removeItem(cacheKey);
        return null;
      }

      // Check if request hash matches
      if (cacheEntry.requestHash !== requestHash) {
        return null;
      }

      return cacheEntry.data;
    } catch (error) {
      console.error(`Error reading cache for key ${cacheKey}:`, error);
      localStorage.removeItem(cacheKey);
      return null;
    }
  }

  // Set cached data
  setCachedData<T>(cacheKey: string, requestHash: string, data: T): void {
    try {
      const config = this.cacheConfigs.get(cacheKey);
      if (!config) {
        console.warn(`Cache config not found for key: ${cacheKey}`);
        return;
      }

      const now = Date.now();
      const cacheEntry: CacheEntry<T> = {
        data,
        timestamp: now,
        requestHash,
        expiresAt: now + config.duration
      };

      localStorage.setItem(cacheKey, JSON.stringify(cacheEntry));

      // Clean up old entries if max size is configured
      this.cleanupCache(cacheKey, config);
    } catch (error) {
      console.error(`Error writing cache for key ${cacheKey}:`, error);
    }
  }

  // Clear specific cache
  clearCache(cacheKey: string): void {
    try {
      localStorage.removeItem(cacheKey);
    } catch (error) {
      console.error(`Error clearing cache for key ${cacheKey}:`, error);
    }
  }

  // Clear all caches
  clearAllCaches(): void {
    try {
      this.cacheConfigs.forEach((config, key) => {
        localStorage.removeItem(key);
      });
    } catch (error) {
      console.error('Error clearing all caches:', error);
    }
  }

  // Get cache info
  getCacheInfo(cacheKey: string): { exists: boolean; expiresAt?: number; age?: number } | null {
    try {
      const cached = localStorage.getItem(cacheKey);
      if (!cached) return { exists: false };

      const cacheEntry: CacheEntry<any> = JSON.parse(cached);
      const now = Date.now();

      return {
        exists: true,
        expiresAt: cacheEntry.expiresAt,
        age: now - cacheEntry.timestamp
      };
    } catch (error) {
      console.error(`Error getting cache info for key ${cacheKey}:`, error);
      return null;
    }
  }

  // Get time until cache expires (in minutes)
  getTimeUntilExpiry(cacheKey: string): number {
    const info = this.getCacheInfo(cacheKey);
    if (!info || !info.expiresAt) return 0;

    const now = Date.now();
    const timeLeft = info.expiresAt - now;
    return Math.max(0, Math.ceil(timeLeft / 60000)); // Convert to minutes
  }

  // Clean up old cache entries
  private cleanupCache(cacheKey: string, config: CacheConfig): void {
    if (!config.maxSize) return;

    try {
      // This is a simplified cleanup - in a real app you might want more sophisticated cleanup
      const allKeys = Object.keys(localStorage);
      const cacheKeys = allKeys.filter(key => key.startsWith(cacheKey));
      
      if (cacheKeys.length > config.maxSize) {
        // Remove oldest entries
        const entries = cacheKeys.map(key => ({
          key,
          timestamp: this.getCacheInfo(key)?.age || 0
        })).sort((a, b) => b.timestamp - a.timestamp);

        // Remove oldest entries beyond max size
        entries.slice(config.maxSize).forEach(entry => {
          localStorage.removeItem(entry.key);
        });
      }
    } catch (error) {
      console.error(`Error cleaning up cache for key ${cacheKey}:`, error);
    }
  }

  // Check if cache is valid
  isCacheValid(cacheKey: string, requestHash: string): boolean {
    const data = this.getCachedData(cacheKey, requestHash);
    return data !== null;
  }
}

// Pre-configured cache instances for common use cases
export const crossDomainCache = CacheManager.getInstance();
crossDomainCache.registerCache({
  key: 'crossdomain_recommendations_cache',
  duration: 30 * 60 * 1000, // 30 minutes
  maxSize: 5
});

export const musicRecommendationsCache = CacheManager.getInstance();
musicRecommendationsCache.registerCache({
  key: 'music_recommendations_cache',
  duration: 15 * 60 * 1000, // 15 minutes
  maxSize: 10
});

// Utility functions for common cache operations
export const cacheUtils = {
  // Format cache expiry time for display
  formatExpiryTime(minutes: number): string {
    if (minutes < 1) return 'Expired';
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  },

  // Get cache status for display
  getCacheStatus(cacheKey: string, requestHash: string): 'valid' | 'expired' | 'missing' | 'error' {
    try {
      const cacheManager = CacheManager.getInstance();
      if (cacheManager.isCacheValid(cacheKey, requestHash)) {
        return 'valid';
      }
      
      const info = cacheManager.getCacheInfo(cacheKey);
      if (!info?.exists) {
        return 'missing';
      }
      
      return 'expired';
    } catch (error) {
      return 'error';
    }
  },

  // Clear all application caches
  clearAllCaches(): void {
    const cacheManager = CacheManager.getInstance();
    cacheManager.clearAllCaches();
  }
}; 