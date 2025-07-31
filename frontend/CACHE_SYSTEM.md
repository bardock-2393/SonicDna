# Caching System for SoniqueDNA

## Overview

The SoniqueDNA application now includes a comprehensive caching system to improve performance and reduce API calls. The caching system is designed to cache cross-domain recommendations and other API responses to avoid repeated loading.

## Features

### 🚀 Performance Benefits
- **Reduced API Calls**: Cached data is served instantly without making new API requests
- **Faster Loading**: Subsequent visits to the same recommendations load immediately
- **Better User Experience**: No more waiting for repeated API calls

### 📦 Cache Management
- **Automatic Expiration**: Cache entries expire after 30 minutes (configurable)
- **Request Hashing**: Unique cache keys based on request parameters
- **Cache Validation**: Automatic validation of cache integrity
- **Manual Controls**: Users can clear cache or force refresh

### 🛠 Technical Features
- **localStorage Based**: Client-side caching using browser localStorage
- **TypeScript Support**: Fully typed cache management
- **Error Handling**: Graceful fallback when cache operations fail
- **Memory Management**: Automatic cleanup of old cache entries

## Implementation

### Cache Manager (`utils/cacheManager.ts`)

The central cache management system provides:

```typescript
// Initialize cache manager
const cacheManager = CacheManager.getInstance();

// Register cache configuration
cacheManager.registerCache({
  key: 'crossdomain_recommendations_cache',
  duration: 30 * 60 * 1000, // 30 minutes
  maxSize: 5
});

// Get cached data
const data = cacheManager.getCachedData(cacheKey, requestHash);

// Set cached data
cacheManager.setCachedData(cacheKey, requestHash, data);

// Clear cache
cacheManager.clearCache(cacheKey);
```

### Cross-Domain Recommendations Caching

The `CrossDomainRecommendations` component now includes:

1. **Automatic Cache Check**: Before making API calls, check if data exists in cache
2. **Cache Status Display**: Visual indicators showing cache status
3. **Manual Cache Controls**: Buttons to refresh or clear cache
4. **Cache Expiry Information**: Shows when cache will expire

### Cache Status Indicator (`components/CacheStatusIndicator.tsx`)

A reusable component that displays cache status and provides controls:

```typescript
<CacheStatusIndicator
  cacheKey="crossdomain_recommendations_cache"
  requestHash={requestHash}
  onClearCache={() => clearCache()}
  onRefresh={() => fetchData()}
  loading={loading}
/>
```

## Usage

### For Users

1. **First Visit**: Data is fetched from API and cached
2. **Subsequent Visits**: Data loads instantly from cache
3. **Cache Expired**: Automatic refresh with new API call
4. **Manual Refresh**: Use "Refresh" button to force new data
5. **Clear Cache**: Use "Clear Cache" button to remove cached data

### For Developers

#### Adding Caching to New Components

1. Import the cache manager:
```typescript
import { crossDomainCache } from '../utils/cacheManager';
```

2. Generate request hash:
```typescript
const requestHash = crossDomainCache.generateRequestHash(requestParams);
```

3. Check cache before API call:
```typescript
const cachedData = crossDomainCache.getCachedData(cacheKey, requestHash);
if (cachedData) {
  // Use cached data
  return cachedData;
}
```

4. Cache API response:
```typescript
crossDomainCache.setCachedData(cacheKey, requestHash, apiResponse);
```

#### Cache Configuration

```typescript
// Register new cache
const myCache = CacheManager.getInstance();
myCache.registerCache({
  key: 'my_api_cache',
  duration: 15 * 60 * 1000, // 15 minutes
  maxSize: 10 // Maximum 10 entries
});
```

## Cache Keys

- `crossdomain_recommendations_cache`: Cross-domain recommendations (30 min expiry)
- `music_recommendations_cache`: Music recommendations (15 min expiry)

## Cache Expiry

- **Cross-domain recommendations**: 30 minutes
- **Music recommendations**: 15 minutes
- **Automatic cleanup**: Old entries are removed automatically
- **Manual expiry**: Users can clear cache manually

## Error Handling

The caching system includes comprehensive error handling:

- **Cache Read Errors**: Automatically removes corrupted cache entries
- **Cache Write Errors**: Logs errors but doesn't break application flow
- **API Errors**: Falls back to fresh API calls when cache is invalid
- **Storage Errors**: Graceful degradation when localStorage is unavailable

## Performance Monitoring

Cache performance can be monitored through:

- **Cache Hit Rate**: Percentage of requests served from cache
- **Cache Expiry**: Time until cache expires
- **Cache Size**: Number of cached entries
- **Cache Status**: Visual indicators in the UI

## Future Enhancements

- **Server-side Caching**: Redis-based caching for better performance
- **Cache Analytics**: Detailed cache usage statistics
- **Smart Prefetching**: Preload cache for anticipated requests
- **Cache Compression**: Reduce storage size for large responses
- **Offline Support**: Cache data for offline usage

## Troubleshooting

### Common Issues

1. **Cache Not Working**: Check if localStorage is available and not full
2. **Stale Data**: Clear cache manually or wait for automatic expiry
3. **Performance Issues**: Monitor cache size and clear if necessary

### Debug Commands

```javascript
// Check cache status in browser console
localStorage.getItem('crossdomain_recommendations_cache')

// Clear all caches
localStorage.clear()

// Check cache info
crossDomainCache.getCacheInfo('crossdomain_recommendations_cache')
```

## Security Considerations

- **Client-side Only**: Cache is stored in browser localStorage
- **No Sensitive Data**: Only API responses are cached, not user credentials
- **Automatic Cleanup**: Expired cache entries are automatically removed
- **User Control**: Users can clear cache at any time 