import React from 'react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Database, RefreshCw, Trash2 } from 'lucide-react';
import { cacheUtils } from '../utils/cacheManager';

interface CacheStatusIndicatorProps {
  cacheKey: string;
  requestHash: string;
  onClearCache?: () => void;
  onRefresh?: () => void;
  loading?: boolean;
  showControls?: boolean;
  className?: string;
}

export default function CacheStatusIndicator({
  cacheKey,
  requestHash,
  onClearCache,
  onRefresh,
  loading = false,
  showControls = true,
  className = ''
}: CacheStatusIndicatorProps) {
  const cacheStatus = cacheUtils.getCacheStatus(cacheKey, requestHash);
  
  const getStatusInfo = () => {
    switch (cacheStatus) {
      case 'valid':
        return {
          text: '📦 From Cache',
          color: 'bg-blue-100 text-blue-800 border-blue-300',
          icon: Database
        };
      case 'expired':
        return {
          text: '⏰ Cache Expired',
          color: 'bg-yellow-100 text-yellow-800 border-yellow-300',
          icon: Database
        };
      case 'missing':
        return {
          text: '🆕 Fresh Data',
          color: 'bg-green-100 text-green-800 border-green-300',
          icon: Database
        };
      case 'error':
        return {
          text: '❌ Cache Error',
          color: 'bg-red-100 text-red-800 border-red-300',
          icon: Database
        };
      default:
        return {
          text: '⏳ Loading...',
          color: 'bg-gray-100 text-gray-800 border-gray-300',
          icon: Database
        };
    }
  };

  const statusInfo = getStatusInfo();
  const IconComponent = statusInfo.icon;

  return (
    <div className={`flex flex-col sm:flex-row items-center justify-center gap-2 ${className}`}>
      {/* Cache Status Badge */}
      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border-2 font-bold text-sm ${statusInfo.color}`}>
        <IconComponent className="w-4 h-4" />
        {statusInfo.text}
      </div>

      {/* Cache Controls */}
      {showControls && (
        <div className="flex items-center gap-2">
          {onRefresh && (
            <Button
              onClick={onRefresh}
              size="sm"
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-1 px-2 rounded border-2 border-black comic-shadow flex items-center gap-1 text-xs"
              disabled={loading}
            >
              <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Refreshing...' : 'Refresh'}
            </Button>
          )}
          
          {onClearCache && (
            <Button
              onClick={onClearCache}
              size="sm"
              variant="outline"
              className="border-2 border-black comic-shadow text-xs py-1 px-2"
              disabled={loading}
            >
              <Trash2 className="w-3 h-3" />
              Clear Cache
            </Button>
          )}
        </div>
      )}
    </div>
  );
} 