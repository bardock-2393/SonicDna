import React from 'react';

interface CacheIndicatorProps {
    fromCache: boolean;
    timestamp?: string;
    className?: string;
}

const CacheIndicator: React.FC<CacheIndicatorProps> = ({
    fromCache,
    timestamp,
    className = ""
}) => {
    if (!fromCache) return null;

    return (
        <div className={`inline-flex items-center gap-2 bg-green-100 border-2 border-green-300 rounded-full px-3 py-1 text-xs font-comic font-bold text-green-800 ${className}`}>
            <span className="animate-pulse">⚡</span>
            <span>Lightning Fast!</span>
            {timestamp && (
                <span className="text-xs opacity-75">
                    {new Date(timestamp).toLocaleTimeString()}
                </span>
            )}
        </div>
    );
};

export default CacheIndicator;