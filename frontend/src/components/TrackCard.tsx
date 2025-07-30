import React, { useState } from 'react';
import { Play, Pause, ExternalLink } from 'lucide-react';

interface Track {
  name: string;
  artist: string;
  album_name: string;
  release_year: string;
  album_art_url?: string;
  preview_url?: string;
  url: string;
  context_score: number;
}

interface TrackCardProps {
  track: Track;
}

const TrackCard: React.FC<TrackCardProps> = ({ track }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null);

  const handlePlayPause = () => {
    if (!track.preview_url) return;
    if (!audio) {
      const newAudio = new Audio(track.preview_url);
      newAudio.addEventListener('ended', () => setIsPlaying(false));
      setAudio(newAudio);
      newAudio.play();
      setIsPlaying(true);
    } else {
      if (isPlaying) {
        audio.pause();
        setIsPlaying(false);
      } else {
        audio.play();
        setIsPlaying(true);
      }
    }
  };

  return (
    <div className="transition-transform duration-200 hover:scale-105 hover:shadow-yellow-400 shadow-xl rounded-2xl border-4 border-black bg-yellow-100 p-2 sm:p-6 flex flex-col sm:flex-row items-center gap-2 sm:gap-6 w-full min-h-[80px] sm:min-h-[120px]">
      {/* Album Art & Play */}
      <div className="relative flex-shrink-0">
        <img
          src={track.album_art_url || '/placeholder.svg'}
          alt={`${track.album_name} cover`}
          className="w-16 h-16 sm:w-24 sm:h-24 md:w-28 md:h-28 rounded-xl object-cover border-2 border-black shadow"
        />
        {track.preview_url && (
          <button
            onClick={handlePlayPause}
            className="absolute bottom-1 right-1 w-6 h-6 sm:w-10 sm:h-10 bg-pink-200 border-2 border-black rounded-full flex items-center justify-center hover:bg-pink-300 transition-all duration-200 shadow"
          >
            {isPlaying ? (
              <Pause className="w-3 h-3 sm:w-6 sm:h-6 text-black" />
            ) : (
              <Play className="w-3 h-3 sm:w-6 sm:h-6 ml-0.5 text-black" />
            )}
          </button>
        )}
      </div>
      
      {/* Info */}
      <div className="flex-1 min-w-0 flex flex-col gap-1 text-center sm:text-left">
        <h3 className="font-extrabold text-black text-sm sm:text-xl md:text-2xl leading-tight break-words line-clamp-2">{track.name}</h3>
        <p className="text-pink-700 font-bold text-xs sm:text-lg leading-tight break-words line-clamp-2">{track.artist}</p>
        <p className="text-gray-700 text-xs sm:text-base md:text-lg font-bold truncate">{track.album_name} • {track.release_year}</p>
      </div>
      
      {/* Score & Spotify */}
      <div className="flex flex-row sm:flex-col items-center gap-1 sm:gap-3 min-w-[70px] sm:min-w-[90px]">
        <div className="text-center bg-yellow-200 border-2 border-black rounded-lg px-1 py-0.5 sm:px-3 sm:py-1">
          <div className="text-xs text-gray-700 font-bold">Score</div>
          <div className="font-extrabold text-pink-700 text-xs sm:text-lg">{track.context_score.toFixed(2)}</div>
        </div>
        <a
          href={track.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 bg-green-200 border-2 border-black hover:bg-green-300 text-black px-1.5 py-1 sm:px-3 sm:py-2 rounded-lg transition-all duration-200 font-bold shadow text-xs sm:text-base"
        >
          <span className="hidden sm:inline">Spotify</span>
          <img 
            src="/icon/spotify.png" 
            alt="Spotify" 
            className="w-3 h-3 sm:w-4 sm:h-4"
          />
          <ExternalLink className="w-2 h-2 sm:w-4 sm:h-4" />
        </a>
      </div>
    </div>
  );
};

export default TrackCard;