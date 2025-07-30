import React from 'react';
import { cn } from '@/lib/utils';

interface ChatBubbleProps {
  message: string;
  isUser: boolean;
  timestamp?: Date;
  children?: React.ReactNode;
  className?: string;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ 
  message, 
  isUser, 
  timestamp,
  children,
  className 
}) => {
  return (
    <div 
      className={cn(
        "chat-bubble-enter w-[80%] mx-auto mb-3 sm:mb-6",
        isUser ? "flex justify-end" : "flex justify-start",
        className
      )}
    >
      <div className={cn(
        "rounded-2xl px-3 py-2 sm:px-6 sm:py-4 max-w-[85vw] sm:max-w-80rem font-comic comic-shadow border-4 border-black",
        isUser 
          ? "bg-yellow-200 text-black rounded-br-[2rem] sm:rounded-br-[2.5rem]" 
          : "bg-white-100 text-black rounded-bl-[2rem] sm:rounded-bl-[2.5rem]"
      )}>
        <div className="mb-1 sm:mb-3">
          <p className="text-sm sm:text-lg font-extrabold leading-relaxed whitespace-pre-wrap comic-shadow">
            {message}
          </p>
        </div>
        
        {children && (
          <div className="mt-2 sm:mt-4 mb-1 sm:mb-2">
            {children}
          </div>
        )}
        
        {timestamp && (
          <div className="mt-1 sm:mt-3 text-xs font-bold text-gray-700 opacity-80 comic-shadow text-right">
            {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatBubble;