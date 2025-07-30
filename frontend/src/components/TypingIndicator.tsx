import React from 'react';

const TypingIndicator: React.FC = () => {
  return (
    <div className="w-[85%] mx-auto mb-6 flex justify-start">
      <div className="bg-ai-bubble text-ai-bubble-foreground rounded-lg px-4 py-3 sm:px-6 sm:py-4 shadow-soft rounded-bl-sm">
        <div className="typing-indicator">
          <div className="typing-dot"></div>
          <div className="typing-dot"></div>
          <div className="typing-dot"></div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;