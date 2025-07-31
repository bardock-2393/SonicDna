"use client";
import React, { useEffect, useRef, useState } from "react";
import { useMotionValueEvent, useScroll } from "motion/react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import { ComicText } from "@/components/magicui/comic-text";

export const StickyScroll = ({
  content,
  contentClassName,
}: {
  content: {
    title: string;
    description: string;
    content?: React.ReactNode | any;
  }[];
  contentClassName?: string;
}) => {
  const [activeCard, setActiveCard] = React.useState(0);
  const ref = useRef<any>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Safety check: ensure content exists and has items
  if (!content || content.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500">
        <p>No content to display</p>
      </div>
    );
  }

  // On scroll, find the card whose top is closest to the top of the container
  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    const handleScroll = () => {
      const containerTop = container.getBoundingClientRect().top;
      let minDistance = Infinity;
      let activeIdx = 0;
      cardRefs.current.forEach((card, idx) => {
        if (card) {
          const cardTop = card.getBoundingClientRect().top;
          const distance = Math.abs(cardTop - containerTop);
          if (distance < minDistance) {
            minDistance = distance;
            activeIdx = idx;
          }
        }
      });
      // If scrolled to (or near) the bottom, always highlight the last card
      if (container.scrollHeight - container.scrollTop - container.clientHeight < 2) {
        setActiveCard(Math.max(0, content.length - 1));
      } else {
        setActiveCard(Math.max(0, Math.min(activeIdx, content.length - 1)));
      }
    };
    container.addEventListener('scroll', handleScroll, { passive: true });
    // Initial call
    handleScroll();
    return () => {
      container.removeEventListener('scroll', handleScroll);
    };
  }, [content.length]);

  const backgroundColors = [
    "#0f172a", // slate-900
    "#000000", // black
    "#171717", // neutral-900
  ];
  const linearGradients = [
    "linear-gradient(to bottom right, #06b6d4, #10b981)", // cyan-500 to emerald-500
    "linear-gradient(to bottom right, #ec4899, #6366f1)", // pink-500 to indigo-500
    "linear-gradient(to bottom right, #f97316, #eab308)", // orange-500 to yellow-500
  ];

  const [backgroundGradient, setBackgroundGradient] = useState(
    linearGradients[0],
  );

  useEffect(() => {
    setBackgroundGradient(linearGradients[activeCard % linearGradients.length]);
  }, [activeCard]);

  const comicTextColors = [
    { color: "#06b6d4" }, // cyan
    { color: "#ec4899" }, // pink
    { color: "#f97316" }, // orange
    { color: "#10b981" }, // emerald
    { color: "#6366f1" }, // indigo
    { color: "#eab308" }, // yellow
  ];

  const comicTextGradients = [
    {
      backgroundImage: "linear-gradient(90deg, #06b6d4, #10b981)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
    {
      backgroundImage: "linear-gradient(90deg, #ec4899, #6366f1)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
    {
      backgroundImage: "linear-gradient(90deg, #f97316, #eab308)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
    {
      backgroundImage: "linear-gradient(90deg, #ff5e62, #ff9966)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
    {
      backgroundImage: "linear-gradient(90deg, #43cea2, #185a9d)",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
    },
  ];

  return (
    <motion.div
      style={{ background: '#fff' }}
      className="relative flex flex-col lg:flex-row h-auto lg:h-[30rem] justify-center space-y-3 lg:space-y-0 lg:space-x-10 overflow-y-auto rounded-md p-2 sm:p-4 lg:p-10 bg-white"
      ref={ref}
    >
      {/* Mobile Layout - Compact Scrollable List */}
      <div className="lg:hidden w-full space-y-3 max-h-[60vh] overflow-y-auto">
        {content.map((item, index) => (
          <div key={item.title + index} className="w-full" ref={el => cardRefs.current[index] = el}>
            {/* Mobile Compact Card */}
            <div
              style={{ background: linearGradients[index % linearGradients.length] }}
              className="w-full h-32 sm:h-40 rounded-xl border-4 border-black shadow-lg comic-shadow flex items-center justify-center overflow-hidden p-3"
            >
              {/* Compact Mobile Content */}
              <div className="flex flex-row h-full w-full items-center justify-center text-black gap-3">
                {/* Small Album Art */}
                <div className="flex-shrink-0">
                  <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-lg border-2 border-black bg-white/20 flex items-center justify-center">
                    <img 
                      src="/icon/spotify.png" 
                      alt="Spotify" 
                      className="w-8 h-8 sm:w-10 sm:h-10"
                    />
                  </div>
                </div>
                
                {/* Track Info */}
                <div className="flex-1 min-w-0 text-center">
                  <h3 className="text-sm sm:text-base font-extrabold text-black truncate">
                    {item.title}
                  </h3>
                  <p className="text-xs sm:text-sm text-black font-bold truncate">
                    {item.description}
                  </p>
                </div>
                
                {/* Compact Score */}
                <div className="flex-shrink-0">
                  <div className="text-xs text-black font-bold bg-white/80 px-2 py-1 rounded comic-shadow border border-black">
                    {Math.floor(Math.random() * 5) + 1}.{Math.floor(Math.random() * 9)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Desktop Layout - Side by Side */}
      <div className="hidden lg:flex items-start px-2 sm:px-4">
        <div className="max-w-2xl">
          {content.map((item, index) => (
            <div key={item.title + index} className="my-12 sm:my-20" ref={el => cardRefs.current[index] = el}>
              <motion.h2
                initial={{
                  opacity: 0,
                }}
                animate={{
                  opacity: activeCard === index ? 1 : 0.3,
                }}
                className="text-xl sm:text-2xl font-bold text-slate-100"
              >
                <ComicText
                  fontSize={1.8}
                  className="sm:text-2xl"
                  style={comicTextGradients[index % comicTextGradients.length]}
                >
                  {item.title}
                </ComicText>
              </motion.h2>
              <motion.p
                initial={{
                  opacity: 0,
                }}
                animate={{
                  opacity: activeCard === index ? 1 : 0.3,
                }}
                className="text-base sm:text-lg mt-6 sm:mt-10 max-w-sm text-slate-300"
              >
                <ComicText
                  fontSize={1.0}
                  className="sm:text-lg"
                  style={comicTextGradients[index % comicTextGradients.length]}
                >
                  {item.description}
                </ComicText>
              </motion.p>
            </div>
          ))}
          <div className="h-20 sm:h-40" />
        </div>
      </div>
      
      {/* Desktop Sticky Image */}
      <div
        style={{ background: backgroundGradient }}
        className={cn(
          "sticky top-4 sm:top-10 hidden lg:block h-96 sm:h-[28rem] w-[28rem] sm:w-[32rem] overflow-hidden rounded-md",
          contentClassName,
        )}
      >
        {(content && content.length > 0 && activeCard >= 0 && activeCard < content.length && content[activeCard]?.content) ?? null}
      </div>
    </motion.div>
  );
}; 