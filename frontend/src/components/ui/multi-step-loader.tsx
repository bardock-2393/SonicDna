import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "motion/react";
import { useState, useEffect } from "react";

const CheckIcon = ({ className }: { className?: string }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth={1.5}
      stroke="currentColor"
      className={cn("w-6 h-6", className)}
    >
      <path d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
    </svg>
  );
};

const CheckFilled = ({ className }: { className?: string }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className={cn("w-6 h-6", className)}
    >
      <path
        fillRule="evenodd"
        d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12Zm13.36-1.814a.75.75 0 1 0-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 0 0-1.06 1.06l2.25 2.25a.75.75 0 0 0 1.14-.094l3.75-5.25Z"
        clipRule="evenodd"
      />
    </svg>
  );
};

type LoadingState = {
  text: string;
  percentage?: number;
  showCat?: boolean;
};

const LoaderCore = ({
  loadingStates,
  value = 0,
}: {
  loadingStates: LoadingState[];
  value?: number;
}) => {
  const currentState = loadingStates[value];
  const showCat = currentState?.showCat;

  return (
    <div className="flex relative justify-start max-w-xl mx-auto flex-col mt-60 font-comic">
      {showCat ? (
        // Cat image with message overlay
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="flex flex-col items-center justify-center w-full"
        >
          <div className="relative flex flex-col items-center">
            <img 
              src="/cat/Me.png" 
              alt="Cat waiting" 
              className="w-80 h-60 object-contain rounded-lg shadow-2xl mx-auto"
            />
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-pink-100 border-2 border-pink-300 rounded-lg px-4 py-2 shadow-lg"
            >
              <span className="text-pink-700 font-bold text-lg">
                Please wait, almost there...
              </span>
            </motion.div>
          </div>
        </motion.div>
      ) : (
        // Regular loading states
        loadingStates.map((loadingState, index) => {
          const distance = Math.abs(index - value);
          const opacity = Math.max(1 - distance * 0.2, 0);

          return (
            <motion.div
              key={index}
              className={cn("text-left flex gap-3 mb-4 items-center")}
              initial={{ opacity: 0, y: -(value * 40) }}
              animate={{ opacity: opacity, y: -(value * 40) }}
              transition={{ duration: 0.5 }}
            >
              <div>
                {index > value && (
                  <CheckIcon className="text-gray-400 w-7 h-7 drop-shadow" />
                )}
                {index <= value && (
                  <CheckFilled
                    className={cn(
                      "w-7 h-7 drop-shadow",
                      value === index ? "text-pink-500" : "text-yellow-500"
                    )}
                  />
                )}
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "font-extrabold text-lg tracking-wide",
                    value === index ? "text-pink-700" : "text-black opacity-80"
                  )}
                >
                  {loadingState.text}
                </span>
                {loadingState.percentage && (
                  <span
                    className={cn(
                      "text-sm font-semibold tracking-wide",
                      value === index ? "text-pink-600" : "text-gray-600 opacity-70"
                    )}
                  >
                    {loadingState.percentage}%
                  </span>
                )}
              </div>
            </motion.div>
          );
        })
      )}
    </div>
  );
};

export const MultiStepLoader = ({
  loadingStates,
  loading,
  duration = 2000,
  loop = true,
}: {
  loadingStates: LoadingState[];
  loading?: boolean;
  duration?: number;
  loop?: boolean;
}) => {
  const [currentState, setCurrentState] = useState(0);

  useEffect(() => {
    if (!loading) {
      setCurrentState(0);
      return;
    }
    const timeout = setTimeout(() => {
      setCurrentState((prevState) =>
        loop
          ? prevState === loadingStates.length - 1
            ? 0
            : prevState + 1
          : Math.min(prevState + 1, loadingStates.length - 1)
      );
    }, duration);

    return () => clearTimeout(timeout);
  }, [currentState, loading, loop, loadingStates.length, duration]);
  
  return (
    <AnimatePresence mode="wait">
      {loading && (
        <motion.div
          initial={{
            opacity: 0,
          }}
          animate={{
            opacity: 1,
          }}
          exit={{
            opacity: 0,
          }}
          className="w-full h-full fixed inset-0 z-[100] flex items-center justify-center backdrop-blur-2xl bg-yellow-50/80"
        >
          <div className="h-96 relative flex items-center justify-center">
            <LoaderCore value={currentState} loadingStates={loadingStates} />
          </div>

          <div className="bg-gradient-to-t inset-x-0 z-20 bottom-0 bg-yellow-100 h-full absolute [mask-image:radial-gradient(900px_at_center,transparent_30%,white)]" />
        </motion.div>
      )}
    </AnimatePresence>
  );
};