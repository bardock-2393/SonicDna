import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { ComicText } from "@/components/magicui/comic-text";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-yellow-50 font-comic px-4 sm:px-0">
      <div className="text-center max-w-sm sm:max-w-md w-full">
        <div className="rounded-2xl border-4 border-black bg-yellow-100 p-6 sm:p-10 shadow-xl comic-shadow">
          <ComicText fontSize={3} className="mb-4 text-black">404</ComicText>
          <p className="text-lg sm:text-xl text-gray-700 mb-6 sm:mb-8 font-bold">Oops! Page not found</p>
          <a 
            href="/" 
            className="px-6 sm:px-8 py-2.5 sm:py-3 bg-pink-200 hover:bg-pink-300 border-2 border-black rounded-full font-comic font-bold text-base sm:text-lg shadow comic-shadow transition inline-block"
          >
            Return to Home
          </a>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
