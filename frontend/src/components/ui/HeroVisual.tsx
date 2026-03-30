import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export const HeroVisual: React.FC<{ className?: string }> = ({ className }) => {
  const [particles, setParticles] = React.useState<{ id: number; duration: number; left: string; top: string; delay: number }[]>([]);

  React.useEffect(() => {
    setParticles([...Array(6)].map((_, i) => ({
      id: i,
      duration: 3 + Math.random() * 2,
      left: `${20 + Math.random() * 60}%`,
      top: `${20 + Math.random() * 60}%`,
      delay: i * 0.5
    })));
  }, []);

  return (
    <div className={cn("relative w-full aspect-square flex items-center justify-center overflow-visible", className)}>
      {/* Background Ambient Glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[140%] h-[140%] bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.12),transparent_70%)] blur-[100px] dark:bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.08),transparent_70%)]" />
      </div>

      <svg
        viewBox="0 0 600 600"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
      >
        <defs>
          <linearGradient id="indigo-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#818cf8" />
            <stop offset="100%" stopColor="#4f46e5" />
          </linearGradient>
          
          <linearGradient id="slate-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="currentColor" className="text-slate-400 dark:text-slate-600" />
            <stop offset="100%" stopColor="currentColor" className="text-slate-600 dark:text-slate-900" />
          </linearGradient>

          <filter id="glow-filter" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="12" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Background Technical Grid */}
        <g className="opacity-[0.1] dark:opacity-[0.05]">
          {[...Array(9)].map((_, i) => (
            <React.Fragment key={i}>
              <line x1={i * 75} y1="0" x2={i * 75} y2="600" stroke="currentColor" strokeWidth="0.5" className="text-muted-foreground" />
              <line x1="0" y1={i * 75} x2="600" y2={i * 75} stroke="currentColor" strokeWidth="0.5" className="text-muted-foreground" />
            </React.Fragment>
          ))}
        </g>

        {/* Bidirectional Arrow 1: Objective Path (Indigo) */}
        <motion.g
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        >
          {/* Main Path */}
          <motion.path
            d="M 100 220 C 250 220 350 380 500 380"
            stroke="url(#indigo-grad)"
            strokeWidth="14"
            strokeLinecap="round"
            className="opacity-80 dark:opacity-100"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1.5, ease: "easeInOut" }}
          />

          {/* Technical Dash Flow (Linear data movement) */}
          <motion.path
            d="M 100 220 C 250 220 350 380 500 380"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="4 12"
            className="text-white dark:text-indigo-200 opacity-40"
            animate={{ strokeDashoffset: [0, -32] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />

          {/* "Data Packet" moving along path */}
          <circle r="4" fill="white" className="drop-shadow-[0_0_8px_white]">
            <animateMotion
              dur="3s"
              repeatCount="indefinite"
              path="M 100 220 C 250 220 350 380 500 380"
            />
          </circle>
          
          {/* Sharp Head Right */}
          <motion.path
            d="M 470 350 L 510 380 L 470 410"
            fill="none"
            stroke="url(#indigo-grad)"
            strokeWidth="14"
            strokeLinecap="square"
            strokeLinejoin="miter"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1.4 }}
          />

          {/* Sharp Head Left */}
          <motion.path
             d="M 130 190 L 90 220 L 130 250"
             fill="none"
             stroke="url(#indigo-grad)"
             strokeWidth="14"
             strokeLinecap="square"
             strokeLinejoin="miter"
             initial={{ opacity: 0, x: 10 }}
             animate={{ opacity: 1, x: 0 }}
             transition={{ delay: 1.4 }}
          />
        </motion.g>

        {/* Bidirectional Arrow 2: Parameter Path (Slate/Dark) - Interlaced */}
        <motion.g
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.3 }}
        >
          {/* Main Path (Interlaced hack for visuals) */}
          <path
            d="M 100 380 C 180 380 250 320 280 300"
            stroke="url(#slate-grad)"
            strokeWidth="14"
            strokeLinecap="round"
          />
          <motion.path
            d="M 320 280 C 350 260 420 220 500 220"
            stroke="url(#slate-grad)"
            strokeWidth="14"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1 }}
          />

          {/* Technical Dash Flow (Opposite direction) */}
          <motion.path
            d="M 100 380 C 250 380 350 220 500 220"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="4 12"
            className="text-white dark:text-slate-400 opacity-20"
            animate={{ strokeDashoffset: [0, 32] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
          />

          {/* "Inverse Data Packet" moving along path */}
          <circle r="4" fill="currentColor" className="text-slate-400 dark:text-slate-200 drop-shadow-[0_0_8px_currentColor]">
            <animateMotion
              dur="4s"
              repeatCount="indefinite"
              path="M 500 220 C 350 220 250 380 100 380"
              begin="1s"
            />
          </circle>

          {/* Crossing Connection (Hidden behind/above logic) */}
          <path
             d="M 280 300 Q 300 290 320 280"
             stroke="url(#slate-grad)"
             strokeWidth="14"
             strokeLinecap="round"
             className="opacity-50"
          />

          {/* Sharp Head Right */}
          <motion.path
            d="M 470 190 L 510 220 L 470 250"
            fill="none"
            stroke="url(#slate-grad)"
            strokeWidth="14"
            strokeLinecap="square"
            strokeLinejoin="miter"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1.8 }}
          />

          {/* Sharp Head Left */}
          <motion.path
             d="M 130 350 L 90 380 L 130 410"
             fill="none"
             stroke="url(#slate-grad)"
             strokeWidth="14"
             strokeLinecap="square"
             strokeLinejoin="miter"
             initial={{ opacity: 0, x: 10 }}
             animate={{ opacity: 1, x: 0 }}
             transition={{ delay: 1.8 }}
          />
        </motion.g>

        {/* Micro Tech Accents */}
        <g className="text-indigo-500/20">
          <circle cx="300" cy="300" r="100" stroke="currentColor" strokeWidth="1" strokeDasharray="4 8" />
          <motion.circle
            cx="300" cy="300" r="100" stroke="currentColor" strokeWidth="2" strokeDasharray="40 160"
            animate={{ rotate: 360 }}
            transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
          />
        </g>
      </svg>

      {/* Floating Sparkles */}
      {particles.map((p) => (
        <motion.div
           key={p.id}
           className="absolute w-1.5 h-1.5 bg-indigo-400/50 rounded-full"
           animate={{
             y: [0, -40, 0],
             opacity: [0, 1, 0],
             scale: [0.5, 1, 0.5]
           }}
           transition={{
             duration: p.duration,
             repeat: Infinity,
             delay: p.delay
           }}
           style={{
             left: p.left,
             top: p.top
           }}
        />
      ))}
    </div>
  );
};
