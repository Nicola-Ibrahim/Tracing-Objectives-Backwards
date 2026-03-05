import React from "react";
import { cn } from "@/lib/utils";

interface LogoProps {
    className?: string;
    simple?: boolean;
}

export const Logo: React.FC<LogoProps> = ({ className, simple = false }) => {
    return (
        <div className={cn("flex items-center gap-2 select-none", className)}>
            <svg
                width="40"
                height="40"
                viewBox="0 0 100 100"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="drop-shadow-sm"
            >
                {/* Background Circle / Shield */}
                <rect width="100" height="100" rx="24" fill="currentColor" className="text-slate-950" />

                {/* The Backward Arrow morphing into data nodes */}
                <path
                    d="M75 50H30M30 50L45 35M30 50L45 65"
                    stroke="white"
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />

                {/* Data Nodes (Abstract Circles) */}
                <circle cx="75" cy="40" r="4" fill="#818cf8" />
                <circle cx="75" cy="60" r="4" fill="#818cf8" />
                <circle cx="65" cy="50" r="4" fill="#c7d2fe" />

                {/* Accents */}
                <path
                    d="M20 30V70"
                    stroke="url(#grad1)"
                    strokeWidth="4"
                    strokeLinecap="round"
                    className="opacity-50"
                />

                <defs>
                    <linearGradient id="grad1" x1="20" y1="30" x2="20" y2="70" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#6366f1" />
                        <stop offset="1" stopColor="#4f46e5" />
                    </linearGradient>
                </defs>
            </svg>
            {!simple && (
                <div className="flex flex-col leading-none">
                    <span className="text-lg font-black tracking-tighter text-slate-900 uppercase italic">
                        Trace<span className="text-indigo-600">.</span>
                    </span>
                    <span className="text-[8px] font-bold text-slate-400 tracking-[0.2em] uppercase">
                        Inversion Engine
                    </span>
                </div>
            )}
        </div>
    );
};
