"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";

export function ThemeToggle() {
    const { setTheme, resolvedTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    // Prevent hydration mismatch
    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) {
        return <div className="h-9 w-9" />; // Placeholder to avoid layout shift
    }

    const toggleTheme = () => {
        if (resolvedTheme === "light") setTheme("dark");
        else setTheme("light");
    };

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="h-9 w-9 rounded-xl hover:bg-muted transition-all duration-300 relative overflow-hidden group"
            title={`Current theme: ${resolvedTheme}. Click to toggle.`}
        >
            <div className="relative h-5 w-5 flex items-center justify-center">
                <AnimatePresence mode="wait" initial={false}>
                    {resolvedTheme === "light" ? (
                        <motion.div
                            key="sun"
                            initial={{ scale: 0.3, rotate: -90, opacity: 0 }}
                            animate={{ scale: 1, rotate: 0, opacity: 1 }}
                            exit={{ scale: 0.3, rotate: 90, opacity: 0 }}
                            transition={{ type: "spring", stiffness: 300, damping: 20 }}
                        >
                            <Sun className="h-5 w-5 text-amber-500" />
                        </motion.div>
                    ) : (
                        <motion.div
                            key="moon"
                            initial={{ scale: 0.3, rotate: 90, opacity: 0 }}
                            animate={{ scale: 1, rotate: 0, opacity: 1 }}
                            exit={{ scale: 0.3, rotate: -90, opacity: 0 }}
                            transition={{ type: "spring", stiffness: 300, damping: 20 }}
                        >
                            <Moon className="h-5 w-5 text-indigo-400" />
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
            <span className="sr-only">Toggle theme</span>
            
            {/* Visual indicator of cycle state with a small glow effect */}
            <div className="absolute bottom-1 right-1 flex gap-0.5">
                <motion.div 
                    animate={{ 
                        scale: resolvedTheme === 'light' ? 1.2 : 1,
                        backgroundColor: resolvedTheme === 'light' ? 'var(--color-indigo-500)' : 'var(--color-border)' 
                    }}
                    className="w-1 h-1 rounded-full" 
                />
                <motion.div 
                    animate={{ 
                        scale: resolvedTheme === 'dark' ? 1.2 : 1,
                        backgroundColor: resolvedTheme === 'dark' ? 'var(--color-indigo-500)' : 'var(--color-border)' 
                    }}
                    className="w-1 h-1 rounded-full" 
                />
            </div>
        </Button>
    );
}
