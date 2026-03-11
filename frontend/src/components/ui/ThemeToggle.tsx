"use client";

import * as React from "react";
import { Moon, Sun, Monitor } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";

export function ThemeToggle() {
    const { setTheme, theme, resolvedTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    // Prevent hydration mismatch
    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) {
        return <div className="h-9 w-9" />; // Placeholder to avoid layout shift
    }

    const toggleTheme = () => {
        if (theme === "light") setTheme("dark");
        else if (theme === "dark") setTheme("system");
        else setTheme("light");
    };

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="h-9 w-9 rounded-xl hover:bg-muted transition-all duration-300 relative overflow-hidden group"
            title={`Current theme: ${theme}. Click to cycle.`}
        >
            <div className="relative h-5 w-5">
                <Sun className="h-5 w-5 transition-all duration-500 rotate-0 scale-100 dark:-rotate-90 dark:scale-0 absolute inset-0" />
                <Moon className="h-5 w-5 transition-all duration-500 rotate-90 scale-0 dark:rotate-0 dark:scale-100 absolute inset-0" />
            </div>
            <span className="sr-only">Toggle theme</span>
            
            {/* Visual indicator of cycle state */}
            <div className="absolute bottom-1 right-1 flex gap-0.5">
                <div className={`w-1 h-1 rounded-full transition-colors ${theme === 'light' ? 'bg-indigo-500' : 'bg-border'}`} />
                <div className={`w-1 h-1 rounded-full transition-colors ${theme === 'dark' ? 'bg-indigo-500' : 'bg-border'}`} />
                <div className={`w-1 h-1 rounded-full transition-colors ${theme === 'system' ? 'bg-indigo-500' : 'bg-border'}`} />
            </div>
        </Button>
    );
}
