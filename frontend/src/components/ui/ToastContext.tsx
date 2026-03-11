"use client";

import React, { createContext, useContext, useState, useCallback } from "react";

interface Toast {
    id: string;
    message: string;
    type: "success" | "error" | "info";
}

interface ToastContextType {
    showToast: (message: string, type: Toast["type"]) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function ToastProvider({ children }: { children: React.ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const showToast = useCallback((message: string, type: Toast["type"]) => {
        const id = Math.random().toString(36).substring(2, 9);
        setToasts((prev) => [...prev, { id, message, type }]);
        setTimeout(() => {
            setToasts((prev) => prev.filter((t) => t.id !== id));
        }, 5000);
    }, []);

    return (
        <ToastContext.Provider value={{ showToast }}>
            {children}
            <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
                {toasts.map((toast) => (
                    <div
                        key={toast.id}
                        className={`px-4 py-3 rounded-xl shadow-lg border animate-in slide-in-from-right-4 duration-300 flex items-center gap-3 min-w-[300px] ${toast.type === "success"
                                ? "bg-emerald-950/30 border-emerald-500/20 text-emerald-400"
                                : toast.type === "error"
                                    ? "bg-red-950/30 border-red-500/20 text-red-400"
                                    : "bg-indigo-950/30 border-indigo-500/20 text-indigo-400"
                            }`}
                    >
                        <div
                            className={`w-2 h-2 rounded-full ${toast.type === "success"
                                    ? "bg-emerald-500"
                                    : toast.type === "error"
                                        ? "bg-red-500"
                                        : "bg-indigo-500"
                                }`}
                        />
                        <span className="text-sm font-medium">{toast.message}</span>
                        <button
                            onClick={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))}
                            className="ml-auto text-muted-foreground hover:text-foreground transition-colors"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                ))}
            </div>
        </ToastContext.Provider>
    );
}

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error("useToast must be used within a ToastProvider");
    }
    return context;
}
