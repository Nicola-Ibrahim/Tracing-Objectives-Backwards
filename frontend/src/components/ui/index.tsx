import React from "react";

interface CardProps {
    children: React.ReactNode;
    className?: string;
    title?: string;
}

export function Card({ children, className = "", title }: CardProps) {
    return (
        <div className={`glass-panel p-6 ${className}`}>
            {title && <h3 className="text-lg font-semibold mb-4 text-slate-800">{title}</h3>}
            {children}
        </div>
    );
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "secondary" | "outline" | "ghost";
    isLoading?: boolean;
}

export function Button({
    children,
    variant = "primary",
    isLoading,
    className = "",
    disabled,
    ...props
}: ButtonProps) {
    const variants = {
        primary: "bg-primary text-white shadow-lg shadow-primary/25 hover:bg-primary/90",
        secondary: "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50",
        outline: "bg-transparent border-2 border-primary text-primary hover:bg-primary/5",
        ghost: "bg-transparent text-slate-500 hover:bg-slate-100 hover:text-slate-700",
    };

    return (
        <button
            className={`px-4 py-2 rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 ${variants[variant]} ${className}`}
            disabled={isLoading || disabled}
            {...props}
        >
            {isLoading && (
                <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
            )}
            {children}
        </button>
    );
}

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label?: string;
}

export function Input({ label, className = "", ...props }: InputProps) {
    return (
        <div className="space-y-1.5 w-full">
            {label && <label className="text-sm font-semibold text-slate-700 ml-1">{label}</label>}
            <input
                className={`w-full px-4 py-2.5 rounded-xl border border-slate-200 bg-white/50 focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all text-sm ${className}`}
                {...props}
            />
        </div>
    );
}
