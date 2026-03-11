import React from "react";

interface CardProps {
    children: React.ReactNode;
    className?: string;
    title?: string;
}

export function Card({ children, className = "", title }: CardProps) {
    return (
        <div className={`bg-card text-card-foreground border border-border shadow-sm rounded-2xl overflow-hidden transition-all duration-300 hover:shadow-md ${className}`}>
            {title && (
                <div className="flex items-center justify-between py-4 px-6 border-b border-border bg-muted/30">
                    <h3 className="text-sm font-bold tracking-tight">{title}</h3>
                </div>
            )}
            <div className="p-6">
                {children}
            </div>
        </div>
    );
}

export function Badge({ children, variant = "default", className = "" }: { children: React.ReactNode, variant?: "default" | "success" | "warning" | "error" | "indigo", className?: string }) {
    const variants = {
        default: "bg-secondary text-secondary-foreground",
        success: "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20",
        warning: "bg-amber-500/10 text-amber-500 border border-amber-500/20",
        error: "bg-rose-500/10 text-rose-500 border border-rose-500/20",
        indigo: "bg-indigo-500/10 text-indigo-500 border border-indigo-500/20",
    };
    return (
        <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${variants[variant]} ${className}`}>
            {children}
        </span>
    );
}

export function StatCard({ label, value, subValue, icon, trend }: { label: string, value: string | number, subValue?: string, icon?: React.ReactNode, trend?: "up" | "down" | "neutral" }) {
    return (
        <Card className="p-0! overflow-hidden">
            <div className="p-6 flex items-start justify-between">
                <div className="space-y-1.5">
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest leading-none">{label}</p>
                    <div className="flex items-baseline gap-2">
                        <h4 className="text-2xl font-black text-foreground tracking-tighter">{value}</h4>
                        {trend && (
                            <span className={`text-[10px] font-bold ${trend === "up" ? "text-emerald-500" : trend === "down" ? "text-rose-500" : "text-muted-foreground"}`}>
                                {trend === "up" ? "↑" : trend === "down" ? "↓" : "•"}
                            </span>
                        )}
                    </div>
                    {subValue && <p className="text-[10px] text-muted-foreground font-medium">{subValue}</p>}
                </div>
                {icon && <div className="p-3 bg-secondary rounded-2xl text-secondary-foreground shadow-sm">{icon}</div>}
            </div>
        </Card>
    );
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "secondary" | "outline" | "ghost" | "amber";
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
        primary: "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20 hover:bg-indigo-700 hover:shadow-indigo-600/30 dark:bg-indigo-500 dark:hover:bg-indigo-600",
        secondary: "bg-secondary text-secondary-foreground border border-border hover:bg-muted shadow-sm",
        outline: "bg-transparent border-2 border-indigo-500 text-indigo-500 hover:bg-indigo-500/10",
        ghost: "bg-transparent text-muted-foreground hover:bg-muted hover:text-foreground",
        amber: "bg-amber-600 text-white shadow-lg shadow-amber-600/20 hover:bg-amber-700 hover:shadow-amber-600/30",
    };

    return (
        <button
            className={`px-4 py-2.5 rounded-xl font-bold transition-all duration-300 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm ${variants[variant]} ${className}`}
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
    description?: string;
}

export function Input({ label, description, className = "", value, ...props }: InputProps) {
    const safeValue = typeof value === "number" && isNaN(value) ? "" : value;

    return (
        <div className="space-y-1.5 w-full text-left">
            <div className="flex flex-col gap-0.5 ml-1">
                {label && <label className="text-sm font-bold text-foreground tracking-tight">{label}</label>}
                {description && <p className="text-[10px] text-muted-foreground font-medium leading-tight">{description}</p>}
            </div>
            <input
                className={`w-full px-4 py-3 rounded-xl border border-input bg-background focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 outline-none transition-all text-sm shadow-sm placeholder:text-muted-foreground ${className}`}
                {...props}
                value={safeValue}
            />
        </div>
    );
}
