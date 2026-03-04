import React from "react";

interface CardProps {
    children: React.ReactNode;
    className?: string;
    title?: string;
}

export function Card({ children, className = "", title }: CardProps) {
    return (
        <div className={`glass-panel p-6 overflow-hidden transition-all duration-300 hover:shadow-xl hover:shadow-primary/5 ${className}`}>
            {title && (
                <div className="flex items-center justify-between mb-5 px-1">
                    <h3 className="text-base font-bold text-slate-800 tracking-tight">{title}</h3>
                </div>
            )}
            {children}
        </div>
    );
}

export function Badge({ children, variant = "default", className = "" }: { children: React.ReactNode, variant?: "default" | "success" | "warning" | "error" | "indigo", className?: string }) {
    const variants = {
        default: "bg-slate-100 text-slate-600",
        success: "bg-emerald-50 text-emerald-600 border border-emerald-100",
        warning: "bg-amber-50 text-amber-600 border border-amber-100",
        error: "bg-rose-50 text-rose-600 border border-rose-100",
        indigo: "bg-indigo-50 text-indigo-600 border border-indigo-100",
    };
    return (
        <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${variants[variant]} ${className}`}>
            {children}
        </span>
    );
}

export function StatCard({ label, value, subValue, icon, trend }: { label: string, value: string | number, subValue?: string, icon?: React.ReactNode, trend?: "up" | "down" | "neutral" }) {
    return (
        <Card className="p-5!">
            <div className="flex items-start justify-between">
                <div className="space-y-1">
                    <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">{label}</p>
                    <div className="flex items-baseline gap-2">
                        <h4 className="text-2xl font-black text-slate-900 tracking-tighter">{value}</h4>
                        {trend && (
                            <span className={`text-[10px] font-bold ${trend === "up" ? "text-emerald-500" : trend === "down" ? "text-rose-500" : "text-slate-400"}`}>
                                {trend === "up" ? "↑" : trend === "down" ? "↓" : "•"}
                            </span>
                        )}
                    </div>
                    {subValue && <p className="text-[10px] text-slate-500 font-medium">{subValue}</p>}
                </div>
                {icon && <div className="p-3 bg-slate-50 rounded-2xl text-slate-600">{icon}</div>}
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
        primary: "bg-indigo-600 text-white shadow-lg shadow-indigo-100 hover:bg-indigo-700 hover:shadow-indigo-200",
        secondary: "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 shadow-sm",
        outline: "bg-transparent border-2 border-indigo-600 text-indigo-600 hover:bg-indigo-50",
        ghost: "bg-transparent text-slate-500 hover:bg-slate-100 hover:text-slate-700",
        amber: "bg-amber-600 text-white shadow-lg shadow-amber-100 hover:bg-amber-700 hover:shadow-amber-200",
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
    // Safely handle NaN values by converting them to empty string
    const safeValue = typeof value === "number" && isNaN(value) ? "" : value;

    return (
        <div className="space-y-1.5 w-full">
            <div className="flex flex-col gap-0.5 ml-1">
                {label && <label className="text-sm font-bold text-slate-700 tracking-tight">{label}</label>}
                {description && <p className="text-[10px] text-slate-400 font-medium leading-tight">{description}</p>}
            </div>
            <input
                className={`w-full px-4 py-3 rounded-xl border border-slate-200 bg-white/50 focus:bg-white focus:ring-4 focus:ring-indigo-500/5 focus:border-indigo-500 outline-none transition-all text-sm shadow-sm ${className}`}
                {...props}
                value={safeValue}
            />
        </div>
    );
}
