"use client";

import React from "react";

interface DatasetSelectorProps {
    datasets: string[];
    selectedDataset: string;
    onSelect: (name: string) => void;
    isLoading?: boolean;
}

export default function DatasetSelector({
    datasets,
    selectedDataset,
    onSelect,
    isLoading
}: DatasetSelectorProps) {
    return (
        <div className="space-y-1.5">
            <label className="text-sm font-semibold text-slate-700 ml-1">
                Dataset Source
            </label>
            <div className="relative">
                <select
                    value={selectedDataset}
                    onChange={(e) => onSelect(e.target.value)}
                    disabled={isLoading}
                    className="w-full px-4 py-2.5 rounded-xl border border-slate-200 bg-white/50 focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all text-sm appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <option value="" disabled>Select a dataset...</option>
                    {datasets.map((name) => (
                        <option key={name} value={name}>
                            {name}
                        </option>
                    ))}
                </select>
                <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-slate-400">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                </div>
            </div>
            {isLoading && (
                <p className="text-[10px] text-primary animate-pulse ml-1 font-medium italic">
                    Loading dataset data...
                </p>
            )}
        </div>
    );
}
