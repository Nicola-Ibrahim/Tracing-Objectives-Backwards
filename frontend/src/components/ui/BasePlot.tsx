"use client";

import React from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { cn } from "@/lib/utils";

// Dynamically import Plot to avoid SSR issues with Plotly.js
const Plot = dynamic(() => import("react-plotly.js"), { 
    ssr: false,
    loading: () => (
        <div className="h-full w-full flex flex-col items-center justify-center bg-slate-50/50 rounded-2xl animate-pulse py-20">
            <div className="h-10 w-10 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin mb-4" />
            <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Initialising Component Canvas</span>
        </div>
    )
});

interface BasePlotProps {
    data: any[];
    layout: any;
    config?: any;
    title?: string;
    description?: string;
    className?: string;
    contentClassName?: string;
    headerIcon?: React.ReactNode;
    headerExtra?: React.ReactNode;
    style?: React.CSSProperties;
}

export function BasePlot({
    data,
    layout,
    config,
    title,
    description,
    className,
    contentClassName,
    headerIcon,
    headerExtra,
    style
}: BasePlotProps) {
    
    const defaultLayout = {
        autosize: true,
        margin: { l: 60, r: 20, t: 30, b: 130 },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'top',
            y: -0.4,
            xanchor: 'center',
            x: 0.5,
            font: { size: 12, color: '#64748b' },
            bgcolor: 'rgba(255, 255, 255, 0.5)',
            bordercolor: 'rgba(241, 245, 249, 1)',
            borderwidth: 1
        },
        xaxis: {
            gridcolor: '#f8fafc',
            zeroline: false,
            tickfont: { size: 12, color: '#94a3b8' },
            title: { font: { size: 14, color: '#94a3b8', weight: 800 } }
        },
        yaxis: {
            gridcolor: '#f8fafc',
            zeroline: false,
            tickfont: { size: 12, color: '#94a3b8' },
            title: { font: { size: 14, color: '#94a3b8', weight: 800 } }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        hovermode: 'closest',
        font: { family: 'Inter, sans-serif' }
    };

    const defaultConfig = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
        toImageButtonOptions: {
            format: 'png',
            scale: 3 
        }
    };

    // Deep merge or simple override for layout/config? 
    // For now, let's just spread props but keep the core standards accessible.
    const mergedLayout = { ...defaultLayout, ...layout };
    const mergedConfig = { ...defaultConfig, ...config };

    const content = (
        <Plot 
            data={data} 
            layout={mergedLayout} 
            config={mergedConfig} 
            style={style || { width: '100%', height: '100%' }}
            useResizeHandler
        />
    );

    if (!title) {
        return <div className={cn("w-full h-full", contentClassName)}>{content}</div>;
    }

    return (
        <Card className={cn("border-slate-200 shadow-sm overflow-hidden flex flex-col", className)}>
            <CardHeader className="bg-slate-50/40 py-3 border-b border-slate-100 px-6">
                <CardTitle className="text-xs font-black text-slate-500 uppercase tracking-widest flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        {headerIcon}
                        {title}
                    </div>
                    {headerExtra}
                </CardTitle>
                {description && <CardDescription className="text-[10px] mt-1">{description}</CardDescription>}
            </CardHeader>
            <CardContent className={cn("p-2 grow", contentClassName)}>
                {content}
            </CardContent>
        </Card>
    );
}
