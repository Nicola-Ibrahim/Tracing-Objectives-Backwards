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
    // 1. Calculate dynamic legend requirements
    const legendItems = data.filter(t => t.showlegend !== false && t.name).length;
    const legendRows = Math.ceil(legendItems / 3);
    const dynamicMarginB = 1 + (legendRows * 25);
    const minPlotHeight = 350 + dynamicMarginB;

    const defaultLayout = {
        autosize: true,
        margin: { l: 60, r: 20, t: 30, b: dynamicMarginB },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'top',
            y: -0.22,
            xanchor: 'center',
            x: 0.5,
            font: { size: 12, color: '#64748b' },
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: '#f1f5f9',
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
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        hovermode: 'closest',
        font: { family: 'Inter, sans-serif' }
    };

    const defaultConfig = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: title?.toLowerCase().replace(/\s+/g, '_') || 'plot_export',
            height: 1080,
            width: 1920,
            scale: 4 // High resolution 4x scale,
        }
    };

    // Deep merge for config to preserve toImageButtonOptions
    const mergedConfig = {
        ...defaultConfig,
        ...config,
        toImageButtonOptions: {
            ...defaultConfig.toImageButtonOptions,
            ...(config?.toImageButtonOptions || {})
        }
    };

    // Merge layout - preserving custom margins but ensuring b is at least dynamicMarginB
    const mergedLayout = {
        ...defaultLayout,
        ...layout,
        margin: {
            ...defaultLayout.margin,
            ...(layout?.margin || {}),
            b: Math.max(dynamicMarginB, layout?.margin?.b || 0)
        }
    };

    const content = (
        <Plot
            data={data}
            layout={mergedLayout}
            config={mergedConfig}
            style={style || { width: '100%', minHeight: '100%' }}
            useResizeHandler
        />
    );

    if (!title) {
        return <div className={cn("w-full h-full bg-white rounded-lg", contentClassName)} style={{ minHeight: minPlotHeight }}>{content}</div>;
    }

    return (
        <Card className={cn("border-border shadow-sm flex flex-col transition-colors duration-300", className)} style={{ minHeight: minPlotHeight }}>
            <CardHeader className="bg-muted/30 py-3 border-b border-border px-6">
                <CardTitle className="text-xs font-black text-muted-foreground uppercase tracking-widest flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        {headerIcon}
                        {title}
                    </div>
                    {headerExtra}
                </CardTitle>
                {description && <CardDescription className="text-[10px] mt-1 text-muted-foreground/80">{description}</CardDescription>}
            </CardHeader>
            <CardContent className={cn("p-2 grow bg-white relative rounded-b-[inherit]", contentClassName)}>
                {content}
            </CardContent>
        </Card>
    );
}

