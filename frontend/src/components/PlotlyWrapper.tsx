"use client";

import dynamic from "next/dynamic";
import React from "react";

// Dynamically import Plot with no SSR to avoid "window is not defined"
const Plot = dynamic(() => import("react-plotly.js"), {
    ssr: false,
    loading: () => (
        <div className="w-full h-[400px] flex items-center justify-center bg-slate-50 rounded-xl border border-dashed border-slate-200">
            <div className="flex flex-col items-center gap-2">
                <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                <p className="text-sm text-slate-400">Initializing visualizing engine...</p>
            </div>
        </div>
    ),
});

interface PlotlyWrapperProps {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    className?: string;
    style?: React.CSSProperties;
    useResizeHandler?: boolean;
}

export default function PlotlyWrapper({ data, layout, config, className, style, useResizeHandler }: PlotlyWrapperProps) {
    return (
        <div className={className}>
            <Plot
                data={data}
                layout={{
                    autosize: true,
                    margin: { t: 40, r: 40, l: 60, b: 60 },
                    font: {
                        family: "var(--font-geist-sans), sans-serif",
                        color: "#475569"
                    },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    xaxis: {
                        gridcolor: "#f1f5f9",
                        linecolor: "#f1f5f9",
                        zerolinecolor: "#f1f5f9",
                        tickfont: { size: 11 }
                    },
                    yaxis: {
                        gridcolor: "#f1f5f9",
                        linecolor: "#f1f5f9",
                        zerolinecolor: "#f1f5f9",
                        tickfont: { size: 11 }
                    },
                    transition: {
                        duration: 500,
                        easing: "cubic-in-out"
                    },
                    ...layout,
                }}
                config={{
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ["lasso2d", "select2d"],
                    ...config,
                }}
                style={style || { width: "100%", height: "100%", minHeight: "400px" }}
                useResizeHandler={useResizeHandler}
            />
        </div>
    );
}
