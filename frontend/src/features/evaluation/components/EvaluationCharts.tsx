"use client";

import React, { useMemo } from "react";
import { MetricPlotData } from "../types";
import { FileDown, BarChart3 } from "lucide-react";
import { BasePlot } from "@/components/ui/BasePlot";

const EVALUATION_COLORS = [
    "#6366f1", // Indigo
    "#10b981", // Emerald
    "#f43f5e", // Rose
    "#f59e0b", // Amber
    "#3b82f6", // Blue
    "#8b5cf6", // Violet
    "#f97316", // Orange
    "#06b6d4", // Cyan
    "#84cc16", // Lime
    "#ec4899", // Pink
    "#14b8a6", // Teal
    "#ef4444", // Red
];

interface PerformanceChartProps {
    title: string;
    description: string;
    data: MetricPlotData;
    xAxisLabel: string;
    yAxisLabel: string;
    showIdeal?: boolean;
}

export function PerformanceChart({
    title,
    description,
    data,
    xAxisLabel,
    yAxisLabel,
    showIdeal = false
}: PerformanceChartProps) {


    const traces = useMemo(() => {
        const plotTraces = Object.keys(data).map((label, index) => ({
            x: data[label].x,
            y: data[label].y,
            name: label,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: {
                color: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
                width: 3,
                shape: 'spline' as const,
                smoothing: 1.3
            },
            hovertemplate: `<b>${label}</b><br>${yAxisLabel}: %{y:.4f}<br>${xAxisLabel}: %{x:.4f}<extra></extra>`,
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
                font: { family: 'Inter, sans-serif', size: 12, color: '#1e293b' }
            }
        }));

        if (showIdeal) {
            plotTraces.push({
                x: [0, 1],
                y: [0, 1],
                name: "Ideal Calibration",
                type: 'scatter' as const,
                mode: 'lines' as const,
                line: {
                    color: 'rgba(100, 116, 139, 0.4)',
                    width: 1.5,
                    dash: 'dash' as const
                },
                fill: 'none' as const,
                fillcolor: 'transparent',
                hovertemplate: "Ideal Baseline<extra></extra>"
            } as any);
        }
        return plotTraces;
    }, [data, showIdeal, xAxisLabel, yAxisLabel]);

    const layout = {
        xaxis: {
            title: { text: xAxisLabel },
            gridcolor: '#f8fafc',
            zeroline: false
        },
        yaxis: {
            title: { text: yAxisLabel },
            range: [0, 1.05]
        },
    };

    return (
        <BasePlot
            title={title}
            description={description}
            data={traces}
            layout={layout}
            headerIcon={<FileDown className="h-4 w-4 text-indigo-500" />}
            className="group"
            config={{
                toImageButtonOptions: {
                    filename: `${title.toLowerCase().replace(/\s+/g, '_')}_benchmark`,
                }
            }}
        />
    );
}

interface MetricBarChartProps {
    title: string;
    description: string;
    data: Record<string, number>;
    yAxisLabel: string;
}

export function MetricBarChart({
    title,
    description,
    data,
    yAxisLabel
}: MetricBarChartProps) {
    const traces = Object.entries(data).map(([label, value], index) => ({
        x: [label],
        y: [value],
        name: label,
        type: 'bar' as const,
        marker: {
            color: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
            line: { width: 0 }
        },
        hovertemplate: `<b>${label}</b><br>${yAxisLabel}: %{y:.4f}<extra></extra>`,
        hoverlabel: {
            bgcolor: 'white',
            bordercolor: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
            font: { family: 'Inter, sans-serif', size: 12, color: '#1e293b' }
        }
    }));

    const layout = {
        showlegend: true,
        xaxis: {
            gridcolor: 'transparent',
            tickfont: { size: 12, weight: 700, color: '#64748b' },
            linecolor: '#f1f5f9',
            automargin: true
        },
        yaxis: {
            title: { text: yAxisLabel },
            gridcolor: '#f8fafc',
            zeroline: false
        },
        margin: { b: 150, t: 40, l: 60, r: 20 },
    };

    return (
        <BasePlot
            title={title}
            description={description}
            data={traces}
            layout={layout}
            headerIcon={<BarChart3 className="h-4 w-4 text-teal-500" />}
            className="group h-full"
            config={{
                toImageButtonOptions: {
                    filename: `${title.toLowerCase().replace(/\s+/g, '_')}_summary`,
                }
            }}
        />
    );
}


