"use client";

import React, { useMemo } from "react";
import { MetricPlotData } from "../types";
import { FileDown, BarChart3 } from "lucide-react";
import { BasePlot } from "@/components/ui/BasePlot";

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
    const colors = [
        "#6366f1", // Indigo
        "#3b82f6", // Blue
        "#ec4899", // Pink
        "#14b8a6", // Teal
        "#f59e0b", // Amber
        "#8b5cf6", // Violet
    ];

    const traces = useMemo(() => {
        const plotTraces = Object.keys(data).map((label, index) => ({
            x: data[label].x,
            y: data[label].y,
            name: label,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: {
                color: colors[index % colors.length],
                width: 3,
                shape: 'spline' as const,
                smoothing: 1.3
            },
            fill: 'tozeroy' as const,
            fillcolor: colors[index % colors.length] + '14', // ~8% opacity
            hovertemplate: `<b>${label}</b><br>${yAxisLabel}: %{y:.4f}<br>${xAxisLabel}: %{x:.4f}<extra></extra>`,
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: colors[index % colors.length],
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
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
    };

    return (
        <BasePlot 
            title={title}
            description={description}
            data={traces}
            layout={layout}
            headerIcon={<FileDown className="h-4 w-4 text-indigo-500" />}
            className="group"
            contentClassName="h-[450px]"
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
    const labels = Object.keys(data);
    const values = Object.values(data);

    const colors = [
        "rgba(99, 102, 241, 0.85)",  // Indigo
        "rgba(20, 184, 166, 0.85)",  // Teal
        "rgba(244, 63, 94, 0.85)",   // Rose
        "rgba(245, 158, 11, 0.85)",  // Amber
        "rgba(139, 92, 246, 0.85)",  // Violet
    ];

    const traces = [{
        x: labels,
        y: values,
        type: 'bar' as const,
        marker: {
            color: labels.map((_, i) => colors[i % colors.length]),
            line: { width: 0 }
        },
        hovertemplate: `<b>%{x}</b><br>${yAxisLabel}: %{y:.4f}<extra></extra>`,
        hoverlabel: {
            bgcolor: 'white',
            bordercolor: '#6366f1',
            font: { family: 'Inter, sans-serif', size: 12, color: '#1e293b' }
        }
    }];

    const layout = {
        xaxis: {
            gridcolor: 'transparent',
            tickfont: { size: 12, weight: 700, color: '#64748b' },
            linecolor: '#f1f5f9'
        },
        yaxis: {
            title: { text: yAxisLabel },
            gridcolor: '#f8fafc',
            zeroline: false
        },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
    };

    return (
        <BasePlot 
            title={title}
            description={description}
            data={traces}
            layout={layout}
            headerIcon={<BarChart3 className="h-4 w-4 text-teal-500" />}
            className="group h-full"
            contentClassName="h-[450px]"
            config={{
                toImageButtonOptions: {
                    filename: `${title.toLowerCase().replace(/\s+/g, '_')}_summary`,
                }
            }}
        />
    );
}
