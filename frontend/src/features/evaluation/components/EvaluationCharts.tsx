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
    xAxisLabel?: string;
    yAxisLabel?: string;
    showIdeal?: boolean;
    xAxisType?: "linear" | "log";
    type?: "ecdf" | "calibration";
}

export function PerformanceChart({
    title,
    description,
    data,
    xAxisLabel,
    yAxisLabel,
    showIdeal,
    xAxisType = "linear",
    type
}: PerformanceChartProps) {

    const xLabel = xAxisLabel || "Predicted Probability";
    let yLabel = yAxisLabel || "Value";
    if (type === "ecdf") yLabel = "Empirical Coverage";
    if (type === "calibration") yLabel = "Calibration Error";

    const isShowIdeal = showIdeal ?? (type === "ecdf");

    const traces = useMemo(() => {
        const plotTraces: Array<Record<string, unknown>> = Object.keys(data).map((label, index) => ({
            x: data[label].x,
            y: data[label].y,
            name: label,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: {
                color: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
                width: 4,
                shape: 'spline' as const,
                smoothing: 1.3
            },
            hovertemplate: `<b>${label}</b><br>${yLabel}: %{y:.4f}<br>${xLabel}: %{x:.4f}<extra></extra>`,
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
                font: { family: 'Inter, sans-serif', size: 12, color: '#1e293b' }
            }
        }));

        if (isShowIdeal) {
            plotTraces.push({
                x: [0, 1],
                y: [0, 1],
                name: "Ideal Calibration",
                type: 'scatter' as const,
                mode: 'lines' as const,
                line: {
                    color: 'rgba(100, 116, 139, 0.4)',
                    width: 2.5,
                    dash: 'dash' as const
                },
                fill: 'none' as const,
                fillcolor: 'transparent',
                hovertemplate: "Ideal Baseline<extra></extra>",
                hoverlabel: {
                    bgcolor: 'transparent',
                    bordercolor: 'transparent',
                    font: { family: 'Inter, sans-serif', size: 1, color: 'transparent' }
                }
            });
        }
        return plotTraces;
    }, [data, isShowIdeal, xLabel, yLabel]);

    const layout = {
        xaxis: {
            title: { text: xLabel },
            type: xAxisType,
            gridcolor: '#f1f5f9', // Muted slate grid
            zeroline: false,
            tickfont: { family: 'Inter, sans-serif', color: '#64748b', size: 14 }
        },
        yaxis: {
            title: { text: yLabel },
            range: [0, 1.05],
            gridcolor: '#f1f5f9',
            tickfont: { family: 'Inter, sans-serif', color: '#64748b', size: 14 }
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
    title?: string;
    description?: string;
    data: Record<string, Record<string, number>>;
    yAxisLabel?: string;
}

export function MetricBarChart({
    title = "Performance Metrics",
    description = "Comparison of standardized performance metrics across engines.",
    data,
    yAxisLabel = "Score"
}: MetricBarChartProps) {
    const engineNames = Object.keys(data);
    
    // Collect all unique metrics
    const allMetrics = new Set<string>();
    for (const metrics of Object.values(data)) {
        for (const metric of Object.keys(metrics || {})) {
            allMetrics.add(metric);
        }
    }
    const metricsNames = Array.from(allMetrics);

    const traces = engineNames.map((engine, index) => {
        const engineMetrics = data[engine] || {};
        const values = metricsNames.map(metric => engineMetrics[metric] || 0);

        return {
            x: metricsNames,
            y: values,
            name: engine,
            type: 'bar' as const,
            marker: {
                color: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
                line: { width: 0 }
            },
            hovertemplate: `<b>${engine}</b><br>%{x}: %{y:.4f}<extra></extra>`,
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: EVALUATION_COLORS[index % EVALUATION_COLORS.length],
                font: { family: 'Inter, sans-serif', size: 12, color: '#1e293b' }
            }
        };
    });

    const layout = {
        showlegend: true,
        barmode: 'group' as const,
        xaxis: {
            gridcolor: 'transparent',
            tickfont: { size: 14, weight: 700, color: '#64748b' },
            linecolor: '#e2e8f0',
            automargin: true
        },
        yaxis: {
            title: { text: yAxisLabel },
            gridcolor: '#f1f5f9',
            zeroline: false,
            tickfont: { family: 'Inter, sans-serif', color: '#64748b', size: 14 }
        },
        margin: { b: 80, t: 40, l: 60, r: 20 },
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


