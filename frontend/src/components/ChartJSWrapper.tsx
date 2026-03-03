"use client";

import React, { useMemo, useRef } from "react";
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    ScatterController,
    Title,
    Decimation,
    BarElement,
    CategoryScale,
} from "chart.js";
import { Scatter, Bar } from "react-chartjs-2";
// Register basic Chart.js components
import annotationPlugin from "chartjs-plugin-annotation";
import zoomPlugin from "chartjs-plugin-zoom";

// Register basic Chart.js components
ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    ScatterController,
    Title,
    Decimation,
    BarElement,
    CategoryScale,
    annotationPlugin,
    zoomPlugin
);

const THEME = {
    background: "#ffffff",
    cardBackground: "#ffffff",
    grid: "rgb(238, 238, 238)", // Plotly-style light grid
    zeroLine: "rgb(68, 68, 68)", // Bolder zero line
    textMain: "#2d3748",
    textSecondary: "#4a5568",
    textMuted: "#a0aec0",
};

export interface ChartDataPoint {
    x: number;
    y: number;
}

export interface ChartDataset {
    label: string;
    data: ChartDataPoint[];
    backgroundColor: string;
    pointRadius?: number;
    pointHoverRadius?: number;
    showLine?: boolean;
    borderColor?: string;
    borderWidth?: number;
    pointStyle?: PointStyle;
    borderDash?: number[];
}

import { PointStyle } from "chart.js";

interface ChartJSWrapperProps {
    type?: "scatter" | "bar";
    title: string;
    datasets: ChartDataset[];
    labels?: string[];
    xAxisTitle?: string;
    yAxisTitle?: string;
    xRange?: [number, number];
    yRange?: [number, number];
    circleAnnotations?: Array<{
        center: { x: number; y: number };
        radius: number;
        color?: string;
        label?: string;
        borderDash?: number[];
    }>;
    lineAnnotation?: {
        value: number;
        label: string;
    };
    height?: string;
}

const ChartJSWrapper: React.FC<ChartJSWrapperProps> = ({
    type = "scatter",
    title,
    datasets,
    labels,
    xAxisTitle,
    yAxisTitle,
    xRange,
    yRange,
    circleAnnotations,
    lineAnnotation,
    height,
}) => {
    const chartRef = useRef<any>(null);

    const resetZoom = () => {
        if (chartRef.current) {
            chartRef.current.resetZoom();
        }
    };

    const options = useMemo(() => {
        const annotations: any = {};

        if (circleAnnotations) {
            circleAnnotations.forEach((circle, index) => {
                annotations[`circle_${index}`] = {
                    type: 'ellipse' as const,
                    xMin: circle.center.x - circle.radius,
                    xMax: circle.center.x + circle.radius,
                    yMin: circle.center.y - circle.radius,
                    yMax: circle.center.y + circle.radius,
                    backgroundColor: 'transparent',
                    borderColor: circle.color ?? 'rgba(239, 68, 68, 0.6)',
                    borderWidth: 2,
                    borderDash: circle.borderDash ?? [6, 6],
                    label: circle.label ? {
                        display: true,
                        content: circle.label,
                        position: 'start' as any,
                        backgroundColor: circle.color ?? 'rgba(239, 68, 68, 0.6)',
                        color: '#fff',
                        font: { size: 9, weight: 'bold' }
                    } : undefined
                };
            });
        }

        if (lineAnnotation) {
            annotations.thresholdLine = {
                type: 'line' as const,
                yMin: lineAnnotation.value,
                yMax: lineAnnotation.value,
                borderColor: 'rgba(239, 68, 68, 0.8)',
                borderWidth: 2,
                borderDash: [6, 6],
                label: {
                    display: true,
                    content: lineAnnotation.label,
                    position: 'end' as any,
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    color: '#fff',
                    font: { size: 10, weight: 'bold' }
                }
            };
        }

        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 400,
                easing: "easeOutQuart" as const,
            },
            interaction: {
                mode: type === "scatter" ? "nearest" : "index" as any,
                axis: "xy" as const,
                intersect: false,
            },
            parsing: false as const,
            normalized: true as const,
            plugins: {
                legend: {
                    display: type === "scatter", // Hide for bar charts (coherence gate)
                    position: "bottom" as const,
                    align: "center" as const,
                    labels: {
                        color: THEME.textSecondary,
                        usePointStyle: true,
                        pointStyle: "circle",
                        padding: 20,
                        font: {
                            family: "Inter, sans-serif",
                            size: 12,
                            weight: 500 as any,
                        },
                    },
                },
                title: {
                    display: true,
                    text: title.toUpperCase(),
                    color: THEME.textMain,
                    font: {
                        family: "Inter, sans-serif",
                        size: 14, // Slightly smaller
                        weight: "800" as any,
                    },
                    padding: { bottom: 25, top: 10 },
                    align: "start" as const,
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: "rgba(255, 255, 255, 0.95)",
                    titleColor: THEME.textMain,
                    titleFont: { weight: "bold" as any },
                    bodyColor: THEME.textSecondary,
                    borderColor: "rgba(0, 0, 0, 0.1)",
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 4,
                    displayColors: true,
                    usePointStyle: true,
                    callbacks: type === "scatter" ? {
                        label: (context: any) => {
                            return `${context.dataset.label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                        }
                    } : undefined
                },
                zoom: {
                    pan: { enabled: true, mode: "xy" as const },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: "xy" as const,
                    },
                },
                annotation: { annotations },
                decimation: { enabled: true, algorithm: "min-max" as const },
            },
            scales: {
                x: {
                    title: {
                        display: !!xAxisTitle,
                        text: xAxisTitle,
                        color: THEME.textSecondary,
                        font: { weight: 600 as any, size: 11 },
                    },
                    grid: { color: THEME.grid, drawBorder: false },
                    ticks: { color: THEME.textMuted, font: { size: 10 } },
                    min: xRange?.[0],
                    max: xRange?.[1],
                    type: type === "bar" ? "category" : "linear",
                    border: { display: true, color: THEME.grid },
                },
                y: {
                    title: {
                        display: !!yAxisTitle,
                        text: yAxisTitle,
                        color: THEME.textSecondary,
                        font: { weight: 600 as any, size: 11 },
                    },
                    grid: { color: THEME.grid, drawBorder: false },
                    ticks: { color: THEME.textMuted, font: { size: 10 } },
                    min: yRange?.[0],
                    max: yRange?.[1],
                    beginAtZero: true,
                    border: { display: true, color: THEME.grid },
                },
            },
        };
    }, [type, title, xAxisTitle, yAxisTitle, xRange, yRange, circleAnnotations, lineAnnotation]);

    const data = useMemo(() => ({
        labels,
        datasets: datasets.map((ds) => ({
            ...ds,
            borderWidth: ds.showLine ? (ds.borderWidth ?? 2) : 1.5,
            borderColor: ds.borderColor ?? ds.backgroundColor,
            pointBorderColor: ds.borderColor ?? "#ffffff",
            pointBorderWidth: 1.5,
            pointStyle: ds.pointStyle ?? "circle",
            pointHoverBorderWidth: 2,
            borderDash: ds.borderDash ?? [],
        })),
    }), [datasets, labels]);

    return (
        <div className={`w-full flex flex-col ${height ? height : (type === 'bar' ? 'h-[350px]' : 'h-[600px]')}`}>
            <div className="glass-panel p-6 flex flex-col h-full overflow-hidden">
                <div className="flex justify-end mb-2">
                    <button
                        onClick={resetZoom}
                        className="text-[10px] uppercase tracking-wider font-bold px-3 py-1 bg-slate-100/50 hover:bg-slate-200/50 text-slate-500 rounded-lg border border-slate-200/50 transition-all duration-200"
                    >
                        Reset Zoom
                    </button>
                </div>
                <div className="grow relative h-0 min-h-0">
                    {type === "scatter" ? (
                        <Scatter ref={chartRef} options={options as any} data={data as any} />
                    ) : (
                        <Bar ref={chartRef} options={options as any} data={data as any} />
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChartJSWrapper;
