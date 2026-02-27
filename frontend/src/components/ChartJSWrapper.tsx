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
} from "chart.js";
import { Scatter } from "react-chartjs-2";
import zoomPlugin from "chartjs-plugin-zoom";
import "hammerjs";

// Register Chart.js components
ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    ScatterController,
    Title,
    Decimation,
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
    title: string;
    datasets: ChartDataset[];
    xAxisTitle?: string;
    yAxisTitle?: string;
    xRange?: [number, number];
    yRange?: [number, number];
}

const ChartJSWrapper: React.FC<ChartJSWrapperProps> = ({
    title,
    datasets,
    xAxisTitle,
    yAxisTitle,
    xRange,
    yRange,
}) => {
    const chartRef = useRef<any>(null);

    const resetZoom = () => {
        if (chartRef.current) {
            chartRef.current.resetZoom();
        }
    };

    const options = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 400,
            easing: "easeOutQuart" as const,
        },
        interaction: {
            mode: "nearest" as const,
            axis: "xy" as const,
            intersect: false,
        },
        parsing: false as const,
        normalized: true as const,
        plugins: {
            legend: {
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
                    size: 14,
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
                callbacks: {
                    label: (context: any) => {
                        return `${context.dataset.label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                    }
                }
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: "xy" as const,
                },
                zoom: {
                    wheel: {
                        enabled: true,
                    },
                    pinch: {
                        enabled: true,
                    },
                    mode: "xy" as const,
                },
            },
            decimation: {
                enabled: true,
                algorithm: "min-max" as const,
            },
        },
        scales: {
            x: {
                title: {
                    display: !!xAxisTitle,
                    text: xAxisTitle,
                    color: THEME.textSecondary,
                    font: {
                        weight: 600 as any,
                        size: 11,
                    },
                },
                grid: {
                    color: THEME.grid,
                    drawBorder: false,
                },
                ticks: {
                    color: THEME.textMuted,
                    font: { size: 10 },
                },
                border: {
                    display: true,
                    color: THEME.grid,
                },
                min: xRange?.[0],
                max: xRange?.[1],
            },
            y: {
                title: {
                    display: !!yAxisTitle,
                    text: yAxisTitle,
                    color: THEME.textSecondary,
                    font: {
                        weight: 600 as any,
                        size: 11,
                    },
                },
                grid: {
                    color: THEME.grid,
                    drawBorder: false,
                },
                ticks: {
                    color: THEME.textMuted,
                    font: { size: 10 },
                },
                border: {
                    display: true,
                    color: THEME.grid,
                },
                min: yRange?.[0],
                max: yRange?.[1],
            },
        },
    }), [title, xAxisTitle, yAxisTitle, xRange, yRange]);

    const data = useMemo(() => ({
        datasets: datasets.map((ds) => ({
            ...ds,
            borderWidth: ds.showLine ? (ds.borderWidth ?? 2) : 1.5,
            borderColor: ds.borderColor ?? ds.backgroundColor,
            pointBorderColor: "#ffffff", // Plotly-style white border around points
            pointBorderWidth: 1.5,
            pointStyle: ds.pointStyle ?? "circle",
            pointHoverBorderWidth: 2,
            borderDash: ds.borderDash ?? [],
        })),
    }), [datasets]);

    return (
        <div className="w-full flex flex-col h-[600px]">
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
                    <Scatter ref={chartRef} options={options} data={data} />
                </div>
            </div>
        </div>
    );
};

export default ChartJSWrapper;
