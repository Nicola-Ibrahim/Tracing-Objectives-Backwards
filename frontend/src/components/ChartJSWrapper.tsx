"use client";

import React, { useMemo, useRef, useEffect } from "react";
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

// Register Chart.js components
ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    ScatterController,
    Title,
    Decimation
);

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
}

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

    const options = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: false as const, // High performance
        parsing: false as const, // Expect {x, y} format
        normalized: true as const,
        plugins: {
            legend: {
                position: "top" as const,
                labels: {
                    color: "#cbd5e1", // Tailwind slate-300
                },
            },
            title: {
                display: true,
                text: title,
                color: "#f8fafc", // Tailwind slate-50
                font: {
                    size: 16,
                },
            },
            tooltip: {
                enabled: true,
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
                    color: "#94a3b8", // Tailwind slate-400
                },
                grid: {
                    color: "rgba(148, 163, 184, 0.1)",
                },
                ticks: {
                    color: "#94a3b8",
                },
                min: xRange?.[0],
                max: xRange?.[1],
            },
            y: {
                title: {
                    display: !!yAxisTitle,
                    text: yAxisTitle,
                    color: "#94a3b8",
                },
                grid: {
                    color: "rgba(148, 163, 184, 0.1)",
                },
                ticks: {
                    color: "#94a3b8",
                },
                min: yRange?.[0],
                max: yRange?.[1],
            },
        },
    }), [title, xAxisTitle, yAxisTitle, xRange, yRange]);

    const data = useMemo(() => ({
        datasets: datasets.map((ds) => ({
            ...ds,
            borderWidth: ds.borderWidth ?? 1,
        })),
    }), [datasets]);

    return (
        <div className="w-full h-full min-h-[400px] p-4 bg-slate-900 rounded-xl shadow-2xl border border-slate-800">
            <Scatter ref={chartRef} options={options} data={data} />
        </div>
    );
};

export default ChartJSWrapper;
