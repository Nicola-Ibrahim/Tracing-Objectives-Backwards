"use client";

import React from "react";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { MetricPlotData } from "../types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

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
        "rgba(99, 102, 241, 1)",   // Indigo
        "rgba(245, 158, 11, 1)",   // Amber
        "rgba(16, 185, 129, 1)",   // Emerald
        "rgba(239, 68, 68, 1)",    // Red
        "rgba(139, 92, 246, 1)",   // Violet
    ];

    const datasets = Object.keys(data.y).map((label, index) => ({
        label,
        data: data.x.map((xVal, i) => ({ x: xVal, y: data.y[label][i] })),
        borderColor: colors[index % colors.length],
        backgroundColor: colors[index % colors.length].replace("1)", "0.1)"),
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
    }));

    if (showIdeal) {
        // Add an "Ideal" 45-degree line or target line if needed
        // For PIT, ideal is uniform (diagonal if plotting CDF, or flat if plotting histogram - here we assume CDF-like data)
        datasets.push({
            label: "Ideal",
            data: data.x.map(xVal => ({ x: xVal, y: xVal })), // Assumes x and y are [0, 1]
            borderColor: "rgba(0, 0, 0, 0.2)",
            backgroundColor: "transparent",
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
        } as any);
    }

    const chartData = { datasets };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: "linear" as const,
                title: { display: true, text: xAxisLabel, font: { size: 10 } },
                grid: { color: "rgba(0,0,0,0.05)" },
            },
            y: {
                title: { display: true, text: yAxisLabel, font: { size: 10 } },
                grid: { color: "rgba(0,0,0,0.05)" },
                min: 0,
                max: 1.05,
            },
        },
        plugins: {
            legend: { position: "bottom" as const, labels: { boxWidth: 12, font: { size: 11 } } },
            tooltip: { mode: "index" as const, intersect: false },
        },
    };

    return (
        <Card className="border-slate-200 overflow-hidden">
            <CardHeader className="bg-slate-50/50 py-3 border-b border-slate-100">
                <CardTitle className="text-sm font-semibold">{title}</CardTitle>
                <CardDescription className="text-[10px] leading-tight">{description}</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px] pt-6">
                <Line data={chartData} options={options} />
            </CardContent>
        </Card>
    );
}
