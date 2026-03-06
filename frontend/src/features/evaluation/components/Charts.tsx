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
        "rgba(255, 99, 132, 1)",   // Vibrant Pink
        "rgba(54, 162, 235, 1)",   // Modern Blue
        "rgba(255, 206, 86, 1)",   // Soft Yellow
        "rgba(75, 192, 192, 1)",   // Teal
        "rgba(153, 102, 255, 1)",  // Purple
        "rgba(255, 159, 64, 1)",   // Orange
    ];

    const datasets = Object.keys(data).map((label, index) => ({
        label,
        data: data[label].x.map((xVal, i) => ({ x: xVal, y: data[label].y[i] })),
        borderColor: colors[index % colors.length],
        backgroundColor: colors[index % colors.length].replace("1)", "0.08)"),
        borderWidth: 3,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointBackgroundColor: colors[index % colors.length],
        pointBorderColor: "#fff",
        pointBorderWidth: 2,
        tension: 0.4, // Smoother curves
        fill: true,   // Add area fill
    }));

    if (showIdeal) {
        datasets.push({
            label: "Ideal Calibration",
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: "rgba(100, 116, 139, 0.4)",
            backgroundColor: "transparent",
            borderWidth: 1.5,
            borderDash: [6, 4],
            pointRadius: 0,
        } as any);
    }

    const chartData = { datasets };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: "index" as const,
            intersect: false,
        },
        scales: {
            x: {
                type: "linear" as const,
                title: {
                    display: true,
                    text: xAxisLabel,
                    font: { size: 11, weight: 600, family: "inherit" },
                    color: "rgb(71, 85, 105)",
                    padding: { top: 10 }
                },
                grid: { color: "rgba(226, 232, 240, 0.5)", borderDash: [2, 2] },
                ticks: { font: { size: 10 }, color: "rgb(100, 116, 139)" }
            },
            y: {
                title: {
                    display: true,
                    text: yAxisLabel,
                    font: { size: 11, weight: 600, family: "inherit" },
                    color: "rgb(71, 85, 105)",
                    padding: { bottom: 10 }
                },
                grid: { color: "rgba(226, 232, 240, 0.5)", borderDash: [2, 2] },
                ticks: { font: { size: 10 }, color: "rgb(100, 116, 139)" },
                min: 0,
                max: 1.05,
            },
        },
        plugins: {
            legend: {
                position: "bottom" as const,
                labels: {
                    usePointStyle: true,
                    pointStyle: "circle",
                    padding: 20,
                    boxWidth: 8,
                    font: { size: 11, weight: 500 },
                    color: "rgb(71, 85, 105)"
                }
            },
            tooltip: {
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                titleColor: "rgb(15, 23, 42)",
                bodyColor: "rgb(71, 85, 105)",
                borderColor: "rgba(226, 232, 240, 1)",
                borderWidth: 1,
                padding: 12,
                boxPadding: 4,
                usePointStyle: true,
                titleFont: { size: 12, weight: 700 },
                bodyFont: { size: 11 },
                callbacks: {
                    label: function (context: any) {
                        let label = context.dataset.label || "";
                        if (label) label += ": ";
                        if (context.parsed.y !== null) {
                            label += context.parsed.y.toFixed(3);
                        }
                        return label;
                    }
                }
            },
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
