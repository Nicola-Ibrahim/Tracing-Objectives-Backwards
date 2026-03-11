"use client";

import React, { useRef } from "react";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import { MetricPlotData } from "../types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Download, FileDown, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
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

    const chartRef = useRef<any>(null);

    const handleDownload = () => {
        const chart = chartRef.current;
        if (!chart) return;

        const base64 = chart.toBase64Image();
        const link = document.createElement("a");
        link.href = base64;
        link.download = `${title.replace(/\s+/g, "_").toLowerCase()}_plot.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <Card className="border-slate-200/60 overflow-hidden shadow-md shadow-slate-200/40 bg-white transition-all hover:shadow-lg hover:shadow-slate-200/50 group">
            <CardHeader className="bg-slate-50/40 py-4 px-6 border-b border-slate-100 flex flex-row items-center justify-between space-y-0">
                <div className="space-y-1">
                    <CardTitle className="text-sm font-bold text-slate-800 tracking-tight flex items-center gap-2">
                        <FileDown className="h-4 w-4 text-indigo-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                        {title}
                    </CardTitle>
                    <CardDescription className="text-[10px] leading-tight text-slate-500 font-medium">{description}</CardDescription>
                </div>
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition-colors"
                    onClick={handleDownload}
                    title="Download plot as PNG"
                >
                    <Download className="h-4 w-4" />
                </Button>
            </CardHeader>
            <CardContent className="h-[320px] p-6">
                <Line ref={chartRef} data={chartData} options={options} />
            </CardContent>
        </Card>
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
        "rgba(99, 102, 241, 0.8)",  // Indigo
        "rgba(20, 184, 166, 0.8)",  // Teal
        "rgba(244, 63, 94, 0.8)",   // Rose
        "rgba(245, 158, 11, 0.8)",  // Amber
        "rgba(139, 92, 246, 0.8)",  // Violet
    ];

    const chartData = {
        labels,
        datasets: [
            {
                label: yAxisLabel,
                data: values,
                backgroundColor: labels.map((_, i) => colors[i % colors.length]),
                borderRadius: 6,
                barThickness: 40,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: "white",
                titleColor: "#1e293b",
                bodyColor: "#475569",
                borderColor: "#e2e8f0",
                borderWidth: 1,
                padding: 12,
                titleFont: { size: 12, weight: 700 },
                bodyFont: { size: 12 },
                displayColors: false,
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: yAxisLabel,
                    font: { size: 11, weight: 600 },
                    color: "#64748b",
                },
                grid: { color: "rgba(226, 232, 240, 0.4)", borderDash: [2, 2] },
            },
            x: {
                grid: { display: false },
                ticks: { font: { size: 10, weight: 500 }, color: "#64748b" },
            },
        },
    };

    const chartRef = useRef<any>(null);

    const handleDownload = () => {
        const chart = chartRef.current;
        if (!chart) return;

        const base64 = chart.toBase64Image();
        const link = document.createElement("a");
        link.href = base64;
        link.download = `${title.replace(/\s+/g, "_").toLowerCase()}_plot.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <Card className="border-slate-200/60 overflow-hidden shadow-md shadow-slate-200/40 bg-white transition-all hover:shadow-lg hover:shadow-slate-200/50 group h-full">
            <CardHeader className="bg-slate-50/40 py-4 px-6 border-b border-slate-100 flex flex-row items-center justify-between space-y-0">
                <div className="space-y-1">
                    <CardTitle className="text-sm font-bold text-slate-800 tracking-tight flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-teal-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                        {title}
                    </CardTitle>
                    <CardDescription className="text-[10px] leading-tight text-slate-500 font-medium">{description}</CardDescription>
                </div>
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-slate-400 hover:text-teal-600 hover:bg-teal-50 transition-colors"
                    onClick={handleDownload}
                    title="Download plot as PNG"
                >
                    <Download className="h-4 w-4" />
                </Button>
            </CardHeader>
            <CardContent className="h-[320px] p-6">
                <Bar ref={chartRef} data={chartData} options={options as any} />
            </CardContent>
        </Card>
    );
}
