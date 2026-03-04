"use client";

import React from "react";
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    Title,
} from "chart.js";
import { Scatter } from "react-chartjs-2";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    Title
);

interface DatasetPlotProps {
    title: string;
    data: number[][]; // (N, D) array
    labelX?: string;
    labelY?: string;
}

export function DatasetPlot({ title, data, labelX = "Dim 1", labelY = "Dim 2" }: DatasetPlotProps) {
    // We only plot the first 2 dimensions for now
    const scatterData = {
        datasets: [
            {
                label: title,
                data: data.map((point) => ({
                    x: point[0] || 0,
                    y: point[1] || 0,
                })),
                backgroundColor: "rgba(99, 102, 241, 0.6)",
                pointRadius: 3,
                pointHoverRadius: 5,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                title: { display: true, text: labelX, font: { size: 11, weight: 'bold' } as any },
                grid: { color: "rgba(0,0,0,0.05)" },
            },
            y: {
                title: { display: true, text: labelY, font: { size: 11, weight: 'bold' } as any },
                grid: { color: "rgba(0,0,0,0.05)" },
            },
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: (context: any) => `(${context.raw.x.toFixed(3)}, ${context.raw.y.toFixed(3)})`,
                },
            },
        },
    };

    return (
        <Card className="border-slate-200">
            <CardHeader className="py-3 px-4 border-b border-slate-100 bg-slate-50/30">
                <CardTitle className="text-sm font-medium text-slate-700">{title}</CardTitle>
            </CardHeader>
            <CardContent className="h-[250px] p-4">
                <Scatter data={scatterData} options={options} />
            </CardContent>
        </Card>
    );
}
