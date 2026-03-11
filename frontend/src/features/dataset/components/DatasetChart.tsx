"use client";

import React from "react";
import { BasePlot } from "@/components/ui/BasePlot";

interface DatasetChartProps {
    title: string;
    data: number[][]; // (N, D) array
    labelX?: string;
    labelY?: string;
}

export function DatasetChart({ title, data, labelX = "Dim 1", labelY = "Dim 2" }: DatasetChartProps) {
    const traces = [{
        x: data.map((p: number[]) => p[0] || 0),
        y: data.map((p: number[]) => p[1] || 0),
        mode: 'markers' as const,
        type: 'scatter' as const,
        name: title,
        marker: {
            color: 'rgba(99, 102, 241, 0.6)',
            size: 8,
            line: { width: 1, color: 'white' }
        },
        hovertemplate: `(${labelX}: %{x:.3f}, ${labelY}: %{y:.3f})<extra></extra>`,
        hoverlabel: {
            bgcolor: 'white',
            bordercolor: 'rgba(99, 102, 241, 0.6)',
            font: { family: 'Inter, sans-serif', size: 12, color: '#1e293b' }
        }
    }];

    const layout = {
        xaxis: {
            title: { text: labelX }
        },
        yaxis: {
            title: { text: labelY }
        },
    };

    return (
        <BasePlot 
            title={title}
            data={traces}
            layout={layout}
            className="border-slate-200 shadow-sm"
            contentClassName="h-[450px]"
        />
    );
}

// Keep the old name as alias for now to prevent immediate breakages before mass-updating imports
export { DatasetChart as DatasetPlot };
