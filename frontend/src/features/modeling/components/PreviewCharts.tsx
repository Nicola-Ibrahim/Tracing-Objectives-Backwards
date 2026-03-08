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
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    Title
);

interface SpaceData {
    X: number[][];
    y: number[][];
}

interface PreviewChartsProps {
    original: SpaceData;
    transformed: SpaceData;
    dims?: [number, number];
    showComparison?: boolean;
}

export function PreviewCharts({ original, transformed, dims = [0, 1], showComparison = true }: PreviewChartsProps) {
    const createScatterData = (data: number[][], label: string, color: string, spaceType: 'X' | 'y') => {
        // Fallback for empty or smaller dimensionality
        const actualXDim = data.length > 0 && data[0].length > dims[0] ? dims[0] : 0;
        const actualYDim = data.length > 0 && data[0].length > dims[1] ? dims[1] : (data.length > 0 && data[0].length > 1 ? 1 : 0);

        return {
            datasets: [
                {
                    label: label,
                    data: data.map((point) => ({
                        x: point[actualXDim] ?? 0,
                        y: point[actualYDim] ?? 0,
                    })),
                    backgroundColor: color,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                },
            ],
        };
    };

    const options = (title: string, labelX: string, labelY: string) => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 } as any,
        scales: {
            x: {
                title: { display: true, text: labelX, font: { size: 10, weight: 'bold' } as any },
                grid: { color: "rgba(0,0,0,0.03)" },
            },
            y: {
                title: { display: true, text: labelY, font: { size: 10, weight: 'bold' } as any },
                grid: { color: "rgba(0,0,0,0.03)" },
            },
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: (context: any) => `(${context.raw.x.toFixed(4)}, ${context.raw.y.toFixed(4)})`,
                },
            },
        },
    });

    return (
        <Card className="border-slate-200 shadow-lg bg-white overflow-hidden rounded-[2.5rem]">
            <CardHeader className="p-6 bg-slate-50/50 border-b border-slate-100 flex flex-row items-center justify-between">
                <div>
                    <CardTitle className="text-xl font-bold text-slate-900">Distribution Analysis</CardTitle>
                    <CardDescription className="text-xs font-medium text-slate-400 mt-1 uppercase tracking-widest">Topology Compare</CardDescription>
                </div>
            </CardHeader>
            <CardContent className="p-6">
                <Tabs defaultValue="x-space" className="space-y-6">
                    <TabsList className="bg-slate-100/50 p-1 h-12 rounded-2xl">
                        <TabsTrigger value="x-space" className="rounded-xl px-8 font-bold data-[state=active]:bg-white data-[state=active]:text-indigo-600 data-[state=active]:shadow-sm">Decision Space (X)</TabsTrigger>
                        <TabsTrigger value="y-space" className="rounded-xl px-8 font-bold data-[state=active]:bg-white data-[state=active]:text-indigo-600 data-[state=active]:shadow-sm">Objective Space (y)</TabsTrigger>
                    </TabsList>

                    <TabsContent value="x-space" className="space-y-6 focus-visible:outline-none">
                        <div className={cn("grid gap-6", showComparison ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1")}>
                            {showComparison && (
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between px-2">
                                        <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Original State</span>
                                        <Badge variant="outline" className="text-[9px] bg-slate-50 border-slate-200">X[{dims[0]}] vs X[{dims[1]}]</Badge>
                                    </div>
                                    <div className="h-[350px] bg-white border border-slate-100 rounded-3xl p-4 shadow-sm">
                                        <Scatter
                                            data={createScatterData(original.X, "Original X", "rgba(100, 116, 139, 0.4)", 'X')}
                                            options={options("Original Decision Space", `X[${dims[0]}]`, `X[${dims[1]}]`)}
                                        />
                                    </div>
                                </div>
                            )}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between px-2">
                                    <span className="text-[10px] font-black uppercase tracking-widest text-indigo-400">
                                        {showComparison ? "Transformed State" : "Result State"}
                                    </span>
                                    <Badge variant="outline" className="text-[9px] bg-indigo-50 border-indigo-100 text-indigo-600">X[{dims[0]}] vs X[{dims[1]}]</Badge>
                                </div>
                                <div className={cn("bg-indigo-50/10 border border-indigo-100/50 rounded-3xl p-4 shadow-sm", showComparison ? "h-[350px]" : "h-[500px]")}>
                                    <Scatter
                                        data={createScatterData(transformed.X, "Transformed X", "rgba(99, 102, 241, 0.7)", 'X')}
                                        options={options("Result Decision Space", `X[${dims[0]}]`, `X[${dims[1]}]`)}
                                    />
                                </div>
                            </div>
                        </div>
                    </TabsContent>

                    <TabsContent value="y-space" className="space-y-6 focus-visible:outline-none">
                        <div className={cn("grid gap-6", showComparison ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1")}>
                            {showComparison && (
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between px-2">
                                        <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Original State</span>
                                        <Badge variant="outline" className="text-[9px] bg-slate-50 border-slate-200">y[0] vs y[1]</Badge>
                                    </div>
                                    <div className="h-[350px] bg-white border border-slate-100 rounded-3xl p-4 shadow-sm">
                                        <Scatter
                                            data={createScatterData(original.y, "Original y", "rgba(100, 116, 139, 0.4)", 'y')}
                                            options={options("Original Objective Space", `y[0]`, original.y[0]?.length > 1 ? `y[1]` : "Value")}
                                        />
                                    </div>
                                </div>
                            )}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between px-2">
                                    <span className="text-[10px] font-black uppercase tracking-widest text-indigo-400">
                                        {showComparison ? "Transformed State" : "Result State"}
                                    </span>
                                    <Badge variant="outline" className="text-[9px] bg-indigo-50 border-indigo-100 text-indigo-600">y[0] vs y[1]</Badge>
                                </div>
                                <div className={cn("bg-indigo-50/10 border border-indigo-100/50 rounded-3xl p-4 shadow-sm", showComparison ? "h-[350px]" : "h-[500px]")}>
                                    <Scatter
                                        data={createScatterData(transformed.y, "Transformed y", "rgba(99, 102, 241, 0.7)", 'y')}
                                        options={options("Result Objective Space", `y[0]`, transformed.y[0]?.length > 1 ? `y[1]` : "Value")}
                                    />
                                </div>
                            </div>
                        </div>
                    </TabsContent>
                </Tabs>
            </CardContent>
        </Card>
    );
}
