"use client";

import React, { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { BasePlot } from "@/components/ui/BasePlot";

interface SpaceData {
    X: number[][];
    y: number[][];
}

interface TransformationPreviewChartProps {
    original: SpaceData;
    transformed: SpaceData;
    dims?: [number, number];
    showComparison?: boolean;
}

export function TransformationPreviewChart({ 
    original, 
    transformed, 
    dims = [0, 1], 
    showComparison = true 
}: TransformationPreviewChartProps) {
    
    const getPlotData = (data: number[][], label: string, color: string, dims: [number, number]) => {
        const xDim = data.length > 0 && data[0].length > dims[0] ? dims[0] : 0;
        const yDim = data.length > 0 && data[0].length > dims[1] ? dims[1] : (data.length > 0 && data[0].length > 1 ? 1 : 0);

        return [{
            x: data.map(p => p[xDim] ?? 0),
            y: data.map(p => p[yDim] ?? 0),
            mode: 'markers' as const,
            type: 'scatter' as const,
            name: label,
            marker: {
                color: color,
                size: 6,
                opacity: 0.7,
                line: { width: 1, color: 'white' }
            },
            hovertemplate: `<b>${label}</b><br>Dim ${xDim}: %{x:.4f}<br>Dim ${yDim}: %{y:.4f}<extra></extra>`,
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: color,
                font: { family: 'Inter, sans-serif', size: 12, color: '#1e293b' }
            }
        }];
    };

    const getLayout = (labelX: string, labelY: string) => ({
        xaxis: { title: { text: labelX } },
        yaxis: { title: { text: labelY } },
    });

    return (
        <Card className="border-slate-200 shadow-lg bg-white overflow-hidden rounded-[2.5rem]">
            <CardHeader className="p-6 bg-slate-50/50 border-b border-slate-100 flex flex-row items-center justify-between">
                <div>
                    <CardTitle className="text-xl font-bold text-slate-900 tracking-tight">Distribution Analysis</CardTitle>
                    <CardDescription className="text-xs font-black text-slate-400 mt-1 uppercase tracking-widest italic opacity-70">Multi-Objective Topology mapping</CardDescription>
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
                                        <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Original Architecture</span>
                                        <Badge variant="outline" className="text-[9px] bg-slate-50 border-slate-200 font-bold opacity-70">X[{dims[0]}] vs X[{dims[1]}]</Badge>
                                    </div>
                                    <div className="bg-white border border-slate-100 rounded-3xl p-1 shadow-sm overflow-hidden">
                                        <BasePlot 
                                            data={getPlotData(original.X, "Reference X", "rgba(148, 163, 184, 0.4)", dims)}
                                            layout={getLayout(`X[${dims[0]}]`, `X[${dims[1]}]`)}
                                        />
                                    </div>
                                </div>
                            )}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between px-2">
                                    <span className="text-[10px] font-black uppercase tracking-widest text-indigo-400">
                                        {showComparison ? "Transformed State" : "Result State"}
                                    </span>
                                    <Badge variant="outline" className="text-[9px] bg-indigo-50 border-indigo-100 text-indigo-600 font-bold">X[{dims[0]}] vs X[{dims[1]}]</Badge>
                                </div>
                                <div className={cn("bg-indigo-50/5 border border-indigo-100/30 rounded-3xl p-1 shadow-sm overflow-hidden")}>
                                    <BasePlot 
                                        data={getPlotData(transformed.X, "Transformed X", "rgba(99, 102, 241, 0.8)", dims)}
                                        layout={getLayout(`X[${dims[0]}]`, `X[${dims[1]}]`)}
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
                                        <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Original Objective Map</span>
                                        <Badge variant="outline" className="text-[9px] bg-slate-50 border-slate-200 font-bold opacity-70">y[0] vs y[1]</Badge>
                                    </div>
                                    <div className="bg-white border border-slate-100 rounded-3xl p-1 shadow-sm overflow-hidden">
                                        <BasePlot 
                                            data={getPlotData(original.y, "Reference y", "rgba(148, 163, 184, 0.4)", [0, 1])}
                                            layout={getLayout(`y[0]`, original.y[0]?.length > 1 ? `y[1]` : "Value")}
                                        />
                                    </div>
                                </div>
                            )}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between px-2">
                                    <span className="text-[10px] font-black uppercase tracking-widest text-indigo-400">
                                        {showComparison ? "Transformed Results" : "Result Mapping"}
                                    </span>
                                    <Badge variant="outline" className="text-[9px] bg-indigo-50 border-indigo-100 text-indigo-600 font-bold">y[0] vs y[1]</Badge>
                                </div>
                                <div className={cn("bg-indigo-50/5 border border-indigo-100/30 rounded-3xl p-1 shadow-sm overflow-hidden")}>
                                    <BasePlot 
                                        data={getPlotData(transformed.y, "Transformed y", "rgba(99, 102, 241, 0.8)", [0, 1])}
                                        layout={getLayout(`y[0]`, transformed.y[0]?.length > 1 ? `y[1]` : "Value")}
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

// Keep old name for compatibility
export { TransformationPreviewChart as PreviewCharts };
