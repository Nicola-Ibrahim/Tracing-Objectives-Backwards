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
        <Card className="border-border shadow-2xl bg-card overflow-hidden rounded-[2.5rem]">
            <CardHeader className="p-8 bg-muted/10 border-b border-border flex flex-row items-center justify-between">
                <div>
                    <CardTitle className="text-2xl font-black text-foreground tracking-tight uppercase">Distribution Analysis</CardTitle>
                    <CardDescription className="text-[10px] font-black text-muted-foreground/50 mt-1 uppercase tracking-[0.2em] italic opacity-80">Multi-Objective Topology mapping</CardDescription>
                </div>
            </CardHeader>
            <CardContent className="p-8">
                <Tabs defaultValue="x-space" className="space-y-8">
                    <TabsList className="bg-muted p-1 h-14 rounded-2xl border border-border/50">
                        <TabsTrigger value="x-space" className="rounded-xl px-10 font-black text-[10px] uppercase tracking-widest data-[state=active]:bg-background data-[state=active]:text-indigo-500 data-[state=active]:shadow-2xl transition-all">Decision Space (X)</TabsTrigger>
                        <TabsTrigger value="y-space" className="rounded-xl px-10 font-black text-[10px] uppercase tracking-widest data-[state=active]:bg-background data-[state=active]:text-indigo-500 data-[state=active]:shadow-2xl transition-all">Objective Space (y)</TabsTrigger>
                    </TabsList>

                    <TabsContent value="x-space" className="space-y-8 focus-visible:outline-none">
                        <div className={cn("grid gap-8", showComparison ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1")}>
                            {showComparison && (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between px-3">
                                        <span className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/40">Original Architecture</span>
                                        <Badge variant="outline" className="text-[9px] bg-muted border-border font-black opacity-60 px-3 rounded-full">X[{dims[0]}] vs X[{dims[1]}]</Badge>
                                    </div>
                                    <div className="bg-background border border-border rounded-[2.5rem] p-4 shadow-inner overflow-hidden">
                                        <BasePlot
                                            data={getPlotData(original.X, "Reference X", "rgba(148, 163, 184, 0.4)", dims)}
                                            layout={getLayout(`X[${dims[0]}]`, `X[${dims[1]}]`)}
                                        />
                                    </div>
                                </div>
                            )}
                            <div className="space-y-4">
                                <div className="flex items-center justify-between px-3">
                                    <span className="text-[10px] font-black uppercase tracking-[0.2em] text-indigo-500/80">
                                        {showComparison ? "Transformed State" : "Result State"}
                                    </span>
                                    <Badge variant="outline" className="text-[9px] bg-indigo-500/5 border-indigo-500/20 text-indigo-500 font-black px-3 rounded-full shadow-lg shadow-indigo-500/5">X[{dims[0]}] vs X[{dims[1]}]</Badge>
                                </div>
                                <div className={cn("bg-indigo-500/5 border border-indigo-500/20 rounded-[2.5rem] p-4 shadow-xl shadow-indigo-500/5 overflow-hidden transition-all hover:scale-[1.01]")}>
                                    <BasePlot
                                        data={getPlotData(transformed.X, "Transformed X", "rgba(99, 102, 241, 0.8)", dims)}
                                        layout={getLayout(`X[${dims[0]}]`, `X[${dims[1]}]`)}
                                    />
                                </div>
                            </div>
                        </div>
                    </TabsContent>

                    <TabsContent value="y-space" className="space-y-8 focus-visible:outline-none">
                        <div className={cn("grid gap-8", showComparison ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1")}>
                            {showComparison && (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between px-3">
                                        <span className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/40">Original Objective Map</span>
                                        <Badge variant="outline" className="text-[9px] bg-muted border-border font-black opacity-60 px-3 rounded-full">y[0] vs y[1]</Badge>
                                    </div>
                                    <div className="bg-background border border-border rounded-[2.5rem] p-4 shadow-inner overflow-hidden">
                                        <BasePlot
                                            data={getPlotData(original.y, "Reference y", "rgba(148, 163, 184, 0.4)", [0, 1])}
                                            layout={getLayout(`y[0]`, original.y[0]?.length > 1 ? `y[1]` : "Value")}
                                        />
                                    </div>
                                </div>
                            )}
                            <div className="space-y-4">
                                <div className="flex items-center justify-between px-3">
                                    <span className="text-[10px] font-black uppercase tracking-[0.2em] text-indigo-500/80">
                                        {showComparison ? "Transformed Results" : "Result Mapping"}
                                    </span>
                                    <Badge variant="outline" className="text-[9px] bg-indigo-500/5 border-indigo-500/20 text-indigo-500 font-black px-3 rounded-full shadow-lg shadow-indigo-500/5">y[0] vs y[1]</Badge>
                                </div>
                                <div className={cn("bg-indigo-500/5 border border-indigo-500/20 rounded-[2.5rem] p-4 shadow-xl shadow-indigo-500/5 overflow-hidden transition-all hover:scale-[1.01]")}>
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
