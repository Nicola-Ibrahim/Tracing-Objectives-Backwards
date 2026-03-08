"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Layers, Database, ChevronRight, Wand2, Loader2, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getDatasets, getDatasetDetails } from "@/features/dataset/api";
import { getTransformers, getPreview, PreviewRequest, TransformationStep } from "@/features/modeling/api";
import { PreviewCharts } from "@/features/modeling/components/PreviewCharts";
import { Label } from "@/components/ui/label";

import { ChainBuilder } from "@/features/modeling/components/ChainBuilder";

export default function TransformationPreviewerPage() {
    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [xChain, setXChain] = useState<TransformationStep[]>([]);
    const [yChain, setYChain] = useState<TransformationStep[]>([]);
    const [compareMode, setCompareMode] = useState<boolean>(true);
    const [selectedDims, setSelectedDims] = useState<[number, number]>([0, 1]);

    const { data: datasets = [] } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
    });

    const datasetDetails = useQuery({
        queryKey: ["dataset", selectedDataset],
        queryFn: () => getDatasetDetails(selectedDataset!),
        enabled: !!selectedDataset,
    });

    const { data: transformersData } = useQuery({
        queryKey: ["transformers"],
        queryFn: getTransformers,
    });

    const previewMutation = useMutation({
        mutationFn: getPreview,
    });

    const handlePreview = () => {
        if (!selectedDataset) return;
        previewMutation.mutate({
            dataset_name: selectedDataset,
            split: "train",
            sampling_limit: 2000,
            x_chain: xChain,
            y_chain: yChain,
        });
    };

    const transformers = transformersData?.transformers || [];

    return (
        <div className="space-y-8 max-w-7xl mx-auto pb-16">
            <div className="flex flex-row items-center justify-between">
                <div className="flex flex-col gap-1">
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
                        <Layers className="h-7 w-7 text-indigo-600" />
                        Transformation
                    </h1>
                    <p className="text-slate-500 font-medium">Verify data topology changes across transformation chains.</p>
                </div>

                <div className="flex items-center gap-4 bg-white p-1.5 rounded-2xl border border-slate-200 shadow-sm">
                    <div className="flex items-center gap-2 px-3">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Layout</span>
                        <Button
                            variant={compareMode ? "default" : "ghost"}
                            size="sm"
                            onClick={() => setCompareMode(true)}
                            className="h-8 rounded-xl font-bold text-[10px]"
                        >
                            Compare
                        </Button>
                        <Button
                            variant={!compareMode ? "default" : "ghost"}
                            size="sm"
                            onClick={() => setCompareMode(false)}
                            className="h-8 rounded-xl font-bold text-[10px]"
                        >
                            Final
                        </Button>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Configuration Sidebar */}
                <div className="lg:col-span-1 space-y-6">
                    <Card className="border-slate-200 shadow-sm overflow-hidden bg-white">
                        <CardHeader className="bg-slate-50/50 border-b border-slate-100 p-4">
                            <CardTitle className="text-sm font-bold flex items-center gap-2 uppercase tracking-wider text-slate-500">
                                <Database className="h-4 w-4" />
                                Dataset
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 space-y-4">
                            <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                                <SelectTrigger className="w-full bg-slate-50 border-slate-200">
                                    <SelectValue placeholder="Select dataset..." />
                                </SelectTrigger>
                                <SelectContent>
                                    {datasets.map((d) => (
                                        <SelectItem key={d.name} value={d.name}>
                                            {d.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>

                            {previewMutation.data && (
                                <div className="space-y-2 pt-2 border-t border-slate-100">
                                    <Label className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">Dimension Selector</Label>
                                    <div className="grid grid-cols-2 gap-2">
                                        <Select
                                            value={selectedDims[0].toString()}
                                            onValueChange={(v) => setSelectedDims([parseInt(v), selectedDims[1]])}
                                        >
                                            <SelectTrigger className="h-8 bg-slate-50 border-slate-200 text-xs">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {Array.from({ length: previewMutation.data.transformed.X[0]?.length || 0 }).map((_, i) => (
                                                    <SelectItem key={i} value={i.toString()}>X[{i}]</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                        <Select
                                            value={selectedDims[1].toString()}
                                            onValueChange={(v) => setSelectedDims([selectedDims[0], parseInt(v)])}
                                        >
                                            <SelectTrigger className="h-8 bg-slate-50 border-slate-200 text-xs">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {Array.from({ length: previewMutation.data.transformed.X[0]?.length || 0 }).map((_, i) => (
                                                    <SelectItem key={i} value={i.toString()}>X[{i}]</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </div>
                            )}
                            {datasetDetails.data && !previewMutation.data && (
                                <div className="space-y-2 pt-2 border-t border-slate-100">
                                    <Label className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">Dimension Selector</Label>
                                    <div className="grid grid-cols-2 gap-2">
                                        <Select
                                            value={selectedDims[0].toString()}
                                            onValueChange={(v) => setSelectedDims([parseInt(v), selectedDims[1]])}
                                        >
                                            <SelectTrigger className="h-8 bg-slate-50 border-slate-200 text-xs">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {Array.from({ length: datasetDetails.data.X[0].length }).map((_, i) => (
                                                    <SelectItem key={i} value={i.toString()}>X[{i}]</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                        <Select
                                            value={selectedDims[1].toString()}
                                            onValueChange={(v) => setSelectedDims([selectedDims[0], parseInt(v)])}
                                        >
                                            <SelectTrigger className="h-8 bg-slate-50 border-slate-200 text-xs">
                                                <SelectValue />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {Array.from({ length: datasetDetails.data.X[0].length }).map((_, i) => (
                                                    <SelectItem key={i} value={i.toString()}>X[{i}]</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </div>
                            )}
                        </CardContent>
                    </Card>

                    <Card className="border-slate-200 shadow-sm overflow-hidden bg-white">
                        <CardHeader className="bg-slate-50/50 border-b border-slate-100 p-4">
                            <CardTitle className="text-sm font-bold flex items-center gap-2 uppercase tracking-wider text-slate-500">
                                <Wand2 className="h-4 w-4" />
                                Chain Builders
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 space-y-6">
                            <ChainBuilder
                                title="Decision Space (X)"
                                chain={xChain}
                                transformers={transformers}
                                onChange={setXChain}
                            />

                            <div className="border-t border-slate-100 pt-6">
                                <ChainBuilder
                                    title="Objective Space (y)"
                                    chain={yChain}
                                    transformers={transformers}
                                    onChange={setYChain}
                                />
                            </div>

                            <Button
                                className="w-full bg-indigo-600 hover:bg-indigo-700 font-bold"
                                onClick={handlePreview}
                                disabled={!selectedDataset || previewMutation.isPending}
                            >
                                {previewMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                ) : (
                                    <ChevronRight className="h-4 w-4 mr-2" />
                                )}
                                Apply Chains
                            </Button>
                        </CardContent>
                    </Card>

                    <div className="bg-blue-50/50 border border-blue-100 p-4 rounded-3xl flex gap-3">
                        <Info className="h-5 w-5 text-blue-500 shrink-0 mt-0.5" />
                        <p className="text-xs text-blue-700 leading-relaxed font-medium">
                            Sampling is capped at <span className="font-bold">2,000 points</span> for real-time visualization performance.
                        </p>
                    </div>
                </div>

                {/* Visualization Area */}
                <div className="lg:col-span-3">
                    {previewMutation.data ? (
                        <PreviewCharts
                            original={previewMutation.data.original}
                            transformed={previewMutation.data.transformed}
                            dims={selectedDims}
                            showComparison={compareMode}
                        />
                    ) : (
                        <div className="h-[600px] border-2 border-dashed border-slate-200 rounded-[3rem] bg-slate-50/50 flex flex-col items-center justify-center p-12 text-center group">
                            <div className="w-20 h-20 bg-white rounded-[2rem] shadow-sm flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500">
                                <Layers className="h-10 w-10 text-slate-300" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-800 mb-2">Ready to Preview</h3>
                            <p className="text-slate-500 max-w-sm font-medium leading-relaxed">
                                Select a dataset and build your transformation chain to visualize spatial distribution changes.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
