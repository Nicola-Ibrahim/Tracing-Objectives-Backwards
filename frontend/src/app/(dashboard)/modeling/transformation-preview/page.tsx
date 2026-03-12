"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Database, ChevronRight, Wand2, Loader2, Info, Sparkles, Binary, Share2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getDatasets, getDatasetDetails } from "@/features/dataset/api";
import { getTransformers, getPreview, PreviewRequest, TransformationStep } from "@/features/modeling/api";
import { TransformationPreviewChart } from "@/features/modeling/components/TransformationPreviewChart";
import { Label } from "@/components/ui/label";

import { ChainBuilder } from "@/features/modeling/components/ChainBuilder";

export default function TransformationPreviewerPage() {
    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [xChain, setXChain] = useState<TransformationStep[]>([]);
    const [yChain, setYChain] = useState<TransformationStep[]>([]);
    const [compareMode, setCompareMode] = useState<boolean>(true);

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
        <div className="space-y-8 max-w-full mx-auto pb-16 px-4 md:px-0">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 relative">
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-2 relative"
                >
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-600 rounded-[1rem]">
                            <Binary className="h-6 w-6 text-white" />
                        </div>
                        <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-foreground via-foreground/90 to-foreground/80 font-sans">
                            Topology Preview
                        </h1>
                    </div>
                    <p className="text-muted-foreground font-medium ml-12">Verify data topology changes across transformation chains.</p>
                    <div className="absolute -top-10 -right-20 opacity-[0.03] dark:opacity-[0.05] pointer-events-none">
                        <Layers className="h-48 w-48 text-indigo-500 rotate-12" />
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-4 bg-muted/30 p-1 rounded-[1rem] border border-border/50 backdrop-blur-sm"
                >
                    <div className="flex items-center gap-1 px-2">
                        <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground/60 mr-2">Layout</span>
                        <Button
                            variant={compareMode ? "default" : "ghost"}
                            size="sm"
                            onClick={() => setCompareMode(true)}
                            className={cn(
                                "h-8 rounded-[1rem] font-bold text-[10px] px-4 transition-all",
                                compareMode ? "bg-background text-indigo-500 border border-border/50 hover:bg-background/80" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Difference
                        </Button>
                        <Button
                            variant={!compareMode ? "default" : "ghost"}
                            size="sm"
                            onClick={() => setCompareMode(false)}
                            className={cn(
                                "h-8 rounded-[1rem] font-bold text-[10px] px-4 transition-all",
                                !compareMode ? "bg-background text-indigo-500 border border-border/50 hover:bg-background/80" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Final State
                        </Button>
                    </div>
                </motion.div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Configuration Sidebar */}
                <div className="lg:col-span-1 space-y-8">
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                    >
                        <Card className="border-border rounded-[1rem] bg-card transition-all">
                            <CardHeader className="bg-muted/30 border-b border-border py-4 px-6 flex flex-row items-center justify-between space-y-0 text-foreground">
                                <CardTitle className="text-sm font-black flex items-center gap-2 uppercase tracking-widest text-muted-foreground/60">
                                    <Database className="h-4 w-4" />
                                    Source
                                </CardTitle>
                                <div className="bg-indigo-500/10 p-1.5 rounded-[1rem]">
                                    <Database className="h-3.5 w-3.5 text-indigo-500" />
                                </div>
                            </CardHeader>
                            <CardContent className="p-6 space-y-4">
                                <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                                    <SelectTrigger className="w-full bg-background border-border font-bold text-foreground h-10">
                                        <SelectValue placeholder="Select dataset..." />
                                    </SelectTrigger>
                                    <SelectContent className="bg-popover border-border text-popover-foreground">
                                        {datasets.map((d) => (
                                            <SelectItem key={d.name} value={d.name} className="font-medium hover:bg-muted focus:bg-muted cursor-pointer transition-colors">
                                                {d.name}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>

                                <div className="pt-4 border-t border-border">
                                    <p className="text-[10px] font-black text-muted-foreground/40 uppercase tracking-widest pl-1">Space Projection</p>
                                    <p className="text-[11px] font-bold text-muted-foreground/60 ml-1 mt-2">Visualizing first two dimensions of the manifold.</p>
                                </div>
                            </CardContent>
                        </Card>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0, transition: { delay: 0.1 } }}
                    >
                        <Card className="border-border rounded-[1rem] bg-card transition-all">
                            <CardHeader className="bg-muted/30 border-b border-border py-4 px-6 flex flex-row items-center justify-between space-y-0 text-foreground">
                                <CardTitle className="text-sm font-black flex items-center gap-2 uppercase tracking-widest text-muted-foreground/60">
                                    <Wand2 className="h-4 w-4" />
                                    Estimators
                                </CardTitle>
                                <div className="bg-indigo-500/10 p-1.5 rounded-[1rem]">
                                    <Sparkles className="h-3.5 w-3.5 text-indigo-500" />
                                </div>
                            </CardHeader>
                            <CardContent className="p-6 space-y-6">
                                <ChainBuilder
                                    title="Manifold Transformation (X)"
                                    chain={xChain}
                                    transformers={transformers}
                                    onChange={setXChain}
                                />

                                <div className="border-t border-border pt-6">
                                    <ChainBuilder
                                        title="Objective Mapping (y)"
                                        chain={yChain}
                                        transformers={transformers}
                                        onChange={setYChain}
                                    />
                                </div>

                                <Button
                                    className="w-full bg-foreground text-background hover:opacity-90 font-black h-11 transition-all hover:scale-[1.02] active:scale-95 text-xs tracking-widest uppercase group"
                                    onClick={handlePreview}
                                    disabled={!selectedDataset || previewMutation.isPending}
                                >
                                    {previewMutation.isPending ? (
                                        <Loader2 className="h-4 w-4 animate-spin mr-2 text-indigo-500" />
                                    ) : (
                                        <ChevronRight className="h-4 w-4 mr-2 text-indigo-500 group-hover:translate-x-1 transition-transform" />
                                    )}
                                    Update Projections
                                </Button>
                            </CardContent>
                        </Card>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1, transition: { delay: 0.2 } }}
                        className="bg-indigo-500/10 border border-indigo-500/20 p-5 rounded-[1rem] flex gap-4"
                    >
                        <div className="bg-background p-2 rounded-[1rem] h-fit border border-indigo-500/20">
                            <Info className="h-4 w-4 text-indigo-500 shrink-0" />
                        </div>
                        <p className="text-[11px] text-muted-foreground leading-relaxed font-bold">
                            High-density sampling is capped at <span className="text-foreground font-extrabold underline decoration-indigo-500/30">2,000 observations</span> to maintain interactive frame rates.
                        </p>
                    </motion.div>
                </div>

                <div className="lg:col-span-3 min-h-[600px]">
                    <AnimatePresence mode="wait">
                        {previewMutation.data ? (
                            <motion.div
                                key="results"
                                initial={{ opacity: 0, scale: 0.98 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.98 }}
                                className="h-full"
                            >
                                <TransformationPreviewChart
                                    original={previewMutation.data.original}
                                    transformed={previewMutation.data.transformed}
                                    dims={[0, 1]}
                                    showComparison={compareMode}
                                />
                            </motion.div>
                        ) : (
                            <motion.div
                                key="empty"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="h-full flex flex-col items-center justify-center p-12 text-center group bg-muted/10 border-2 border-dashed border-border/60 rounded-[1rem]"
                            >
                                <div className="relative mb-10">
                                    <div className="absolute inset-0 bg-indigo-500/20 blur-3xl rounded-full scale-150 animate-pulse" />
                                    <div className="relative w-24 h-24 bg-card rounded-[1rem] flex items-center justify-center group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 border border-border">
                                        <Layers className="h-12 w-12 text-indigo-400 opacity-60" />
                                    </div>
                                    <div className="absolute -bottom-2 -right-2 w-10 h-10 bg-indigo-600 rounded-[1rem] flex items-center justify-center">
                                        <Share2 className="h-5 w-5 text-white" />
                                    </div>
                                </div>
                                <h3 className="text-2xl font-black text-foreground tracking-tight mb-3">Initialize Mapping Preview</h3>
                                <p className="text-muted-foreground max-w-sm font-medium leading-relaxed mb-8">
                                    Select a reference dataset and construct a transformation manifold to visualize spatial coherence and topology drift.
                                </p>
                                <div className="flex items-center gap-2 text-indigo-500 font-black text-[10px] uppercase tracking-widest animate-bounce">
                                    <ChevronRight className="h-4 w-4 rotate-90" />
                                    Configure parameters in the sidebar
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
