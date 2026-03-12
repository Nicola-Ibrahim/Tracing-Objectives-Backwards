"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { getDatasets, generateCandidates } from "@/features/inverse/api";
import { getDatasetDetails } from "@/features/dataset/api";
import { GenerateCandidatesForm } from "@/features/inverse/components/GenerateCandidatesForm";
import { CandidateManifoldChart } from "@/features/inverse/components/CandidateManifoldChart";
import { DatasetDetails } from "@/features/dataset/types";
import { CandidateGenerationRequest, CandidateGenerationResponse } from "@/features/inverse/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Settings2, BarChart3, Loader2, Sparkles, AlertCircle, Target, Database, Blocks } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function GeneratePage() {
    const [result, setResult] = useState<CandidateGenerationResponse | null>(null);

    const { data: datasets = [], isLoading: loadingDatasets } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
        select: (data) => data.map((d) => d.name),
    });

    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [activeTab, setActiveTab] = useState<string>("generation");

    const { data: datasetDetails, isLoading: loadingDetails } = useQuery({
        queryKey: ["dataset-details", selectedDataset],
        queryFn: () => getDatasetDetails(selectedDataset),
        enabled: !!selectedDataset,
    });

    const mutation = useMutation({
        mutationFn: generateCandidates,
        onSuccess: (data) => {
            setResult(data);
        },
    });

    const handleGenerate = async (params: CandidateGenerationRequest) => {
        await mutation.mutateAsync(params);
    };

    return (
        <div className="space-y-8 max-w-7xl mx-auto pb-16 px-4 md:px-0">
            <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col gap-2 relative"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-500 rounded-lg">
                        <Sparkles className="h-6 w-6 text-white" />
                    </div>
                    <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-foreground via-foreground/90 to-foreground/80 font-sans">
                        Candidate Generation
                    </h1>
                </div>
                <p className="text-muted-foreground font-medium ml-12 italic">Generate decision vectors targeting specific objective profiles.</p>
                <div className="absolute -top-10 -right-10 opacity-5 pointer-events-none">
                    <Target className="h-64 w-64 text-indigo-500 rotate-12" />
                </div>
            </motion.div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
                <TabsList className="bg-muted p-1 rounded-xl border border-border/50 backdrop-blur-sm">
                    <TabsTrigger value="generation" className="flex items-center gap-2 px-6 rounded-lg data-[state=active]:bg-background data-[state=active]:text-indigo-500 font-bold transition-all">
                        <Settings2 className="h-4 w-4" />
                        Parameters
                    </TabsTrigger>
                    <TabsTrigger value="explorer" className="flex items-center gap-2 px-6 rounded-lg data-[state=active]:bg-background data-[state=active]:text-indigo-500 font-bold transition-all" disabled={!selectedDataset}>
                        <BarChart3 className="h-4 w-4" />
                        Explorer
                    </TabsTrigger>
                </TabsList>

                <div className={activeTab === "generation" ? "block" : "hidden"}>
                    <motion.div 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-8"
                    >
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                            <Card className="lg:col-span-1 border-border rounded-3xl bg-card transition-all">
                                <CardHeader className="bg-muted/30 border-b border-border py-6 px-8 flex flex-row items-center justify-between space-y-0">
                                    <div className="space-y-1">
                                        <CardTitle className="text-xl font-black text-foreground tracking-tight uppercase">Configuration</CardTitle>
                                        <CardDescription className="text-muted-foreground text-[10px] font-black uppercase tracking-[0.2em] opacity-70">Engine Target Settings</CardDescription>
                                    </div>
                                    <div className="bg-background p-2 rounded-xl border border-border">
                                        <Blocks className="h-5 w-5 text-indigo-500" />
                                    </div>
                                </CardHeader>
                                <CardContent className="p-8">
                                    {loadingDatasets ? (
                                        <div className="flex flex-col items-center justify-center py-12 gap-4">
                                            <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                                            <span className="text-muted-foreground font-black uppercase tracking-widest text-[10px] animate-pulse">Syncing Registry...</span>
                                        </div>
                                    ) : (
                                        <GenerateCandidatesForm
                                            datasets={datasets}
                                            onSubmit={handleGenerate}
                                            isLoading={mutation.isPending}
                                            onDatasetChange={setSelectedDataset}
                                        />
                                    )}
                                </CardContent>
                            </Card>

                            <div className="lg:col-span-2 space-y-8">
                                {mutation.isError && (
                                    <Alert variant="destructive" className="rounded-2xl border-destructive/20 bg-destructive/10 border-l-4 border-l-destructive">
                                        <AlertCircle className="h-4 w-4 text-destructive" />
                                        <AlertTitle className="font-bold text-destructive">Generation Stopped</AlertTitle>
                                        <AlertDescription className="font-medium text-destructive/80">
                                            {(mutation.error as any)?.response?.data?.detail || "An unexpected error occurred while generating candidates."}
                                        </AlertDescription>
                                    </Alert>
                                )}

                                {(result || datasetDetails) ? (
                                    <Card className="border-border rounded-[2rem] overflow-hidden bg-card/50 backdrop-blur-sm">
                                        <CardHeader className="bg-muted/30 border-b border-border py-4 px-6 flex justify-between items-center flex-row">
                                            <CardTitle className="text-[10px] font-black text-foreground uppercase tracking-[0.2em] opacity-70">
                                                {result ? "Generation Results" : "Dataset Preview"}
                                            </CardTitle>
                                            <div className="flex items-center gap-2">
                                                <div className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                                                <span className="text-[8px] font-black text-indigo-500 uppercase tracking-widest">Live Manifold</span>
                                            </div>
                                        </CardHeader>
                                        <CardContent className="p-6">
                                            <CandidateManifoldChart
                                                data={result || undefined}
                                                backgroundX={datasetDetails?.X}
                                                backgroundY={datasetDetails?.y}
                                            />
                                        </CardContent>
                                    </Card>
                                ) : (
                                    <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-border rounded-[2.5rem] bg-muted/10 p-10 text-center group transition-all hover:bg-muted/20">
                                        <div className="bg-background p-6 rounded-full border border-border mb-6 group-hover:scale-110 transition-transform ring-8 ring-muted/50">
                                            <Sparkles className="h-10 w-10 text-muted-foreground/30" />
                                        </div>
                                        <h3 className="text-xl font-black text-foreground uppercase tracking-tight mb-2">Ready for Inference</h3>
                                        <p className="text-muted-foreground max-w-sm mb-6 font-medium italic">Select a dataset and a trained engine to start generating inverse candidates.</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                </div>

                <div className={activeTab === "explorer" ? "pt-2 block animate-in fade-in duration-300" : "hidden"}>
                    {(result || datasetDetails) && (
                        <CandidateManifoldChart
                            data={result || undefined}
                            backgroundX={datasetDetails?.X}
                            backgroundY={datasetDetails?.y}
                        />
                    )}
                </div>
            </Tabs>
        </div>
    );
}
