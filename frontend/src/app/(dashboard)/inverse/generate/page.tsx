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
                    <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-200">
                        <Sparkles className="h-6 w-6 text-white" />
                    </div>
                    <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-slate-900 via-indigo-900 to-indigo-800 font-sans">
                        Candidate Generation
                    </h1>
                </div>
                <p className="text-slate-500 font-medium ml-12">Generate decision vectors targeting specific objective profiles.</p>
                <div className="absolute -top-10 -right-10 opacity-5 pointer-events-none">
                    <Target className="h-64 w-64 text-indigo-900 rotate-12" />
                </div>
            </motion.div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
                <TabsList className="bg-slate-100/50 p-1 rounded-xl border border-slate-200/60 backdrop-blur-sm">
                    <TabsTrigger value="generation" className="flex items-center gap-2 px-6 rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-indigo-600 font-bold transition-all">
                        <Settings2 className="h-4 w-4" />
                        Parameters
                    </TabsTrigger>
                    <TabsTrigger value="explorer" className="flex items-center gap-2 px-6 rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm data-[state=active]:text-indigo-600 font-bold transition-all" disabled={!selectedDataset}>
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
                            <Card className="lg:col-span-1 border-slate-200/60 shadow-lg shadow-slate-200/40 overflow-hidden h-fit rounded-3xl bg-white/80 backdrop-blur-sm">
                                <CardHeader className="bg-slate-50/40 border-b border-slate-100 py-6 px-8 flex flex-row items-center justify-between space-y-0">
                                    <div className="space-y-1">
                                        <CardTitle className="text-xl font-black text-slate-800 tracking-tight">Configuration</CardTitle>
                                        <CardDescription className="text-slate-500 text-xs font-bold uppercase tracking-widest opacity-60">Engine Target Settings</CardDescription>
                                    </div>
                                    <div className="bg-white p-2 rounded-xl shadow-sm border border-slate-100">
                                        <Blocks className="h-5 w-5 text-indigo-500" />
                                    </div>
                                </CardHeader>
                                <CardContent className="p-8">
                                    {loadingDatasets ? (
                                        <div className="flex flex-col items-center justify-center py-12 gap-4">
                                            <Loader2 className="h-10 w-10 animate-spin text-indigo-500" />
                                            <span className="text-slate-400 font-bold uppercase tracking-widest text-[10px] animate-pulse">Syncing Registry...</span>
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
                                    <Alert variant="destructive" className="rounded-2xl border-rose-200 bg-rose-50/50 border-l-4 border-l-rose-500 shadow-sm">
                                        <AlertCircle className="h-4 w-4 text-rose-600" />
                                        <AlertTitle className="font-bold">Generation Stopped</AlertTitle>
                                        <AlertDescription className="font-medium text-rose-800">
                                            {(mutation.error as any)?.response?.data?.detail || "An unexpected error occurred while generating candidates."}
                                        </AlertDescription>
                                    </Alert>
                                )}

                                {(result || datasetDetails) ? (
                                    <Card className="border-indigo-100 shadow-md overflow-hidden w-full min-w-0">
                                        <CardHeader className="bg-indigo-50/30 border-b border-indigo-100 py-4">
                                            <CardTitle className="text-sm font-bold text-indigo-900 uppercase tracking-wider">
                                                {result ? "Generation Results" : "Dataset Preview"}
                                            </CardTitle>
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
                                    <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/10 p-10 text-center">
                                        <div className="bg-white p-4 rounded-full shadow-sm mb-4">
                                            <Sparkles className="h-8 w-8 text-slate-300" />
                                        </div>
                                        <h3 className="text-lg font-semibold text-slate-900 mb-2">Ready for Inference</h3>
                                        <p className="text-slate-500 max-w-sm mb-6">Select a dataset and a trained engine to start generating inverse candidates.</p>
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
