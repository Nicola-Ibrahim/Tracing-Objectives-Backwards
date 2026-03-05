"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { getDatasets, generateCandidates } from "@/features/inverse/api";
import { getDatasetDetails } from "@/features/dataset/api";
import { GenerateCandidatesForm } from "@/features/inverse/components/GenerateCandidatesForm";
import { CandidateExplorer } from "@/features/inverse/components/CandidateExplorer";
import { DatasetDetails } from "@/features/dataset/types";
import { CandidateGenerationRequest, CandidateGenerationResponse } from "@/features/inverse/types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Settings2, BarChart3, Loader2, Sparkles, AlertCircle } from "lucide-react";
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
        <div className="space-y-6 max-w-7xl mx-auto pb-10">
            <div className="flex flex-col gap-1">
                <h1 className="text-3xl font-bold tracking-tight text-slate-900 font-sans leading-8 flex items-center gap-2">
                    <Sparkles className="h-7 w-7 text-indigo-600" />
                    Candidate Generation
                </h1>
                <p className="text-slate-500 font-medium">Generate decision vectors targeting specific objective profiles.</p>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                <TabsList className="bg-slate-100 p-1">
                    <TabsTrigger value="generation" className="flex items-center gap-2">
                        <Settings2 className="h-4 w-4" />
                        Parameters
                    </TabsTrigger>
                    <TabsTrigger value="explorer" className="flex items-center gap-2" disabled={!selectedDataset}>
                        <BarChart3 className="h-4 w-4" />
                        Explorer
                    </TabsTrigger>
                </TabsList>

                <div className={activeTab === "generation" ? "block animate-in fade-in duration-300" : "hidden"}>
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <Card className="lg:col-span-1 border-slate-200 shadow-sm overflow-hidden h-fit">
                                <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                                    <CardTitle className="text-lg font-semibold text-slate-800">Generation Config</CardTitle>
                                    <CardDescription className="text-slate-500 text-sm">Target constraints and cohort settings.</CardDescription>
                                </CardHeader>
                                <CardContent className="p-6">
                                    {loadingDatasets ? (
                                        <div className="flex flex-col items-center justify-center py-10 gap-3">
                                            <Loader2 className="h-8 w-8 animate-spin text-slate-400" />
                                            <span className="text-slate-500 text-sm italic">Synchronizing datasets...</span>
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

                            <div className="lg:col-span-2 space-y-6">
                                {mutation.isError && (
                                    <Alert variant="destructive">
                                        <AlertCircle className="h-4 w-4" />
                                        <AlertTitle>Generation Failed</AlertTitle>
                                        <AlertDescription>
                                            {(mutation.error as any)?.response?.data?.detail || "An unexpected error occurred while generating candidates."}
                                        </AlertDescription>
                                    </Alert>
                                )}

                                {(result || datasetDetails) ? (
                                    <Card className="border-indigo-100 shadow-md">
                                        <CardHeader className="bg-indigo-50/30 border-b border-indigo-100 py-4">
                                            <CardTitle className="text-sm font-bold text-indigo-900 uppercase tracking-wider">
                                                {result ? "Generation Results" : "Dataset Preview"}
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent className="p-6">
                                            <CandidateExplorer
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
                    </div>
                </div>

                <div className={activeTab === "explorer" ? "pt-2 block animate-in fade-in duration-300" : "hidden"}>
                    {(result || datasetDetails) && (
                        <CandidateExplorer
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
