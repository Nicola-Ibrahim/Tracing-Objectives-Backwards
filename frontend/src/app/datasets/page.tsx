"use client";

import { useQuery } from "@tanstack/react-query";
import { getDatasets } from "@/features/inverse/api";
import { getDatasetDetails, generateDataset } from "@/features/dataset/api";
import { DatasetPlot } from "@/features/dataset/components/DatasetPlot";
import { GenerateDatasetForm } from "@/features/dataset/components/GenerateDatasetForm";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Database, FileText, Activity, Layers, Loader2, Search, X, Wand2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export default function DatasetsPage() {
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedDatasetName, setSelectedDatasetName] = useState<string | null>(null);
    const [showGenerator, setShowGenerator] = useState(false);
    const queryClient = useQueryClient();

    const { data: datasets = [], isLoading } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
    });

    const { data: selectedDetails, isLoading: isLoadingDetails } = useQuery({
        queryKey: ["dataset-details", selectedDatasetName],
        queryFn: () => getDatasetDetails(selectedDatasetName!),
        enabled: !!selectedDatasetName,
    });

    const generateMutation = useMutation({
        mutationFn: generateDataset,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["datasets"] });
            setShowGenerator(false);
        },
    });

    const filteredDatasets = datasets.filter(d =>
        d.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="space-y-6 max-w-7xl mx-auto pb-10">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div className="flex flex-col gap-1">
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
                        <Database className="h-7 w-7 text-indigo-600" />
                        Data Hub
                    </h1>
                    <p className="text-slate-500 font-medium">Manage and explore your multi-objective datasets.</p>
                </div>

                <div className="flex items-center gap-3">
                    <Button
                        onClick={() => setShowGenerator(!showGenerator)}
                        variant={showGenerator ? "secondary" : "default"}
                        className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold"
                    >
                        <Wand2 className="h-4 w-4 mr-2" />
                        {showGenerator ? "Hide Generator" : "New Dataset"}
                    </Button>
                    <div className="relative w-full md:w-64">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                        <Input
                            placeholder="Search datasets..."
                            className="pl-9 bg-white border-slate-200"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>
                </div>
            </div>

            {showGenerator && (
                <div className="max-w-2xl mx-auto animate-in zoom-in-95 duration-200">
                    <GenerateDatasetForm
                        onSubmit={(vals) => generateMutation.mutate(vals)}
                        isLoading={generateMutation.isPending}
                    />
                </div>
            )}

            {selectedDatasetName && (
                <div className="bg-slate-50/50 border border-slate-200 rounded-xl p-6 relative animate-in fade-in slide-in-from-top-4 duration-300">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="absolute right-4 top-4 hover:bg-slate-200"
                        onClick={() => setSelectedDatasetName(null)}
                    >
                        <X className="h-4 w-4" />
                    </Button>

                    <div className="flex items-center gap-3 mb-6">
                        <div className="h-10 w-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                            <Activity className="h-5 w-5 text-indigo-600" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-slate-900">{selectedDatasetName}</h2>
                            <p className="text-xs text-slate-500 font-medium">Detailed visualization of decision and objective spaces</p>
                        </div>
                    </div>

                    {isLoadingDetails ? (
                        <div className="flex items-center justify-center h-[300px]">
                            <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
                            <span className="ml-3 text-slate-500">Loading spatial data...</span>
                        </div>
                    ) : selectedDetails ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <DatasetPlot
                                title="Decision Space (X)"
                                data={selectedDetails.X}
                                labelX="X[0]"
                                labelY="X[1]"
                            />
                            <DatasetPlot
                                title="Objective Space (y)"
                                data={selectedDetails.y}
                                labelX="y[0]"
                                labelY="y[1]"
                            />
                        </div>
                    ) : null}
                </div>
            )}

            {isLoading ? (
                <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/10">
                    <Loader2 className="h-8 w-8 animate-spin text-indigo-600 mb-4" />
                    <p className="text-slate-500 font-medium animate-pulse">Synchronizing with registry...</p>
                </div>
            ) : datasets.length === 0 ? (
                <Card className="border-slate-200 shadow-sm">
                    <CardContent className="p-12 text-center">
                        <div className="bg-slate-50 p-6 rounded-full inline-block mb-4 ring-8 ring-slate-50/50">
                            <FileText className="h-10 w-10 text-slate-300" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900">No datasets found</h3>
                        <p className="text-slate-500 max-w-sm mx-auto mt-2">
                            Generate your first dataset in the "Generator" section or upload a CSV to start training inverse models.
                        </p>
                    </CardContent>
                </Card>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredDatasets.map((dataset) => (
                        <Card
                            key={dataset.name}
                            className={`border-slate-200 shadow-sm hover:shadow-lg transition-all duration-300 group overflow-hidden bg-white cursor-pointer ${selectedDatasetName === dataset.name ? 'ring-2 ring-indigo-500 shadow-md transform scale-[1.02]' : ''
                                }`}
                            onClick={() => setSelectedDatasetName(dataset.name === selectedDatasetName ? null : dataset.name)}
                        >
                            <CardHeader className="bg-slate-50/50 border-b border-slate-100 py-4 group-hover:bg-indigo-50/30 transition-colors">
                                <div className="flex items-center justify-between">
                                    <CardTitle className="text-base font-bold text-slate-800 truncate pr-2">
                                        {dataset.name}
                                    </CardTitle>
                                    <Badge variant="outline" className="bg-white text-[10px] font-mono shrink-0 shadow-sm">
                                        {dataset.name.slice(0, 8)}
                                    </Badge>
                                </div>
                            </CardHeader>
                            <CardContent className="p-6">
                                <div className="space-y-5">
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="space-y-1.5 p-3 bg-slate-50/50 rounded-lg border border-slate-100">
                                            <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Samples</p>
                                            <div className="flex items-center gap-2">
                                                <Activity className="h-4 w-4 text-indigo-500" />
                                                <span className="text-base font-bold text-slate-700">{dataset.n_samples.toLocaleString()}</span>
                                            </div>
                                        </div>
                                        <div className="space-y-1.5 p-3 bg-slate-50/50 rounded-lg border border-slate-100">
                                            <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Models</p>
                                            <div className="flex items-center gap-2">
                                                <Layers className="h-4 w-4 text-emerald-500" />
                                                <span className="text-base font-bold text-slate-700">{dataset.trained_engines_count}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="pt-2">
                                        <div className="flex items-center justify-between mb-2">
                                            <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Dimensionality</p>
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                            <div className="flex flex-col gap-1 flex-1 min-w-[100px] p-2 rounded border border-slate-100 bg-white">
                                                <span className="text-[9px] font-semibold text-slate-400 uppercase">Features (X)</span>
                                                <span className="text-sm font-mono text-slate-600">{dataset.n_features} dims</span>
                                            </div>
                                            <div className="flex flex-col gap-1 flex-1 min-w-[100px] p-2 rounded border border-slate-100 bg-white">
                                                <span className="text-[9px] font-semibold text-slate-400 uppercase">Objectives (y)</span>
                                                <span className="text-sm font-mono text-slate-600">{dataset.n_objectives} dims</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ))}

                    {filteredDatasets.length === 0 && searchQuery && (
                        <div className="col-span-full py-20 text-center">
                            <p className="text-slate-500 italic">No datasets match "{searchQuery}"</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
