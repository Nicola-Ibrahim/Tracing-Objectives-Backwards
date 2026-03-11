"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { getDatasets, getDatasetDetails, generateDataset, deleteDatasets } from "@/features/dataset/api";
import { DatasetChart } from "@/features/dataset/components/DatasetChart";
import { GenerateDatasetForm } from "@/features/dataset/components/GenerateDatasetForm";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Database, FileText, Activity, Layers, Loader2, Search, X, Wand2, Trash2, AlertCircle, Sparkles, TrendingUp } from "lucide-react";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/components/ui/ToastContext";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "@/components/ui/dialog";

export default function DatasetsPage() {
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedDatasetName, setSelectedDatasetName] = useState<string | null>(null);
    const [showGenerator, setShowGenerator] = useState(false);
    const [selectedNames, setSelectedNames] = useState<string[]>([]);
    const [datasetToDelete, setDatasetToDelete] = useState<string | null>(null);
    const [isBulkDeleting, setIsBulkDeleting] = useState(false);

    const queryClient = useQueryClient();
    const { showToast } = useToast();

    const { data: datasets = [], isLoading } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
    });

    const { data: selectedDetails, isLoading: isLoadingDetails } = useQuery({
        queryKey: ["dataset-details", selectedDatasetName, "all"],
        queryFn: () => getDatasetDetails(selectedDatasetName!, "all"),
        enabled: !!selectedDatasetName,
    });

    const generateMutation = useMutation({
        mutationFn: generateDataset,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["datasets"] });
            setShowGenerator(false);
            showToast("Dataset generation started", "success");
        },
    });

    const deleteMutation = useMutation({
        mutationFn: (name: string) => deleteDatasets([name]),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["datasets"] });
            setDatasetToDelete(null);
            showToast("Dataset deleted successfully", "success");
        },
        onError: (err: any) => {
            showToast(`Deletion failed: ${err.message}`, "error");
        }
    });

    const bulkDeleteMutation = useMutation({
        mutationFn: deleteDatasets,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["datasets"] });
            setSelectedNames([]);
            setIsBulkDeleting(false);
            showToast("Datasets deleted successfully", "success");
        },
        onError: (err: any) => {
            showToast(`Bulk deletion failed: ${err.message}`, "error");
        }
    });

    const toggleSelection = (name: string) => {
        setSelectedNames(prev =>
            prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
        );
    };

    const filteredDatasets = datasets.filter(d =>
        d.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="space-y-8 max-w-7xl mx-auto pb-16 px-4 md:px-0">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 relative">
                <motion.div 
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-2 relative"
                >
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-200">
                            <Database className="h-6 w-6 text-white" />
                        </div>
                        <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-slate-900 via-indigo-900 to-indigo-800 font-sans">
                            Data Hub
                        </h1>
                    </div>
                    <p className="text-slate-500 font-medium ml-12">Manage and explore your multi-objective datasets.</p>
                    <div className="absolute -top-10 -right-20 opacity-5 pointer-events-none">
                        <Database className="h-48 w-48 text-indigo-900 rotate-6" />
                    </div>
                </motion.div>

                <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-3 relative z-10"
                >
                    <div className="flex items-center gap-3">
                        {selectedNames.length > 0 && (
                            <Button
                                variant="destructive"
                                onClick={() => setIsBulkDeleting(true)}
                                className="font-bold shadow-lg shadow-rose-100 transition-all hover:scale-105 active:scale-95"
                            >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete ({selectedNames.length})
                            </Button>
                        )}
                        <Button
                            onClick={() => setShowGenerator(true)}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold shadow-lg shadow-indigo-100 transition-all hover:scale-105 active:scale-95"
                        >
                            <Wand2 className="h-4 w-4 mr-2" />
                            New Dataset
                        </Button>
                        <div className="relative w-full md:w-64">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                            <Input
                                placeholder="Search datasets..."
                                className="pl-9 bg-white border-slate-200 shadow-sm focus:ring-2 focus:ring-indigo-100 transition-all"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* Individual Delete Confirmation */}
            <Dialog open={!!datasetToDelete} onOpenChange={(open) => !open && setDatasetToDelete(null)}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-red-600">
                            <AlertCircle className="h-5 w-5" />
                            Confirm Deletion
                        </DialogTitle>
                        <DialogDescription className="py-2">
                            Are you sure you want to delete <span className="font-bold text-slate-900">"{datasetToDelete}"</span>?
                            This will also remove all its associated trained engines.
                            This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex justify-end gap-3 pt-4">
                        <Button variant="outline" onClick={() => setDatasetToDelete(null)}>
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => datasetToDelete && deleteMutation.mutate(datasetToDelete)}
                            disabled={deleteMutation.isPending}
                        >
                            {deleteMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : "Delete Dataset"}
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>

            {/* Bulk Delete Confirmation */}
            <Dialog open={isBulkDeleting} onOpenChange={setIsBulkDeleting}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-red-600">
                            <AlertCircle className="h-5 w-5" />
                            Confirm Bulk Deletion
                        </DialogTitle>
                        <DialogDescription className="py-2">
                            You are about to delete <span className="font-bold text-slate-900">{selectedNames.length}</span> datasets.
                            This will also remove all their associated trained engines.
                            This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex justify-end gap-3 pt-4">
                        <Button variant="outline" onClick={() => setIsBulkDeleting(false)}>
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => bulkDeleteMutation.mutate(selectedNames)}
                            disabled={bulkDeleteMutation.isPending}
                        >
                            {bulkDeleteMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : `Delete ${selectedNames.length} Datasets`}
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>

            <Dialog open={showGenerator} onOpenChange={setShowGenerator}>
                <DialogContent className="sm:max-w-[750px] p-0 border-slate-200 shadow-2xl h-[85vh] flex flex-col">
                    <DialogHeader className="p-6 border-b border-slate-100 shrink-0">
                        <DialogTitle className="text-xl font-bold flex items-center gap-2">
                            <Wand2 className="h-5 w-5 text-indigo-600" />
                            Generate New Dataset
                        </DialogTitle>
                        <DialogDescription className="text-slate-500">
                            Configure parameters for multi-objective dataset synthesis.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-slate-200">
                        <GenerateDatasetForm
                            onSubmit={(vals) => generateMutation.mutate(vals)}
                            isLoading={generateMutation.isPending}
                        />
                    </div>
                </DialogContent>
            </Dialog>

            <Dialog open={!!selectedDatasetName} onOpenChange={(open) => !open && setSelectedDatasetName(null)}>
                <DialogContent className="sm:max-w-[1000px] border-slate-200 bg-white shadow-2xl p-0 overflow-hidden">
                    {selectedDatasetName && (
                        <div className="flex flex-col h-full max-h-[90vh]">
                            <div className="bg-slate-50 border-b border-slate-200 p-6 text-indigo-600">
                                <div className="flex items-center gap-4">
                                    <div className="h-12 w-12 bg-indigo-100 rounded-xl flex items-center justify-center shrink-0">
                                        <Activity className="h-6 w-6 text-indigo-600" />
                                    </div>
                                    <div className="min-w-0">
                                        <h2 className="text-2xl font-bold text-slate-900 truncate tracking-tight">{selectedDatasetName}</h2>
                                        <div className="flex items-center gap-3 mt-1">
                                            <Badge variant="outline" className="bg-white text-indigo-600 border-indigo-100 text-[10px] font-mono shadow-sm">
                                                ID: {selectedDatasetName.slice(0, 10)}
                                            </Badge>
                                            <p className="text-xs text-slate-500 font-medium">Decision & Objective Space Visualization</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1 overflow-y-auto p-8 bg-white/50">
                                {isLoadingDetails ? (
                                    <div className="flex flex-col items-center justify-center h-[400px]">
                                        <Loader2 className="h-10 w-10 animate-spin text-indigo-600 mb-4" />
                                        <span className="text-slate-500 font-medium tracking-wide">Fetching spatial matrices...</span>
                                    </div>
                                ) : selectedDetails ? (
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                        <div className="space-y-4">
                                            <DatasetChart
                                                title="Decision Space (X)"
                                                data={selectedDetails.X}
                                                labelX="X[0] (Feature 1)"
                                                labelY="X[1] (Feature 2)"
                                            />
                                        </div>
                                        <div className="space-y-4">
                                            <DatasetChart
                                                title="Objective Space (y)"
                                                data={selectedDetails.y}
                                                labelX="y[0] (Objective 1)"
                                                labelY="y[1] (Objective 2)"
                                            />
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex items-center justify-center h-[400px] text-slate-400">
                                        Failed to load dataset details.
                                    </div>
                                )}
                            </div>

                            <div className="bg-slate-50 border-t border-slate-100 p-4 flex justify-end gap-3">
                                <Button variant="outline" size="sm" onClick={() => setSelectedDatasetName(null)} className="font-semibold px-6">
                                    Close Inspector
                                </Button>
                            </div>
                        </div>
                    )}
                </DialogContent>
            </Dialog>

            {isLoading ? (
                <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-slate-200 rounded-3xl bg-slate-50/10">
                    <Loader2 className="h-8 w-8 animate-spin text-indigo-500 mb-4" />
                    <p className="text-slate-500 font-bold uppercase tracking-widest text-[10px] animate-pulse">Synchronizing with registry...</p>
                </div>
            ) : datasets.length === 0 ? (
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                >
                    <Card className="border-slate-200 shadow-sm rounded-3xl overflow-hidden">
                        <CardContent className="p-16 text-center bg-linear-to-b from-white to-slate-50/50">
                            <div className="bg-white p-8 rounded-full inline-block mb-6 shadow-xl shadow-slate-200/50 ring-8 ring-slate-50">
                                <FileText className="h-12 w-12 text-slate-300" />
                            </div>
                            <h3 className="text-2xl font-black text-slate-900 tracking-tight">No datasets found</h3>
                            <p className="text-slate-500 max-w-sm mx-auto mt-3 font-medium leading-relaxed">
                                Generate your first dataset in the "Generator" section or upload a CSV to start training inverse models.
                            </p>
                            <Button
                                onClick={() => setShowGenerator(true)}
                                className="mt-8 bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-8 shadow-lg shadow-indigo-100"
                            >
                                <Sparkles className="h-4 w-4 mr-2" />
                                Create Discovery Dataset
                            </Button>
                        </CardContent>
                    </Card>
                </motion.div>
            ) : (
                <div className="flex flex-col gap-4">
                    <AnimatePresence initial={false}>
                        {filteredDatasets.map((dataset, index) => (
                            <motion.div
                                key={dataset.name}
                                layout
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0, transition: { delay: index * 0.03 } }}
                                exit={{ opacity: 0, scale: 0.95 }}
                            >
                                <Card
                                    className={`border-slate-200/60 shadow-md shadow-slate-200/30 hover:shadow-xl hover:shadow-slate-200/50 transition-all duration-300 group overflow-hidden bg-white rounded-2xl border-l-4 ${
                                        selectedNames.includes(dataset.name) 
                                            ? 'border-l-indigo-500 ring-2 ring-indigo-500/10 bg-indigo-50/5' 
                                            : 'border-l-transparent hover:border-l-indigo-400'
                                    }`}
                                >
                                    <div className="flex flex-row items-center p-5 gap-4">
                                        <div className="px-2" onClick={(e) => e.stopPropagation()}>
                                            <Checkbox
                                                checked={selectedNames.includes(dataset.name)}
                                                onCheckedChange={() => toggleSelection(dataset.name)}
                                                className="h-5 w-5 rounded-md border-slate-300 data-[state=checked]:bg-indigo-600 data-[state=checked]:border-indigo-600"
                                            />
                                        </div>
                                        <div
                                            className="flex items-center gap-5 flex-1 min-w-0 cursor-pointer"
                                            onClick={() => setSelectedDatasetName(dataset.name)}
                                        >
                                            <div className="h-12 w-12 bg-slate-50 rounded-xl flex items-center justify-center shrink-0 group-hover:bg-indigo-50 transition-all duration-300 shadow-inner">
                                                <Database className="h-6 w-6 text-slate-400 group-hover:text-indigo-600 group-hover:scale-110 transition-all duration-300" />
                                            </div>
                                            <div className="min-w-0 flex-1">
                                                <div className="flex items-center gap-3">
                                                    <h3 className="text-base font-bold text-slate-800 truncate tracking-tight group-hover:text-indigo-900 transition-colors">
                                                        {dataset.name}
                                                    </h3>
                                                    <Badge variant="outline" className="text-[10px] h-4.5 px-2 bg-slate-50 font-bold text-slate-400 border-slate-100 group-hover:bg-white group-hover:text-indigo-500 transition-colors">
                                                        {dataset.name.slice(0, 8)}
                                                    </Badge>
                                                </div>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <div className="h-1 w-1 rounded-full bg-slate-300" />
                                                    <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">Active Discovery Asset</p>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="hidden sm:flex items-center gap-10 px-8 border-x border-slate-100/60 cursor-pointer" onClick={() => setSelectedDatasetName(dataset.name)}>
                                            <div className="flex flex-col gap-1 min-w-[70px]">
                                                <p className="text-[9px] uppercase font-black text-slate-400 tracking-widest">Samples</p>
                                                <div className="flex items-center gap-2">
                                                    <TrendingUp className="h-3.5 w-3.5 text-indigo-500/80" />
                                                    <span className="text-sm font-extrabold text-slate-700 tabular-nums">{dataset.n_samples.toLocaleString()}</span>
                                                </div>
                                            </div>
                                            <div className="flex flex-col gap-1 min-w-[70px]">
                                                <p className="text-[9px] uppercase font-black text-slate-400 tracking-widest">Engines</p>
                                                <div className="flex items-center gap-2">
                                                    <Layers className="h-3.5 w-3.5 text-teal-500/80" />
                                                    <span className="text-sm font-extrabold text-slate-700 tabular-nums">{dataset.trained_engines_count}</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="hidden md:flex items-center gap-8 pr-6 cursor-pointer" onClick={() => setSelectedDatasetName(dataset.name)}>
                                            <div className="flex flex-col gap-1">
                                                <p className="text-[9px] uppercase font-bold text-slate-300 tracking-tighter">Feature Dim.</p>
                                                <span className="text-xs font-black text-slate-500 bg-slate-50 px-2 py-0.5 rounded border border-slate-100">{dataset.n_features}X</span>
                                            </div>
                                            <div className="flex flex-col gap-1">
                                                <p className="text-[9px] uppercase font-bold text-slate-300 tracking-tighter">Objective</p>
                                                <span className="text-xs font-black text-slate-500 bg-slate-50 px-2 py-0.5 rounded border border-slate-100">{dataset.n_objectives}y</span>
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-end gap-2">
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-10 w-10 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 transition-all rounded-xl"
                                                onClick={() => setSelectedDatasetName(dataset.name)}
                                            >
                                                <Search className="h-5 w-5" />
                                            </Button>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-10 w-10 text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-all rounded-xl"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setDatasetToDelete(dataset.name);
                                                }}
                                            >
                                                <Trash2 className="h-5 w-5" />
                                            </Button>
                                        </div>
                                    </div>
                                </Card>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {filteredDatasets.length === 0 && searchQuery && (
                        <div className="py-24 text-center">
                            <div className="bg-slate-50 p-6 rounded-full inline-block mb-4">
                                <Search className="h-8 w-8 text-slate-300" />
                            </div>
                            <p className="text-slate-500 font-bold tracking-tight">No datasets match "<span className="text-indigo-600">{searchQuery}</span>"</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
