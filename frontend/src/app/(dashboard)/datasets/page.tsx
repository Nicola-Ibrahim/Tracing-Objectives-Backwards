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

    const { data: trainDetails, isLoading: isLoadingTrain } = useQuery({
        queryKey: ["dataset-details", selectedDatasetName, "train"],
        queryFn: () => getDatasetDetails(selectedDatasetName!, "train"),
        enabled: !!selectedDatasetName,
    });

    const { data: testDetails, isLoading: isLoadingTest } = useQuery({
        queryKey: ["dataset-details", selectedDatasetName, "test"],
        queryFn: () => getDatasetDetails(selectedDatasetName!, "test"),
        enabled: !!selectedDatasetName,
    });

    const isLoadingDetails = isLoadingTrain || isLoadingTest;

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
        <div className="space-y-8 max-w-full mx-auto pb-16 px-4 md:px-0">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 relative">
                <motion.div 
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-2 relative"
                >
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-600 rounded-lg">
                            <Database className="h-6 w-6 text-white" />
                        </div>
                        <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-foreground via-foreground/90 to-foreground/80 font-heading uppercase">
                            Data Hub
                        </h1>
                    </div>
                    <p className="text-muted-foreground font-medium ml-12 italic">Manage and explore your multi-objective datasets.</p>
                    <div className="absolute -top-10 -right-20 opacity-5 pointer-events-none">
                        <Database className="h-48 w-48 text-indigo-500 rotate-6" />
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
                                className="font-bold transition-all hover:scale-105 active:scale-95"
                            >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete ({selectedNames.length})
                            </Button>
                        )}
                        <Button
                            onClick={() => setShowGenerator(true)}
                            className="bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600 text-white font-bold transition-all hover:scale-105 active:scale-95"
                        >
                            <Wand2 className="h-4 w-4 mr-2" />
                            New Dataset
                        </Button>
                        <div className="relative w-full md:w-64">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                            <Input
                                placeholder="Search datasets..."
                                className="pl-9 bg-background border-border focus:ring-2 focus:ring-indigo-500/10 transition-all"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* Individual Delete Confirmation */}
            <Dialog open={!!datasetToDelete} onOpenChange={(open) => !open && setDatasetToDelete(null)}>
                <DialogContent className="sm:max-w-[425px] bg-card border-border">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-rose-500 font-bold font-heading">
                            <AlertCircle className="h-5 w-5" />
                            Confirm Deletion
                        </DialogTitle>
                        <DialogDescription className="py-2 text-muted-foreground font-medium">
                            Are you sure you want to delete <span className="font-bold text-foreground">"{datasetToDelete}"</span>?
                            This will also remove all its associated trained engines.
                            This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex justify-end gap-3 pt-4">
                        <Button variant="outline" onClick={() => setDatasetToDelete(null)} className="border-border">
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
                <DialogContent className="sm:max-w-[425px] bg-card border-border">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-rose-500">
                            <AlertCircle className="h-5 w-5" />
                            Confirm Bulk Deletion
                        </DialogTitle>
                        <DialogDescription className="py-2 text-muted-foreground">
                            You are about to delete <span className="font-bold text-foreground">{selectedNames.length}</span> datasets.
                            This will also remove all their associated trained engines.
                            This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex justify-end gap-3 pt-4">
                        <Button variant="outline" onClick={() => setIsBulkDeleting(false)} className="border-border">
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
                <DialogContent className="sm:max-w-[750px] p-0 border-border bg-card h-[85vh] flex flex-col">
                    <DialogHeader className="p-6 border-b border-border shrink-0">
                        <DialogTitle className="text-xl font-bold flex items-center gap-2 text-foreground uppercase tracking-tight font-heading">
                            <Wand2 className="h-5 w-5 text-indigo-500" />
                            Generate New Dataset
                        </DialogTitle>
                        <DialogDescription className="text-muted-foreground font-medium italic">
                            Configure parameters for multi-objective dataset synthesis.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-muted-foreground/10">
                        <GenerateDatasetForm
                            onSubmit={(vals) => generateMutation.mutate(vals)}
                            isLoading={generateMutation.isPending}
                        />
                    </div>
                </DialogContent>
            </Dialog>

            <Dialog open={!!selectedDatasetName} onOpenChange={(open) => !open && setSelectedDatasetName(null)}>
                <DialogContent className="sm:max-w-[1400px] w-[95vw] border-border bg-card p-0 overflow-hidden">
                    {selectedDatasetName && (
                        <div className="flex flex-col h-[90vh]">
                            <div className="bg-muted/30 border-b border-border p-6 shrink-0">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-4">
                                        <div className="h-12 w-12 bg-indigo-500/10 rounded-xl flex items-center justify-center shrink-0 border border-indigo-500/20">
                                            <Activity className="h-6 w-6 text-indigo-500" />
                                        </div>
                                        <div className="min-w-0">
                                            <h2 className="text-2xl font-bold text-foreground truncate tracking-tight uppercase font-heading">{selectedDatasetName}</h2>
                                            <div className="flex items-center gap-3 mt-1">
                                                <Badge variant="outline" className="bg-background text-indigo-500 border-indigo-500/20 text-[10px] font-mono">
                                                    ID: {selectedDatasetName.slice(0, 10)}
                                                </Badge>
                                                <p className="text-[10px] text-muted-foreground font-bold uppercase tracking-widest opacity-60 font-heading">Decision & Objective Space Visualization</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <Button variant="outline" size="sm" onClick={() => setSelectedDatasetName(null)} className="font-bold border-border rounded-xl h-10 px-6">
                                            Close Inspector
                                        </Button>
                                    </div>
                                </div>

                                    {((trainDetails || testDetails)?.metadata) && (
                                        <div className="mt-8 pt-6 border-t border-border/60">
                                            <div className="flex items-center gap-2 mb-4">
                                                <Layers className="h-4 w-4 text-indigo-500" />
                                                <h4 className="text-[10px] font-bold text-foreground uppercase tracking-[0.2em] font-heading">Synthesis Metadata</h4>
                                            </div>
                                            <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
                                                <div className="space-y-1 p-3 bg-muted/20 rounded-xl border border-border/50 hover:border-indigo-500/30 transition-colors group">
                                                    <p className="text-[8px] font-bold uppercase text-muted-foreground tracking-widest opacity-60 font-heading">Aggregate</p>
                                                    <p className="text-lg font-bold text-foreground tabular-nums group-hover:text-indigo-500 transition-colors font-mono">
                                                        {(trainDetails || testDetails)!.metadata.n_samples.toLocaleString()}
                                                    </p>
                                                </div>
                                                <div className="space-y-1 p-3 bg-indigo-500/5 rounded-xl border border-indigo-500/10 hover:border-indigo-500/30 transition-colors group">
                                                    <p className="text-[8px] font-bold uppercase text-indigo-500 tracking-widest opacity-70 font-heading">Training</p>
                                                    <p className="text-lg font-bold text-foreground tabular-nums group-hover:text-indigo-500 transition-colors font-mono">
                                                        {(trainDetails || testDetails)!.metadata.n_train.toLocaleString()}
                                                    </p>
                                                </div>
                                                <div className="space-y-1 p-3 bg-emerald-500/5 rounded-xl border border-emerald-500/10 hover:border-emerald-500/30 transition-colors group">
                                                    <p className="text-[8px] font-bold uppercase text-emerald-500 tracking-widest opacity-70 font-heading">Testing</p>
                                                    <p className="text-lg font-bold text-foreground tabular-nums group-hover:text-emerald-500 transition-colors font-mono">
                                                        {(trainDetails || testDetails)!.metadata.n_test.toLocaleString()}
                                                    </p>
                                                </div>
                                                <div className="space-y-1 p-3 bg-muted/20 rounded-xl border border-border/50 hover:border-indigo-500/30 transition-colors group">
                                                    <p className="text-[8px] font-bold uppercase text-muted-foreground tracking-widest opacity-60 font-heading">Split Ratio</p>
                                                    <p className="text-lg font-bold text-foreground tabular-nums group-hover:text-indigo-500 transition-colors font-mono">
                                                        {((trainDetails || testDetails)!.metadata.split_ratio * 100).toFixed(0)}%
                                                    </p>
                                                </div>
                                                <div className="space-y-1 p-3 bg-muted/20 rounded-xl border border-border/50 hover:border-indigo-500/30 transition-colors group">
                                                    <p className="text-[8px] font-bold uppercase text-muted-foreground tracking-widest opacity-60 font-heading">Random Seed</p>
                                                    <p className="text-lg font-bold text-foreground tabular-nums group-hover:text-indigo-500 transition-colors font-mono">
                                                        {(trainDetails || testDetails)!.metadata.random_state}
                                                    </p>
                                                </div>
                                                <div className="space-y-1 p-3 bg-muted/20 rounded-xl border border-border/50 hover:border-indigo-500/30 transition-colors group">
                                                    <p className="text-[8px] font-bold uppercase text-muted-foreground tracking-widest opacity-60 font-heading">Generation</p>
                                                    <p className="text-xs font-bold text-foreground tabular-nums truncate group-hover:text-indigo-500 transition-colors font-mono">
                                                        {new Date((trainDetails || testDetails)!.metadata.created_at).toLocaleDateString(undefined, {
                                                            year: 'numeric',
                                                            month: 'short',
                                                            day: 'numeric'
                                                        })}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                <div className="flex-1 overflow-y-auto p-12 bg-background/50 backdrop-blur-sm scrollbar-thin scrollbar-thumb-indigo-500/10 scrollbar-track-transparent">
                                {isLoadingDetails ? (
                                    <div className="flex flex-col items-center justify-center h-[500px]">
                                        <div className="relative">
                                            <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full scale-150 animate-pulse" />
                                            <Loader2 className="h-12 w-12 animate-spin text-indigo-500 relative z-10" />
                                        </div>
                                        <span className="text-muted-foreground font-bold uppercase tracking-[0.3em] text-[10px] mt-8 opacity-60 font-heading">Decrypting Spatial Manifolds</span>
                                    </div>
                                ) : (trainDetails || testDetails) ? (
                                    <div className="max-w-7xl mx-auto space-y-24 pb-12">
                                        {/* Training Section */}
                                        {trainDetails && (
                                            <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700 ease-out">
                                                <div className="flex items-center gap-4">
                                                    <div className="px-4 py-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
                                                        <h3 className="text-[11px] font-bold uppercase tracking-[0.4em] text-indigo-500 font-heading">Training Manifold</h3>
                                                    </div>
                                                    <div className="h-px flex-1 bg-linear-to-r from-indigo-500/20 to-transparent" />
                                                </div>
                                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                                                    <div className="space-y-6">
                                                        <div className="flex items-center justify-between px-2">
                                                            <div className="flex items-center gap-2">
                                                                <div className="h-2 w-2 rounded-full bg-indigo-500" />
                                                                <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground font-heading">Decision Space (X)</span>
                                                            </div>
                                                            <Badge variant="outline" className="text-[9px] font-mono opacity-50 uppercase tracking-tighter">Latent Geometry</Badge>
                                                        </div>
                                                        <div className="bg-card/50 rounded-2xl border border-border/50 p-2 shadow-2xl shadow-indigo-500/5">
                                                            <DatasetChart
                                                                title=""
                                                                data={trainDetails.X}
                                                                labelX="Dimension 1"
                                                                labelY="Dimension 2"
                                                            />
                                                        </div>
                                                    </div>
                                                    <div className="space-y-6">
                                                        <div className="flex items-center justify-between px-2">
                                                            <div className="flex items-center gap-2">
                                                                <div className="h-2 w-2 rounded-full bg-emerald-500" />
                                                                <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground font-heading">Objective Space (y)</span>
                                                            </div>
                                                            <Badge variant="outline" className="text-[9px] font-mono opacity-50 uppercase tracking-tighter">Performance Pareto</Badge>
                                                        </div>
                                                        <div className="bg-card/50 rounded-2xl border border-border/50 p-2 shadow-2xl shadow-emerald-500/5">
                                                            <DatasetChart
                                                                title=""
                                                                data={trainDetails.y}
                                                                labelX="Objective 1"
                                                                labelY="Objective 2"
                                                            />
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* Testing Section */}
                                        {testDetails && testDetails.y.length > 0 && (
                                            <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-200 ease-out">
                                                <div className="flex items-center gap-4">
                                                    <div className="px-4 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                                                        <h3 className="text-[11px] font-bold uppercase tracking-[0.4em] text-emerald-500 font-heading">Testing Manifold</h3>
                                                    </div>
                                                    <div className="h-px flex-1 bg-linear-to-r from-emerald-500/20 to-transparent" />
                                                </div>
                                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                                                    <div className="space-y-6">
                                                        <div className="flex items-center justify-between px-2">
                                                            <div className="flex items-center gap-2">
                                                                <div className="h-2 w-2 rounded-full bg-indigo-400" />
                                                                <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground font-heading">Decision Space (X - Test)</span>
                                                            </div>
                                                        </div>
                                                        <div className="bg-card/50 rounded-2xl border border-border/50 p-2 shadow-2xl shadow-indigo-500/5">
                                                            <DatasetChart
                                                                title=""
                                                                data={testDetails.X}
                                                                labelX="Dimension 1"
                                                                labelY="Dimension 2"
                                                            />
                                                        </div>
                                                    </div>
                                                    <div className="space-y-6">
                                                        <div className="flex items-center justify-between px-2">
                                                            <div className="flex items-center gap-2">
                                                                <div className="h-2 w-2 rounded-full bg-emerald-400" />
                                                                <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground font-heading">Objective Space (y - Test)</span>
                                                            </div>
                                                        </div>
                                                        <div className="bg-card/50 rounded-2xl border border-border/50 p-2 shadow-2xl shadow-emerald-500/5">
                                                            <DatasetChart
                                                                title=""
                                                                data={testDetails.y}
                                                                labelX="Objective 1"
                                                                labelY="Objective 2"
                                                            />
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-[500px] text-muted-foreground/40 space-y-4">
                                        <AlertCircle className="h-12 w-12 opacity-20" />
                                        <p className="font-bold uppercase tracking-widest text-xs font-heading">Extraction Failed</p>
                                    </div>
                                )}
                            </div>

                            <div className="bg-muted/10 border-t border-border/40 p-3 flex justify-center shrink-0">
                                <p className="text-[9px] font-bold text-muted-foreground/40 uppercase tracking-[0.4em] font-heading">Proprietary Manifold Inspection Interface v4.0</p>
                            </div>
                        </div>
                    )}
                </DialogContent>
            </Dialog>

            {isLoading ? (
                <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-border rounded-[1rem] bg-muted/10">
                    <Loader2 className="h-8 w-8 animate-spin text-indigo-500 mb-4" />
                    <p className="text-muted-foreground font-bold uppercase tracking-[0.2em] text-[10px] animate-pulse font-heading">Synchronizing with registry...</p>
                </div>
            ) : datasets.length === 0 ? (
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                >
                    <Card className="border-border rounded-[1rem] overflow-hidden bg-card transition-all">
                        <CardContent className="p-16 text-center bg-linear-to-b from-card to-muted/20">
                            <div className="bg-background p-8 rounded-full inline-block mb-6 border border-border group-hover:scale-110 transition-transform ring-8 ring-muted/50">
                                <FileText className="h-12 w-12 text-muted-foreground/30" />
                            </div>
                             <h3 className="text-2xl font-bold text-foreground tracking-tight uppercase font-heading">No datasets found</h3>
                            <p className="text-muted-foreground max-w-sm mx-auto mt-4 font-medium italic leading-relaxed">
                                Generate your first dataset in the "Generator" section or upload a CSV to start training inverse models.
                            </p>
                            <Button
                                onClick={() => setShowGenerator(true)}
                                className="mt-10 bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600 text-white font-bold h-14 px-12 rounded-2xl"
                            >
                                <Sparkles className="h-4 w-4 mr-2" />
                                Create Discovery Asset
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
                                    className={`border-border transition-all duration-300 group overflow-hidden bg-card rounded-2xl border-l-[6px] ${
                                        selectedNames.includes(dataset.name) 
                                            ? 'border-l-indigo-500 bg-indigo-500/5 ring-1 ring-indigo-500/20' 
                                            : 'border-l-transparent hover:border-l-indigo-500/50'
                                    }`}
                                >
                                    <div className="flex flex-row items-center p-5 gap-4">
                                        <div className="px-2" onClick={(e) => e.stopPropagation()}>
                                            <Checkbox
                                                checked={selectedNames.includes(dataset.name)}
                                                onCheckedChange={() => toggleSelection(dataset.name)}
                                                className="h-5 w-5 rounded-md border-border data-[state=checked]:bg-indigo-500 data-[state=checked]:border-indigo-500"
                                            />
                                        </div>
                                        <div
                                            className="flex items-center gap-6 flex-1 min-w-0 cursor-pointer"
                                            onClick={() => setSelectedDatasetName(dataset.name)}
                                        >
                                            <div className="h-12 w-12 bg-muted/50 rounded-xl flex items-center justify-center shrink-0 group-hover:bg-indigo-500/10 border border-border group-hover:border-indigo-500/20 transition-all duration-300">
                                                <Database className="h-6 w-6 text-muted-foreground/50 group-hover:text-indigo-500 group-hover:scale-110 transition-all duration-300" />
                                            </div>
                                            <div className="min-w-0 flex-1">
                                                <div className="flex items-center gap-3">
                                                     <h3 className="text-base font-bold text-foreground truncate tracking-tight group-hover:text-indigo-500 transition-colors uppercase font-heading">
                                                         {dataset.name}
                                                     </h3>
                                                    <Badge variant="outline" className="text-[10px] h-5 px-2 bg-muted/50 font-bold text-muted-foreground/70 border-border group-hover:bg-background group-hover:text-indigo-500 transition-colors">
                                                        #{dataset.name.slice(0, 8).toUpperCase()}
                                                    </Badge>
                                                </div>
                                                <div className="flex items-center gap-2 mt-1.5">
                                                    <div className="h-1 w-1 rounded-full bg-indigo-500/50" />
                                                    <p className="text-[9px] text-muted-foreground/60 font-bold uppercase tracking-widest font-heading">Active Discovery Asset</p>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="hidden sm:flex items-center gap-10 px-8 border-x border-border/60 cursor-pointer h-12" onClick={() => setSelectedDatasetName(dataset.name)}>
                                            <div className="flex flex-col gap-1 min-w-[70px]">
                                                 <p className="text-[9px] uppercase font-bold text-muted-foreground/40 tracking-widest font-heading">Samples</p>
                                                 <div className="flex items-center gap-2">
                                                     <TrendingUp className="h-3.5 w-3.5 text-indigo-500" />
                                                     <span className="text-sm font-bold text-foreground tabular-nums font-mono">{dataset.metadata.n_samples.toLocaleString()}</span>
                                                 </div>
                                            </div>
                                            <div className="flex flex-col gap-1 min-w-[70px]">
                                                <p className="text-[9px] uppercase font-bold text-muted-foreground/40 tracking-widest font-heading">Engines</p>
                                                <div className="flex items-center gap-2">
                                                    <Layers className="h-3.5 w-3.5 text-emerald-500" />
                                                    <span className="text-sm font-bold text-foreground tabular-nums font-mono">{dataset.trained_engines_count}</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="hidden md:flex items-center gap-8 pr-6 cursor-pointer" onClick={() => setSelectedDatasetName(dataset.name)}>
                                             <div className="flex flex-col gap-1">
                                                 <p className="text-[9px] uppercase font-bold text-muted-foreground/30 tracking-tight font-heading">X-Space</p>
                                                 <span className="text-[10px] font-bold text-indigo-500 bg-indigo-500/5 px-2 py-0.5 rounded border border-indigo-500/20 font-mono">{dataset.n_features}D</span>
                                             </div>
                                             <div className="flex flex-col gap-1">
                                                 <p className="text-[9px] uppercase font-bold text-muted-foreground/30 tracking-tight font-heading">y-Space</p>
                                                 <span className="text-[10px] font-bold text-emerald-500 bg-emerald-500/5 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">{dataset.n_objectives}D</span>
                                             </div>
                                        </div>

                                        <div className="flex items-center justify-end gap-2 pr-2">
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-10 w-10 text-muted-foreground/40 hover:text-indigo-500 hover:bg-indigo-500/10 transition-all rounded-xl border border-transparent hover:border-indigo-500/20"
                                                onClick={() => setSelectedDatasetName(dataset.name)}
                                            >
                                                <Search className="h-5 w-5" />
                                            </Button>
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-10 w-10 text-muted-foreground/40 hover:text-rose-500 hover:bg-rose-500/10 transition-all rounded-xl border border-transparent hover:border-rose-500/20"
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
                            <div className="bg-muted/30 p-8 rounded-[1rem] ring-8 ring-muted/10">
                                <Search className="h-10 w-10 text-muted-foreground/30" />
                            </div>
                            <p className="text-muted-foreground font-bold uppercase tracking-widest text-xs font-heading">
                                No datasets match "<span className="text-indigo-500">{searchQuery}</span>"
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
