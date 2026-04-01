"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { listAllEngines, deleteEngines, listEnginesForDataset } from "@/features/inverse/api";
import { EngineListItem } from "@/features/inverse/types";
import { getDatasets } from "@/features/dataset/api";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/components/ui/ToastContext";
import { Input } from "@/components/ui/input";
import {
    ChevronDown,
    ChevronUp,
    Cpu,
    Search,
    Trash2,
    AlertCircle,
    Calendar,
    Layers,
    Loader2,
    ChevronRight,
    Box,
    RefreshCcw,
    Info
} from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "@/components/ui/dialog";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { EngineDetailsInspector } from "@/features/inverse/components/EngineDetailsInspector";

export default function EnginesPage() {
    const [searchQuery, setSearchQuery] = useState("");
    const [datasetFilter, setDatasetFilter] = useState<string>("all");
    const [selectedEngines, setSelectedEngines] = useState<EngineListItem[]>([]);
    const [engineToDelete, setEngineToDelete] = useState<EngineListItem | null>(null);
    const [isBulkDeleting, setIsBulkDeleting] = useState(false);
    const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
    const [expandedDatasets, setExpandedDatasets] = useState<Record<string, boolean>>({});
    const [inspectingEngine, setInspectingEngine] = useState<EngineListItem | null>(null);

    const queryClient = useQueryClient();
    const { showToast } = useToast();

    const { data: engines = [], isLoading, refetch, isFetching } = useQuery({
        queryKey: ["engines", datasetFilter],
        queryFn: () => datasetFilter === "all" ? listAllEngines() : listEnginesForDataset(datasetFilter),
        refetchOnWindowFocus: true,
        staleTime: 0, // Ensure it's always considered stale so it refetches on mount
    });

    const { data: datasets = [] } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
    });

    const deleteMutation = useMutation({
        mutationFn: (e: EngineListItem) => deleteEngines([{
            dataset_name: e.dataset_name!,
            solver_type: e.solver_type,
            version: e.version
        }]),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ["engines"] });
            setEngineToDelete(null);

            // Check for API-level silent failures
            const errors = Array.isArray(data) ? (data as { status: string }[]).filter((r) => r.status === "error" || r.status === "not_found") : [];
            if (errors.length > 0) {
                showToast(`Some engines could not be deleted: ${errors[0].status}`, "error");
            } else {
                showToast("Engine deleted successfully", "success");
            }
        },
        onError: (err: Error) => {
            showToast(`Deletion failed: ${err.message}`, "error");
        }
    });

    const bulkDeleteMutation = useMutation({
        mutationFn: deleteEngines,
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ["engines"] });
            setIsBulkDeleting(false);

            // Check for API-level silent failures
            const errors = Array.isArray(data) ? (data as { status: string }[]).filter((r) => r.status === "error" || r.status === "not_found") : [];
            if (errors.length > 0) {
                showToast(`Failed to delete ${errors.length} engines. The others were deleted.`, "error");
                // Remove successful deletes from selection
                const successful = Array.isArray(data) ? (data as { status: string; dataset_name: string; solver_type: string; version: number }[]).filter((r) => r.status === "deleted") : [];
                setSelectedEngines(prev => prev.filter(p => !successful.some((s) => s.dataset_name === p.dataset_name && s.solver_type === p.solver_type && s.version === p.version)));
            } else {
                setSelectedEngines([]);
                showToast("Engines deleted successfully", "success");
            }
        },
        onError: (err: Error) => {
            showToast(`Bulk deletion failed: ${err.message}`, "error");
        }
    });

    const toggleSelection = (engine: EngineListItem) => {
        const id = `${engine.dataset_name}-${engine.solver_type}-${engine.version}`;
        setSelectedEngines(prev => {
            const exists = prev.some(e => `${e.dataset_name}-${e.solver_type}-${e.version}` === id);
            return exists
                ? prev.filter(e => `${e.dataset_name}-${e.solver_type}-${e.version}` !== id)
                : [...prev, {
                    dataset_name: engine.dataset_name,
                    solver_type: engine.solver_type,
                    version: engine.version,
                    created_at: engine.created_at
                } as EngineListItem];
        });
    };


    const filteredEngines = engines.filter(e =>
        e.solver_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (e.dataset_name && e.dataset_name.toLowerCase().includes(searchQuery.toLowerCase()))
    );

    // Grouping logic: Dataset -> Solver Type -> Versions
    const groupedEngines = useMemo(() => {
        const groups: Record<string, Record<string, EngineListItem[]>> = {};

        filteredEngines.forEach(engine => {
            const ds = engine.dataset_name || "Unknown Dataset";
            const st = engine.solver_type;

            if (!groups[ds]) groups[ds] = {};
            if (!groups[ds][st]) groups[ds][st] = [];

            groups[ds][st].push(engine);
        });

        // Sort versions within each group (newest first)
        Object.keys(groups).forEach(ds => {
            Object.keys(groups[ds]).forEach(st => {
                groups[ds][st].sort((a, b) => b.version - a.version);
            });
        });

        return groups;
    }, [filteredEngines]);

    const handleSelectAll = () => {
        if (selectedEngines.length === filteredEngines.length && filteredEngines.length > 0) {
            setSelectedEngines([]);
        } else {
            setSelectedEngines(filteredEngines.map(e => ({
                dataset_name: e.dataset_name,
                solver_type: e.solver_type,
                version: e.version,
                created_at: e.created_at
            } as EngineListItem)));
        }
    };

    const handleSelectGroup = (enginesInGroup: EngineListItem[], isCurrentlyAllSelected: boolean) => {
        if (isCurrentlyAllSelected) {
            // Deselect all in this group
            const groupIds = new Set(enginesInGroup.map(e => `${e.dataset_name}-${e.solver_type}-${e.version}`));
            setSelectedEngines(prev => prev.filter(e => !groupIds.has(`${e.dataset_name}-${e.solver_type}-${e.version}`)));
        } else {
            // Select all in this group (avoid duplicates)
            setSelectedEngines(prev => {
                const existingIds = new Set(prev.map(e => `${e.dataset_name}-${e.solver_type}-${e.version}`));
                const newItems = enginesInGroup
                    .filter(e => !existingIds.has(`${e.dataset_name}-${e.solver_type}-${e.version}`))
                    .map(e => ({
                        dataset_name: e.dataset_name,
                        solver_type: e.solver_type,
                        version: e.version,
                        created_at: e.created_at
                    } as EngineListItem));
                return [...prev, ...newItems];
            });
        }
    };

    const toggleGroupExpand = (groupKey: string) => {
        setExpandedGroups(prev => ({
            ...prev,
            [groupKey]: !prev[groupKey]
        }));
    };

    const toggleDatasetExpand = (datasetName: string) => {
        setExpandedDatasets(prev => ({
            ...prev,
            [datasetName]: !prev[datasetName]
        }));
    };

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
                            <Layers className="h-6 w-6 text-white" />
                        </div>
                        <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-foreground via-foreground/90 to-foreground/80 font-heading uppercase">
                            Engine Registry
                        </h1>
                    </div>
                    <p className="text-muted-foreground font-medium ml-12 italic">Manage and monitor high-fidelity inverse estimators.</p>
                    <div className="absolute -top-10 -right-20 opacity-5 pointer-events-none">
                        <Cpu className="h-48 w-48 text-indigo-500 rotate-6" />
                    </div>
                </motion.div>

                <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex flex-col md:flex-row items-center gap-3 relative z-10"
                >
                    <div className="flex items-center gap-3 w-full md:w-auto">
                        <div className="relative w-full md:w-64">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/40" />
                            <Input
                                placeholder="Filter registry..."
                                className="pl-10 bg-background border-border focus:ring-2 focus:ring-indigo-500/10 transition-all font-medium rounded-[1rem] h-11"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                        <Select value={datasetFilter} onValueChange={setDatasetFilter}>
                            <SelectTrigger className="w-full md:w-[180px] bg-background border-border font-bold text-[11px] uppercase tracking-wider text-foreground h-11 rounded-[1rem]">
                                <SelectValue placeholder="All Datasets" />
                            </SelectTrigger>
                            <SelectContent className="bg-popover border-border text-popover-foreground rounded-[1rem]">
                                <SelectItem value="all" className="font-bold">All Datasets</SelectItem>
                                {datasets.map((d) => (
                                    <SelectItem key={d.name} value={d.name} className="font-bold">{d.name}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    <div className="flex items-center gap-3 w-full md:w-auto">
                        {selectedEngines.length > 0 && (
                            <Button
                                variant="destructive"
                                onClick={() => setIsBulkDeleting(true)}
                                className="font-bold uppercase tracking-widest transition-all hover:scale-105 active:scale-95 px-6 h-11 rounded-[1rem] text-[10px]"
                            >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Decommission ({selectedEngines.length})
                            </Button>
                        )}
                        <Button
                            variant="outline"
                            size="icon"
                            onClick={() => refetch()}
                            disabled={isFetching}
                            className="bg-background border-border text-muted-foreground hover:text-indigo-500 hover:border-indigo-500/20 h-11 w-11 rounded-[1rem] transition-all active:scale-95 group"
                            title="Refresh Engines"
                        >
                            <RefreshCcw className={cn("h-4 w-4 transition-transform duration-700", isFetching ? "animate-spin" : "group-hover:rotate-180")} />
                        </Button>
                    </div>
                </motion.div>
            </div>
            
            <div className="flex items-center gap-3 bg-muted/30 border border-border/50 rounded-[1rem] px-4 h-11 transition-all hover:border-border w-fit">
                <Checkbox
                    id="select-all"
                    checked={filteredEngines.length > 0 && selectedEngines.length === filteredEngines.length}
                    onCheckedChange={handleSelectAll}
                    className="border-border/50 data-[state=checked]:bg-indigo-500 data-[state=checked]:border-indigo-500"
                />
                <label
                    htmlFor="select-all"
                    className="text-[10px] font-bold text-muted-foreground uppercase cursor-pointer select-none tracking-[0.2em]"
                >
                    Global Selection
                </label>
            </div>

            {/* Individual Delete Confirmation */}
            <Dialog open={!!engineToDelete} onOpenChange={(open) => !open && setEngineToDelete(null)}>
                <DialogContent className="sm:max-w-[425px] p-0 rounded-[2rem] overflow-hidden bg-card border-border">
                    <DialogHeader className="p-8 bg-destructive/5 border-b border-border space-y-3">
                        <DialogTitle className="flex items-center gap-3 text-destructive font-bold uppercase tracking-tight text-xl font-heading">
                            <AlertCircle className="h-6 w-6" />
                            Purge Engine
                        </DialogTitle>
                        <DialogDescription className="text-muted-foreground font-medium italic">
                             You are about to permanently delete <span className="font-bold text-foreground uppercase tracking-tight font-heading">{engineToDelete?.solver_type} <span className="font-mono text-sm">v{engineToDelete?.version}</span></span> from <span className="font-bold text-foreground italic">&ldquo;{engineToDelete?.dataset_name}&rdquo;</span>.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="p-8 bg-card flex justify-end gap-3">
                        <Button variant="ghost" onClick={() => setEngineToDelete(null)} className="rounded-[1rem] font-bold uppercase tracking-widest text-[10px] h-11 px-6 font-heading">
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => engineToDelete && deleteMutation.mutate(engineToDelete)}
                            disabled={deleteMutation.isPending}
                            className="rounded-[1rem] font-bold uppercase tracking-widest text-[10px] h-11 px-8 font-heading"
                        >
                            {deleteMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : "Confirm Purge"}
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>

            {/* Bulk Delete Confirmation */}
            <Dialog open={isBulkDeleting} onOpenChange={setIsBulkDeleting}>
                <DialogContent className="sm:max-w-[425px] p-0 rounded-[2rem] overflow-hidden bg-card border-border">
                    <DialogHeader className="p-8 bg-destructive/5 border-b border-border space-y-3">
                        <DialogTitle className="flex items-center gap-3 text-destructive font-bold uppercase tracking-tight text-xl font-heading">
                            <AlertCircle className="h-6 w-6" />
                            Bulk Decommission
                        </DialogTitle>
                        <DialogDescription className="text-muted-foreground font-medium italic">
                            You are about to decommission <span className="font-bold text-foreground uppercase tracking-tight">{selectedEngines.length}</span> trained engines across the registry. This operation is irreversible.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="p-8 bg-card flex justify-end gap-3">
                        <Button variant="ghost" onClick={() => setIsBulkDeleting(false)} className="rounded-[1rem] font-bold uppercase tracking-widest text-[10px] h-11 px-6 font-heading">
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => bulkDeleteMutation.mutate(selectedEngines)}
                            disabled={bulkDeleteMutation.isPending}
                            className="rounded-[1rem] font-bold uppercase tracking-widest text-[10px] h-11 px-8 font-heading"
                        >
                            {bulkDeleteMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : `Purge ${selectedEngines.length} Engines`}
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>

            {isLoading ? (
                <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-border rounded-[1rem] bg-muted/5 backdrop-blur-sm transition-all">
                    <Loader2 className="h-10 w-10 animate-spin text-indigo-500 mb-6" />
                    <p className="text-muted-foreground/60 font-bold uppercase tracking-[0.25em] text-[10px] animate-pulse font-heading">Scanning Engine Registry...</p>
                </div>
            ) : Object.keys(groupedEngines).length === 0 ? (
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                >
                    <Card className="border-border rounded-[1rem] overflow-hidden bg-card">
                        <CardContent className="p-20 text-center bg-linear-to-b from-card to-muted/10">
                            <div className="bg-background border border-border p-10 rounded-full inline-block mb-8 ring-8 ring-muted/5 transition-transform hover:scale-110 duration-500">
                                <Cpu className="h-14 w-14 text-muted-foreground/20" />
                            </div>
                            <h3 className="text-3xl font-bold text-foreground tracking-tight uppercase mb-4 font-heading">Registry Empty</h3>
                            <p className="text-muted-foreground/70 max-w-sm mx-auto font-medium italic leading-relaxed">
                                You haven&apos;t registered any inverse engines yet. Initialize your first model in the construction pipeline.
                            </p>
                        </CardContent>
                    </Card>
                </motion.div>
            ) : (
                <div className="flex flex-col gap-12">
                    <AnimatePresence>
                        {Object.entries(groupedEngines).map(([datasetName, solverGroups], groupIndex) => (
                            <motion.div
                                key={datasetName}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0, transition: { delay: groupIndex * 0.1 } }}
                                className="space-y-8"
                            >
                                <div className="flex items-center gap-6 group px-2">
                                    <div 
                                        className="flex items-center gap-5 cursor-pointer select-none flex-1 group/dataset"
                                        onClick={() => toggleDatasetExpand(datasetName)}
                                    >
                                        <div className={cn(
                                            "h-7 w-1.5 rounded-full transition-all duration-500",
                                            expandedDatasets[datasetName] === false ? "bg-muted" : "bg-indigo-500"
                                        )} />
                                        <div className="flex flex-col">
                                            <h2 className="text-[11px] font-bold uppercase text-muted-foreground/40 tracking-[0.3em] flex items-center gap-3 group-hover/dataset:text-indigo-500 transition-colors font-heading">
                                                Registry Partition
                                                {expandedDatasets[datasetName] === false ? (
                                                    <ChevronRight className="h-4 w-4 animate-pulse" />
                                                ) : (
                                                    <ChevronDown className="h-4 w-4" />
                                                )}
                                            </h2>
                                            <span className="text-2xl font-bold text-foreground tracking-tight uppercase group-hover/dataset:translate-x-1 transition-transform font-heading">{datasetName}</span>
                                        </div>
                                        <Badge variant="outline" className="bg-indigo-500/5 text-indigo-500 border-indigo-500/20 font-bold text-[10px] uppercase tracking-widest px-4 py-1.5 rounded-full ml-4 font-mono">
                                            {Object.values(solverGroups).flat().length} Assets
                                        </Badge>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div 
                                            className="flex items-center gap-3 px-5 py-2.5 bg-background border border-border/50 rounded-[1rem] hover:border-indigo-500/40 transition-all cursor-pointer select-none group/select"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                const datasetEngines = Object.values(solverGroups).flat();
                                                const allSelected = datasetEngines.every(en => selectedEngines.some(se => se.dataset_name === en.dataset_name && se.solver_type === en.solver_type && se.version === en.version));
                                                
                                                handleSelectGroup(datasetEngines, allSelected);
                                            }}
                                        >
                                            <Checkbox
                                                checked={Object.values(solverGroups).flat().every(en => selectedEngines.some(se => se.dataset_name === en.dataset_name && se.solver_type === en.solver_type && se.version === en.version))}
                                                className="h-4 w-4 border-border/60 data-[state=checked]:bg-indigo-500 data-[state=checked]:border-indigo-500 pointer-events-none rounded-md"
                                            />
                                            <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground/60 group-hover/select:text-indigo-500 transition-colors font-heading">
                                                Select Partition
                                            </span>
                                        </div>
                                        <div className="h-px w-24 bg-border/40" />
                                    </div>
                                </div>

                                {expandedDatasets[datasetName] !== false && (
                                    <motion.div 
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: "auto" }}
                                        exit={{ opacity: 0, height: 0 }}
                                        className="grid grid-cols-1 gap-10"
                                    >
                                    {Object.entries(solverGroups as Record<string, EngineListItem[]>).map(([solverType, versions]) => {
                                        const groupKey = `${datasetName}-${solverType}`;
                                        const isExpanded = expandedGroups[groupKey] !== false;
                                        const enginesInGroup = versions;
                                        const isAllInGroupSelected = enginesInGroup.every(e => selectedEngines.some(se => se.dataset_name === e.dataset_name && se.solver_type === e.solver_type && se.version === e.version));
                                        const isSomeInGroupSelected = enginesInGroup.some(e => selectedEngines.some(se => se.dataset_name === e.dataset_name && se.solver_type === e.solver_type && se.version === e.version)) && !isAllInGroupSelected;

                                        return (
                                            <div key={solverType} className="space-y-6">
                                                <div className="flex items-center gap-5 mb-2 px-6 group/header">
                                                    <div className="flex items-center gap-4 flex-1">
                                                        <Checkbox
                                                            checked={isAllInGroupSelected}
                                                            onCheckedChange={() => handleSelectGroup(enginesInGroup, isAllInGroupSelected)}
                                                            className={isSomeInGroupSelected ? "data-[state=unchecked]:bg-indigo-500/10 data-[state=unchecked]:border-indigo-500/40" : "border-border/60 data-[state=checked]:bg-indigo-500 data-[state=checked]:border-indigo-500 rounded-md"}
                                                        />
                                                        <div
                                                            className="flex items-center gap-3 cursor-pointer select-none group/title"
                                                            onClick={() => toggleGroupExpand(groupKey)}
                                                        >
                                                            <div className="p-2.5 bg-muted/30 rounded-[1rem] border border-border group-hover/title:bg-indigo-500/10 group-hover/title:border-indigo-500/30 transition-all">
                                                                <Layers className="h-5 w-5 text-indigo-500" />
                                                            </div>
                                                            <h3 className="text-lg font-bold text-foreground tracking-tight uppercase group-hover/title:text-indigo-500 transition-colors font-heading">
                                                                {solverType}
                                                            </h3>
                                                            <Badge variant="secondary" className="bg-muted text-muted-foreground/60 border-border/50 text-[9px] h-5 font-bold uppercase tracking-widest px-3 rounded-full font-mono">
                                                                {versions.length} VARIATIONS
                                                            </Badge>
                                                            {isExpanded ? (
                                                                <ChevronUp className="h-5 w-5 text-muted-foreground/40 group-hover/title:text-indigo-500 transition-all transform group-hover/title:-translate-y-0.5" />
                                                            ) : (
                                                                <ChevronDown className="h-5 w-5 text-muted-foreground/40 group-hover/title:text-indigo-500 transition-all transform group-hover/title:translate-y-0.5" />
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>

                                                {isExpanded && (
                                                    <div className="flex flex-col gap-4 pl-6 md:pl-10 border-l-2 border-border/40 ml-10">
                                                        {versions.map((engine) => {
                                                            const isEngineSelected = selectedEngines.some(
                                                                (e) => e.dataset_name === engine.dataset_name &&
                                                                    e.solver_type === engine.solver_type &&
                                                                    e.version === engine.version
                                                            );
                                                            return (
                                                                <motion.div 
                                                                    key={`${engine.solver_type}-${engine.version}`}
                                                                    whileHover={{ y: -3, x: 2 }}
                                                                    className="transition-all duration-300"
                                                                >
                                                                    <Card
                                                                        className={`border-border transition-all duration-500 group overflow-hidden bg-card/60 backdrop-blur-sm rounded-[1rem] cursor-pointer border-l-[6px] ${
                                                                            isEngineSelected 
                                                                                ? 'border-l-indigo-500 ring-2 ring-indigo-500/20 bg-indigo-500/5' 
                                                                                : 'border-l-muted/20 hover:border-l-indigo-400/60'
                                                                        }`}
                                                                        onClick={() => setInspectingEngine(engine)}
                                                                    >
                                                                        <div className="p-6 flex flex-col md:flex-row items-center gap-8">
                                                                            <div className="h-12 w-12 bg-muted rounded-[1rem] flex items-center justify-center group-hover:bg-indigo-500/10 group-hover:border-indigo-500/20 border border-transparent transition-all shrink-0 transform group-hover:rotate-12">
                                                                                <Box className={`h-6 w-6 transition-all duration-500 ${isEngineSelected ? 'text-indigo-500 scale-110' : 'text-muted-foreground/30 group-hover:text-indigo-500'}`} />
                                                                            </div>
                                                                            
                                                                            <div className="flex-1 grid grid-cols-1 md:grid-cols-4 gap-6 items-center w-full">
                                                                                <div className="space-y-1.5">
                                                                                    <h3 className="text-lg font-bold text-foreground tracking-tight group-hover:text-indigo-500 transition-colors uppercase font-heading">
                                                                                        Variation <span className="font-mono text-base ml-1">v{engine.version}</span>
                                                                                    </h3>
                                                                                    <div className="flex items-center gap-3">
                                                                                        <Badge className="bg-indigo-500 text-white font-bold text-[9px] px-3 h-4.5 rounded-full uppercase tracking-tighter font-mono">
                                                                                            REV-{engine.version}
                                                                                        </Badge>
                                                                                        <span className="text-[10px] text-muted-foreground font-bold uppercase tracking-[0.2em] italic opacity-60">
                                                                                            Active Assets
                                                                                        </span>
                                                                                    </div>
                                                                                </div>

                                                                                <div className="hidden md:block">
                                                                                    <p className="text-[9px] uppercase font-bold text-muted-foreground/20 tracking-[0.2em] mb-2 font-heading">Registered At</p>
                                                                                    <p className="text-[10px] font-medium text-muted-foreground/70 flex items-center gap-2 uppercase tracking-tight font-mono">
                                                                                        <Calendar className="h-3.5 w-3.5 text-indigo-500/40" />
                                                                                        {new Date(engine.created_at).toLocaleDateString()}
                                                                                    </p>
                                                                                </div>

                                                                                <div className="hidden md:block">
                                                                                    <p className="text-[9px] uppercase font-bold text-muted-foreground/20 tracking-[0.2em] mb-2 font-heading">Telemetry Status</p>
                                                                                    <div className="flex items-center gap-2.5">
                                                                                        <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                                                                                        <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest font-heading">Stable</span>
                                                                                    </div>
                                                                                </div>

                                                                                <div className="flex justify-end gap-3 items-center">
                                                                                    <div className="flex items-center gap-4">
                                                                                        <Button
                                                                                            variant="ghost"
                                                                                            size="icon"
                                                                                            className="h-10 w-10 text-muted-foreground/40 hover:text-indigo-500 hover:bg-indigo-500/10 transition-all rounded-[1rem] border border-transparent hover:border-indigo-500/20"
                                                                                            onClick={(e) => {
                                                                                                e.stopPropagation();
                                                                                                setInspectingEngine(engine);
                                                                                            }}
                                                                                        >
                                                                                            <Info className="h-5 w-5" />
                                                                                        </Button>
                                                                                        <Checkbox
                                                                                            checked={isEngineSelected}
                                                                                            onCheckedChange={() => toggleSelection(engine)}
                                                                                            className="h-6 w-6 rounded-[1rem] border-border/60 data-[state=checked]:bg-indigo-500 data-[state=checked]:border-indigo-500 transition-all active:scale-90"
                                                                                        />
                                                                                        <Button
                                                                                            variant="ghost"
                                                                                            size="icon"
                                                                                            className="h-10 w-10 text-muted-foreground/20 hover:text-rose-500 hover:bg-rose-500/10 transition-all rounded-[1rem] border border-transparent hover:border-rose-500/20"
                                                                                            onClick={(e) => {
                                                                                                e.stopPropagation();
                                                                                                setEngineToDelete(engine);
                                                                                            }}
                                                                                        >
                                                                                            <Trash2 className="h-5 w-5" />
                                                                                        </Button>
                                                                                    </div>
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    </Card>
                                                                </motion.div>
                                                            );
                                                        })}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                    </motion.div>
                                )}
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {filteredEngines.length === 0 && searchQuery && (
                        <div className="py-32 text-center animate-in fade-in zoom-in-95 duration-700">
                            <div className="bg-muted p-10 rounded-[1rem] inline-block mb-8 border border-border/50">
                                <Search className="h-12 w-12 text-muted-foreground/20" />
                            </div>
                            <p className="text-muted-foreground font-bold uppercase tracking-[0.2em] text-sm font-heading">No registry assets match &ldquo;<span className="text-indigo-500">{searchQuery}</span>&rdquo;</p>
                        </div>
                    )}
                </div>
            )}

            <EngineDetailsInspector 
                engine={inspectingEngine}
                open={!!inspectingEngine}
                onOpenChange={(open) => !open && setInspectingEngine(null)}
            />
        </div>
    );
}
