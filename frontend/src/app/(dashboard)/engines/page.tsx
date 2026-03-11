"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { listAllEngines, deleteEngines } from "@/features/inverse/api";
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
    Filter,
    Database,
    Calendar,
    Layers,
    Loader2,
    ChevronRight,
    CheckSquare,
    Square,
    Zap,
    Box,
    RefreshCcw
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

export default function EnginesPage() {
    const [searchQuery, setSearchQuery] = useState("");
    const [datasetFilter, setDatasetFilter] = useState<string>("all");
    const [selectedEngines, setSelectedEngines] = useState<any[]>([]);
    const [engineToDelete, setEngineToDelete] = useState<any | null>(null);
    const [isBulkDeleting, setIsBulkDeleting] = useState(false);
    const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});

    const queryClient = useQueryClient();
    const { showToast } = useToast();

    const { data: engines = [], isLoading, refetch, isFetching } = useQuery({
        queryKey: ["engines", datasetFilter],
        queryFn: () => listAllEngines(datasetFilter === "all" ? undefined : datasetFilter),
        refetchOnWindowFocus: true,
        staleTime: 0, // Ensure it's always considered stale so it refetches on mount
    });

    const { data: datasets = [] } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
    });

    const deleteMutation = useMutation({
        mutationFn: (e: any) => deleteEngines([{
            dataset_name: e.dataset_name,
            solver_type: e.solver_type,
            version: e.version
        }]),
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ["engines"] });
            setEngineToDelete(null);

            // Check for API-level silent failures
            const errors = Array.isArray(data) ? data.filter((r: any) => r.status === "error" || r.status === "not_found") : [];
            if (errors.length > 0) {
                showToast(`Some engines could not be deleted: ${errors[0].status}`, "error");
            } else {
                showToast("Engine deleted successfully", "success");
            }
        },
        onError: (err: any) => {
            showToast(`Deletion failed: ${err.message}`, "error");
        }
    });

    const bulkDeleteMutation = useMutation({
        mutationFn: deleteEngines,
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ["engines"] });
            setIsBulkDeleting(false);

            // Check for API-level silent failures
            const errors = Array.isArray(data) ? data.filter((r: any) => r.status === "error" || r.status === "not_found") : [];
            if (errors.length > 0) {
                showToast(`Failed to delete ${errors.length} engines. The others were deleted.`, "error");
                // Remove successful deletes from selection
                const successful = Array.isArray(data) ? data.filter((r: any) => r.status === "deleted") : [];
                setSelectedEngines(prev => prev.filter(p => !successful.some((s: any) => s.dataset_name === p.dataset_name && s.solver_type === p.solver_type && s.version === p.version)));
            } else {
                setSelectedEngines([]);
                showToast("Engines deleted successfully", "success");
            }
        },
        onError: (err: any) => {
            showToast(`Bulk deletion failed: ${err.message}`, "error");
        }
    });

    const toggleSelection = (engine: any) => {
        const id = `${engine.dataset_name}-${engine.solver_type}-${engine.version}`;
        setSelectedEngines(prev => {
            const exists = prev.some(e => `${e.dataset_name}-${e.solver_type}-${e.version}` === id);
            return exists
                ? prev.filter(e => `${e.dataset_name}-${e.solver_type}-${e.version}` !== id)
                : [...prev, {
                    dataset_name: engine.dataset_name,
                    solver_type: engine.solver_type,
                    version: engine.version
                }];
        });
    };

    const isSelected = (engine: any) => {
        const id = `${engine.dataset_name}-${engine.solver_type}-${engine.version}`;
        return selectedEngines.some(e => `${e.dataset_name}-${e.solver_type}-${e.version}` === id);
    };

    const filteredEngines = engines.filter(e =>
        e.solver_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (e.dataset_name && e.dataset_name.toLowerCase().includes(searchQuery.toLowerCase()))
    );

    // Grouping logic: Dataset -> Solver Type -> Versions
    const groupedEngines = useMemo(() => {
        const groups: Record<string, Record<string, any[]>> = {};

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
                version: e.version
            })));
        }
    };

    const handleSelectGroup = (enginesInGroup: any[], isCurrentlyAllSelected: boolean) => {
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
                        version: e.version
                    }));
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
                            <Layers className="h-6 w-6 text-white" />
                        </div>
                        <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-slate-900 via-indigo-900 to-indigo-800 font-sans">
                            Engine Registry
                        </h1>
                    </div>
                    <p className="text-slate-500 font-medium ml-12">Manage and monitor high-fidelity inverse estimators.</p>
                    <div className="absolute -top-10 -right-20 opacity-5 pointer-events-none">
                        <Cpu className="h-48 w-48 text-indigo-900 rotate-6" />
                    </div>
                </motion.div>

                <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex flex-col md:flex-row items-center gap-3 relative z-10"
                >
                    <div className="flex items-center gap-3 w-full md:w-auto">
                        <div className="relative w-full md:w-64">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                            <Input
                                placeholder="Filter registry..."
                                className="pl-9 bg-white border-slate-200 shadow-sm focus:ring-2 focus:ring-indigo-100 transition-all font-medium"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                        <Select value={datasetFilter} onValueChange={setDatasetFilter}>
                            <SelectTrigger className="w-full md:w-[180px] bg-white border-slate-200 font-bold text-slate-700">
                                <SelectValue placeholder="All Datasets" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="all">All Datasets</SelectItem>
                                {datasets.map((d: any) => (
                                    <SelectItem key={d.name} value={d.name}>{d.name}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    <div className="flex items-center gap-3 w-full md:w-auto">
                        {selectedEngines.length > 0 && (
                            <Button
                                variant="destructive"
                                onClick={() => setIsBulkDeleting(true)}
                                className="font-bold shadow-lg shadow-rose-100 transition-all hover:scale-105 active:scale-95 px-6"
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
                            className="bg-white border-slate-200 text-slate-500 hover:text-indigo-600 hover:border-indigo-200 h-10 w-10 rounded-xl shadow-sm transition-all active:scale-95 group"
                            title="Refresh Engines"
                        >
                            <RefreshCcw className={cn("h-4 w-4 transition-transform duration-500", isFetching ? "animate-spin" : "group-hover:rotate-180")} />
                        </Button>
                    </div>
                </motion.div>
            </div>
            <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-md px-3 h-10 shadow-sm transition-all hover:border-slate-300 w-fit">
                <Checkbox
                    id="select-all"
                    checked={filteredEngines.length > 0 && selectedEngines.length === filteredEngines.length}
                    onCheckedChange={handleSelectAll}
                />
                <label
                    htmlFor="select-all"
                    className="text-xs font-bold text-slate-500 uppercase cursor-pointer select-none"
                >
                    Select All
                </label>
            </div>

            {/* Individual Delete Confirmation */}
            <Dialog open={!!engineToDelete} onOpenChange={(open) => !open && setEngineToDelete(null)}>
                <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-red-600">
                            <AlertCircle className="h-5 w-5" />
                            Confirm Engine Deletion
                        </DialogTitle>
                        <DialogDescription className="py-2">
                            Are you sure you want to delete <span className="font-bold text-slate-900">{engineToDelete?.solver_type} v{engineToDelete?.version}</span> for dataset <span className="font-bold text-slate-900">"{engineToDelete?.dataset_name}"</span>?
                            This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex justify-end gap-3 pt-4">
                        <Button variant="outline" onClick={() => setEngineToDelete(null)}>
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => engineToDelete && deleteMutation.mutate(engineToDelete)}
                            disabled={deleteMutation.isPending}
                        >
                            {deleteMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : "Delete Engine"}
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
                            You are about to delete <span className="font-bold text-slate-900">{selectedEngines.length}</span> trained engines.
                            This action cannot be undone.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="flex justify-end gap-3 pt-4">
                        <Button variant="outline" onClick={() => setIsBulkDeleting(false)}>
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={() => bulkDeleteMutation.mutate(selectedEngines)}
                            disabled={bulkDeleteMutation.isPending}
                        >
                            {bulkDeleteMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : `Delete ${selectedEngines.length} Engines`}
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>

            {isLoading ? (
                <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-slate-200 rounded-3xl bg-slate-50/10">
                    <Loader2 className="h-8 w-8 animate-spin text-indigo-500 mb-4" />
                    <p className="text-slate-500 font-bold uppercase tracking-widest text-[10px] animate-pulse">Scanning Engine Registry...</p>
                </div>
            ) : Object.keys(groupedEngines).length === 0 ? (
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                >
                    <Card className="border-slate-200 shadow-sm rounded-3xl overflow-hidden">
                        <CardContent className="p-16 text-center bg-linear-to-b from-white to-slate-50/50">
                            <div className="bg-white p-8 rounded-full inline-block mb-6 shadow-xl shadow-slate-200/50 ring-8 ring-slate-50">
                                <Cpu className="h-12 w-12 text-slate-300" />
                            </div>
                            <h3 className="text-2xl font-black text-slate-900 tracking-tight">No engines available</h3>
                            <p className="text-slate-500 max-w-sm mx-auto mt-3 font-medium leading-relaxed">
                                You haven't trained any inverse engines yet. Head over to the Construction page to build your first model.
                            </p>
                        </CardContent>
                    </Card>
                </motion.div>
            ) : (
                <div className="flex flex-col gap-10">
                    <AnimatePresence>
                        {Object.entries(groupedEngines).map(([datasetName, solverGroups], groupIndex) => (
                            <motion.div
                                key={datasetName}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0, transition: { delay: groupIndex * 0.1 } }}
                                className="space-y-6"
                            >
                                <div className="flex items-center gap-4 group px-1">
                                    <div className="h-6 w-1 bg-indigo-500 rounded-full" />
                                    <h2 className="text-sm font-black uppercase text-slate-400 tracking-widest flex items-center gap-2">
                                        {datasetName}
                                    </h2>
                                    <Badge variant="outline" className="bg-indigo-50 text-indigo-600 border-indigo-100 font-black text-[10px]">
                                        {Object.values(solverGroups).flat().length} Registered Assets
                                    </Badge>
                                    <div className="h-px flex-1 bg-slate-100 ml-2" />
                                </div>

                                <div className="grid grid-cols-1 gap-8">
                                    {Object.entries(solverGroups as Record<string, any[]>).map(([solverType, versions]) => {
                                        const groupKey = `${datasetName}-${solverType}`;
                                        const isExpanded = expandedGroups[groupKey] !== false;
                                        const enginesInGroup = versions;
                                        const isAllInGroupSelected = enginesInGroup.every(e => selectedEngines.some(se => se.dataset_name === e.dataset_name && se.solver_type === e.solver_type && se.version === e.version));
                                        const isSomeInGroupSelected = enginesInGroup.some(e => selectedEngines.some(se => se.dataset_name === e.dataset_name && se.solver_type === e.solver_type && se.version === e.version)) && !isAllInGroupSelected;

                                        return (
                                            <div key={solverType} className="space-y-4">
                                                <div className="flex items-center gap-4 mb-1 px-4 group/header">
                                                    <div className="flex items-center gap-2 flex-1">
                                                        <Checkbox
                                                            checked={isAllInGroupSelected}
                                                            onCheckedChange={() => handleSelectGroup(enginesInGroup, isAllInGroupSelected)}
                                                            className={isSomeInGroupSelected ? "data-[state=unchecked]:bg-slate-100" : ""}
                                                        />
                                                        <div
                                                            className="flex items-center gap-2 cursor-pointer select-none"
                                                            onClick={() => toggleGroupExpand(groupKey)}
                                                        >
                                                            <div className="p-1.5 bg-slate-50 rounded-lg group-hover/header:bg-indigo-50 transition-colors">
                                                                <Layers className="h-4 w-4 text-indigo-500" />
                                                            </div>
                                                            <h3 className="text-base font-black text-slate-800 tracking-tight">
                                                                {solverType}
                                                            </h3>
                                                            <Badge variant="secondary" className="bg-slate-100 text-slate-500 border-none text-[9px] h-4 font-black">
                                                                {versions.length} VERSIONS
                                                            </Badge>
                                                            {isExpanded ? (
                                                                <ChevronUp className="h-4 w-4 text-slate-400 group-hover/header:text-slate-600 transition-colors" />
                                                            ) : (
                                                                <ChevronDown className="h-4 w-4 text-slate-400 group-hover/header:text-slate-600 transition-colors" />
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>

                                                {isExpanded && (
                                                    <div className="flex flex-col gap-3 pl-4 md:pl-8 border-l-2 border-slate-100 ml-6">
                                                        {versions.map((engine) => {
                                                            const isEngineSelected = selectedEngines.some(
                                                                (e) => e.dataset_name === engine.dataset_name &&
                                                                    e.solver_type === engine.solver_type &&
                                                                    e.version === engine.version
                                                            );
                                                            return (
                                                                <motion.div 
                                                                    key={`${engine.solver_type}-${engine.version}`}
                                                                    whileHover={{ y: -2 }}
                                                                >
                                                                    <Card
                                                                        className={`border-slate-200/60 shadow-lg shadow-slate-200/20 transition-all duration-300 group overflow-hidden bg-white rounded-2xl cursor-pointer border-l-4 ${
                                                                            isEngineSelected 
                                                                                ? 'border-l-indigo-500 ring-2 ring-indigo-500/10 bg-indigo-50/5' 
                                                                                : 'border-l-slate-100/50 hover:border-l-indigo-400'
                                                                        }`}
                                                                        onClick={() => toggleSelection(engine)}
                                                                    >
                                                                        <div className="p-4 flex flex-col md:flex-row items-center gap-6">
                                                                            <div className="h-10 w-10 bg-slate-50 rounded-xl flex items-center justify-center group-hover:bg-indigo-50 transition-colors shadow-inner shrink-0">
                                                                                <Box className={`h-5 w-5 transition-all duration-300 ${isEngineSelected ? 'text-indigo-600' : 'text-slate-400 group-hover:text-indigo-500'}`} />
                                                                            </div>
                                                                            
                                                                            <div className="flex-1 grid grid-cols-1 md:grid-cols-4 gap-4 items-center w-full">
                                                                                <div className="space-y-1">
                                                                                    <h3 className="text-base font-black text-slate-800 tracking-tight group-hover:text-indigo-900 transition-colors">
                                                                                        Version {engine.version}
                                                                                    </h3>
                                                                                    <div className="flex items-center gap-2">
                                                                                        <Badge className="bg-slate-900 text-white font-black text-[9px] px-2 h-4 rounded-full">
                                                                                            V{engine.version}
                                                                                        </Badge>
                                                                                        <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">
                                                                                            Online
                                                                                        </span>
                                                                                    </div>
                                                                                </div>

                                                                                <div className="hidden md:block">
                                                                                    <p className="text-[9px] uppercase font-black text-slate-300 tracking-tighter mb-1">Registered</p>
                                                                                    <p className="text-[10px] font-bold text-slate-500 flex items-center gap-1.5">
                                                                                        <Calendar className="h-3 w-3 text-slate-300" />
                                                                                        {new Date(engine.created_at).toLocaleDateString()}
                                                                                    </p>
                                                                                </div>

                                                                                <div className="hidden md:block">
                                                                                    <p className="text-[9px] uppercase font-black text-slate-300 tracking-tighter mb-1">Status</p>
                                                                                    <div className="flex items-center gap-1.5">
                                                                                        <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse" />
                                                                                        <span className="text-[10px] font-black text-emerald-600">Online</span>
                                                                                    </div>
                                                                                </div>

                                                                                <div className="flex justify-end gap-3 items-center">
                                                                                    <div className="flex gap-2">
                                                                                        <Checkbox
                                                                                            checked={isEngineSelected}
                                                                                            onCheckedChange={() => toggleSelection(engine)}
                                                                                            className="h-5 w-5 rounded-md border-slate-300"
                                                                                        />
                                                                                        <Button
                                                                                            variant="ghost"
                                                                                            size="icon"
                                                                                            className="h-8 w-8 text-slate-300 hover:text-rose-600 hover:bg-rose-50 transition-all rounded-lg"
                                                                                            onClick={(e) => {
                                                                                                e.stopPropagation();
                                                                                                setEngineToDelete(engine);
                                                                                            }}
                                                                                        >
                                                                                            <Trash2 className="h-4 w-4" />
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
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {filteredEngines.length === 0 && searchQuery && (
                        <div className="py-24 text-center">
                            <div className="bg-slate-50 p-6 rounded-full inline-block mb-4">
                                <Search className="h-8 w-8 text-slate-300" />
                            </div>
                            <p className="text-slate-500 font-bold tracking-tight">No registry assets match "<span className="text-indigo-600 uppercase tracking-widest">{searchQuery}</span>"</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
