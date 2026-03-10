"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listAllEngines, deleteEngines } from "@/features/inverse/api";
import { getDatasets } from "@/features/dataset/api";
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
    Square
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

    const { data: engines = [], isLoading } = useQuery({
        queryKey: ["engines", datasetFilter],
        queryFn: () => listAllEngines(datasetFilter === "all" ? undefined : datasetFilter),
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
        <div className="space-y-6 max-w-7xl mx-auto pb-10">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div className="flex flex-col gap-1">
                    <h1 className="text-3xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
                        <Cpu className="h-7 w-7 text-emerald-600" />
                        Inference Hub
                    </h1>
                    <p className="text-slate-500 font-medium">Manage and monitor your trained inverse mapping engines.</p>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-md px-3 h-10 shadow-sm transition-all hover:border-slate-300">
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

                    {selectedEngines.length > 0 && (
                        <Button
                            variant="destructive"
                            onClick={() => setIsBulkDeleting(true)}
                            className="font-bold ripple-effect"
                        >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete ({selectedEngines.length})
                        </Button>
                    )}

                    <div className="flex items-center gap-2">
                        <div className="relative w-full md:w-48">
                            <Select value={datasetFilter} onValueChange={setDatasetFilter}>
                                <SelectTrigger className="bg-white border-slate-200">
                                    <div className="flex items-center gap-2">
                                        <Filter className="h-3.5 w-3.5 text-slate-400" />
                                        <SelectValue placeholder="All Datasets" />
                                    </div>
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All Datasets</SelectItem>
                                    {datasets.map(d => (
                                        <SelectItem key={d.name} value={d.name}>{d.name}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="relative w-full md:w-64">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                            <Input
                                placeholder="Search engines..."
                                className="pl-9 bg-white border-slate-200"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                    </div>
                </div>
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
                <div className="flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/10">
                    <Loader2 className="h-8 w-8 animate-spin text-emerald-600 mb-4" />
                    <p className="text-slate-500 font-medium animate-pulse">Scanning context registry...</p>
                </div>
            ) : engines.length === 0 ? (
                <Card className="border-slate-200 shadow-sm">
                    <CardContent className="p-12 text-center">
                        <div className="bg-slate-50 p-6 rounded-full inline-block mb-4 ring-8 ring-slate-50/50">
                            <Cpu className="h-10 w-10 text-slate-300" />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900">No engines found</h3>
                        <p className="text-slate-500 max-w-sm mx-auto mt-2">
                            Head over to the "Train" section to create your first inverse mapping engine.
                        </p>
                    </CardContent>
                </Card>
            ) : (
                <div className="flex flex-col gap-10">
                    {Object.entries(groupedEngines).map(([datasetName, solverGroups]) => (
                        <div key={datasetName} className="space-y-6">
                            <div className="flex items-center gap-2 group px-1">
                                <div className="h-6 w-1 bg-emerald-500 rounded-full" />
                                <h2 className="text-sm font-black uppercase text-slate-500 tracking-wider flex items-center gap-2">
                                    <Database className="h-3.5 w-3.5" />
                                    {datasetName}
                                </h2>
                                <div className="h-px flex-1 bg-slate-100 ml-2" />
                            </div>

                            <div className="grid grid-cols-1 gap-8">
                                {Object.entries(solverGroups).map(([solverType, versions]) => {
                                    const groupKey = `${datasetName}-${solverType}`;
                                    const isExpanded = expandedGroups[groupKey] !== false; // Default to expanded
                                    const enginesInGroup = versions;
                                    const isAllInGroupSelected = enginesInGroup.every(e => isSelected(e));
                                    const isSomeInGroupSelected = enginesInGroup.some(e => isSelected(e)) && !isAllInGroupSelected;

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
                                                        <Layers className="h-4 w-4 text-indigo-500" />
                                                        <h3 className="text-sm font-bold text-slate-800">
                                                            {solverType}
                                                        </h3>
                                                        <Badge variant="secondary" className="bg-slate-100 text-slate-500 border-none text-[9px] h-4">
                                                            {versions.length} versions
                                                        </Badge>
                                                        {isExpanded ? (
                                                            <ChevronUp className="h-3.5 w-3.5 text-slate-400 group-hover/header:text-slate-600 transition-colors" />
                                                        ) : (
                                                            <ChevronDown className="h-3.5 w-3.5 text-slate-400 group-hover/header:text-slate-600 transition-colors" />
                                                        )}
                                                    </div>
                                                </div>
                                            </div>

                                            {isExpanded && (
                                                <div className="flex flex-col gap-2 pl-4 md:pl-8 border-l border-slate-100 ml-6 animate-in fade-in slide-in-from-top-2 duration-300">
                                                    {versions.map((engine) => (
                                                        <Card
                                                            key={`${engine.dataset_name}-${engine.solver_type}-${engine.version}`}
                                                            className={`border-slate-200 shadow-sm hover:shadow-md transition-all duration-200 group overflow-hidden bg-white ${isSelected(engine) ? 'ring-1 ring-emerald-500 bg-emerald-50/5' : ''}`}
                                                        >
                                                            <div className="flex flex-row items-center p-3 gap-4">
                                                                <div className="pl-1">
                                                                    <Checkbox
                                                                        checked={isSelected(engine)}
                                                                        onCheckedChange={() => toggleSelection(engine)}
                                                                    />
                                                                </div>

                                                                <div className="flex items-center gap-3 flex-1 min-w-0">
                                                                    <div className="h-8 w-8 bg-slate-50 rounded flex items-center justify-center shrink-0 group-hover:bg-emerald-100 transition-colors">
                                                                        <span className="text-[10px] font-black text-slate-400 group-hover:text-emerald-600 uppercase">
                                                                            V{engine.version}
                                                                        </span>
                                                                    </div>
                                                                    <div className="min-w-0 flex-1">
                                                                        <div className="flex items-center gap-2">
                                                                            <h3 className="text-xs font-bold text-slate-700 truncate">
                                                                                Version {engine.version}
                                                                            </h3>
                                                                            <Badge variant="secondary" className="bg-slate-100 text-slate-400 border-none px-1 h-3 text-[8px] font-semibold">
                                                                                {new Date(engine.created_at).toLocaleDateString()}
                                                                            </Badge>
                                                                        </div>
                                                                    </div>
                                                                </div>

                                                                <div className="hidden md:flex items-center gap-6 px-6 border-x border-slate-50">
                                                                    <div className="flex flex-col gap-0.5 min-w-[100px]">
                                                                        <p className="text-[8px] uppercase font-bold text-slate-400 tracking-wider">Trained On</p>
                                                                        <div className="flex items-center gap-1.5">
                                                                            <Calendar className="h-2.5 w-2.5 text-indigo-400" />
                                                                            <span className="text-[10px] font-semibold text-slate-600">
                                                                                {new Date(engine.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                                            </span>
                                                                        </div>
                                                                    </div>
                                                                    <div className="flex flex-col gap-0.5 min-w-[70px]">
                                                                        <p className="text-[8px] uppercase font-bold text-slate-400 tracking-wider">Status</p>
                                                                        <div className="flex items-center gap-1.5">
                                                                            <div className="h-1 w-1 rounded-full bg-emerald-500 animate-pulse" />
                                                                            <span className="text-[10px] font-bold text-slate-600 tracking-tight">Active</span>
                                                                        </div>
                                                                    </div>
                                                                </div>

                                                                <div className="flex items-center justify-end gap-1 pr-1">
                                                                    <Button
                                                                        variant="ghost"
                                                                        size="sm"
                                                                        className="h-8 w-8 p-0 text-slate-400 hover:text-red-600 ripple-effect"
                                                                        onClick={() => setEngineToDelete(engine)}
                                                                    >
                                                                        <Trash2 className="h-3.5 w-3.5" />
                                                                    </Button>
                                                                    <div className="h-8 w-8 text-slate-200 flex items-center justify-center">
                                                                        <ChevronRight className="h-4 w-4" />
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </Card>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}

                    {filteredEngines.length === 0 && searchQuery && (
                        <div className="py-20 text-center">
                            <Cpu className="h-10 w-10 text-slate-200 mx-auto mb-4" />
                            <p className="text-slate-500 italic">No engines match "{searchQuery}"</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
