"use client";

import React, { useState, useEffect } from "react";
import { EngineListItem } from "../../inverse/types";
import { listEnginesForDataset } from "../../inverse/api";
import { DiagnoseRequest, EngineCandidate } from "../types";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Activity, Loader2, Plus, Info, X } from "lucide-react";

interface EngineComparisonPanelProps {
    datasets: string[];
    onDiagnose: (params: DiagnoseRequest) => Promise<void>;
    isLoading?: boolean;
}

export function EngineComparisonPanel({ datasets, onDiagnose, isLoading }: EngineComparisonPanelProps) {
    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [availableEngines, setAvailableEngines] = useState<EngineListItem[]>([]);
    const [selectedEngines, setSelectedEngines] = useState<EngineListItem[]>([]);
    const [fetchingEngines, setFetchingEngines] = useState(false);

    useEffect(() => {
        if (selectedDataset) {
            setFetchingEngines(true);
            listEnginesForDataset(selectedDataset)
                .then(setAvailableEngines)
                .finally(() => setFetchingEngines(false));
        }
    }, [selectedDataset]);

    const toggleEngine = (engine: EngineListItem) => {
        const exists = selectedEngines.find(e => e.solver_type === engine.solver_type && e.version === engine.version);
        if (exists) {
            setSelectedEngines(selectedEngines.filter(e => !(e.solver_type === engine.solver_type && e.version === engine.version)));
        } else {
            setSelectedEngines([...selectedEngines, engine]);
        }
    };

    const handleDiagnose = () => {
        if (!selectedDataset || selectedEngines.length === 0) return;

        const request: DiagnoseRequest = {
            dataset_name: selectedDataset,
            candidates: selectedEngines.map(e => ({ solver_type: e.solver_type, version: e.version })),
            scale_method: "sd",
            num_samples: 200
        };

        onDiagnose(request);
    };

    return (
        <Card className="border-border shadow-2xl bg-card/50 backdrop-blur-sm rounded-[2rem] overflow-hidden">
            <CardHeader className="bg-muted/10 border-b border-border p-8">
                <CardTitle className="text-xl font-black flex items-center gap-3 text-foreground uppercase tracking-tight">
                    <Activity className="h-6 w-6 text-indigo-500" />
                    Diagnostic Controller
                </CardTitle>
                <CardDescription className="text-muted-foreground font-medium italic">Select dataset and engines for comparative analysis.</CardDescription>
            </CardHeader>
            <CardContent className="p-8 space-y-8">
                <div className="space-y-3">
                    <label className="text-[10px] font-black text-muted-foreground/60 uppercase tracking-[0.2em] ml-1">Dataset Scope</label>
                    <Select onValueChange={setSelectedDataset} value={selectedDataset}>
                        <SelectTrigger className="bg-background border-border text-foreground h-12 rounded-2xl focus:ring-2 focus:ring-indigo-500/10 transition-all">
                            <SelectValue placeholder="Select target dataset" />
                        </SelectTrigger>
                        <SelectContent className="bg-popover border-border text-popover-foreground rounded-2xl shadow-2xl">
                            {datasets.map(d => (
                                <SelectItem key={d} value={d} className="hover:bg-muted focus:bg-muted cursor-pointer font-bold tracking-tight">{d}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                {selectedDataset && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-top-2 duration-500">
                        <label className="text-[10px] font-black text-muted-foreground/60 uppercase tracking-[0.2em] ml-1">Available Engines</label>
                        {fetchingEngines ? (
                            <div className="flex items-center justify-center p-12 border-2 border-dashed rounded-3xl border-border/40 bg-muted/5">
                                <Loader2 className="h-6 w-6 animate-spin text-indigo-500" />
                            </div>
                        ) : availableEngines.length === 0 ? (
                            <div className="p-12 border-2 border-dashed rounded-3xl border-border/40 text-center bg-muted/5">
                                <p className="text-xs font-black text-muted-foreground/40 uppercase tracking-widest italic">No trained engines found.</p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 gap-3">
                                {availableEngines.map((e) => {
                                    const isSelected = selectedEngines.some(se => se.solver_type === e.solver_type && se.version === e.version);
                                    return (
                                        <div
                                            key={`${e.solver_type}_v${e.version}`}
                                            onClick={() => toggleEngine(e)}
                                            className={`flex items-center justify-between p-4 border rounded-2xl cursor-pointer transition-all duration-300 relative overflow-hidden group ${
                                                isSelected 
                                                    ? "bg-indigo-500/10 border-indigo-500/50 text-foreground ring-1 ring-indigo-500/20 shadow-lg shadow-indigo-500/5" 
                                                    : "hover:border-indigo-500/30 bg-background border-border text-foreground hover:shadow-md"
                                            }`}
                                        >
                                            {isSelected && <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500" />}
                                            <div className="flex items-center gap-4">
                                                <Checkbox 
                                                    checked={isSelected} 
                                                    onCheckedChange={() => toggleEngine(e)} 
                                                    className="border-border data-[state=checked]:bg-indigo-500 data-[state=checked]:border-indigo-500 rounded-md transition-all" 
                                                />
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-black uppercase tracking-tight">{e.solver_type}</span>
                                                    <span className="text-[9px] font-black text-muted-foreground/60 uppercase tracking-widest italic">v{e.version}</span>
                                                </div>
                                            </div>
                                            <Badge variant="secondary" className="text-[9px] font-black bg-muted/50 text-muted-foreground/60 border-border/50 uppercase tracking-widest px-3 rounded-full">Trained</Badge>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}

                {selectedEngines.length > 0 && (
                    <div className="pt-6 border-t border-border flex flex-col gap-6 animate-in fade-in duration-500">
                        <div className="flex flex-wrap gap-2 pt-2">
                            {selectedEngines.map(e => (
                                <Badge key={`${e.solver_type}_v${e.version}`} variant="indigo" className="flex items-center gap-2 group px-4 py-1.5 rounded-full font-black text-[10px] uppercase tracking-widest shadow-lg shadow-indigo-500/10">
                                    {e.solver_type} v{e.version}
                                    <X className="h-3 w-3 cursor-pointer opacity-50 group-hover:opacity-100 hover:scale-125 transition-all text-white" onClick={(e_event) => {
                                        e_event.stopPropagation();
                                        toggleEngine(e);
                                    }} />
                                </Badge>
                            ))}
                        </div>

                        <Button
                            disabled={isLoading}
                            onClick={handleDiagnose}
                            className={`w-full py-8 transition-all duration-500 relative overflow-hidden group/btn rounded-2xl border border-transparent shadow-2xl ${
                                isLoading
                                    ? "bg-muted cursor-not-allowed text-muted-foreground opacity-50"
                                    : "bg-foreground text-background hover:bg-indigo-500 hover:text-white active:scale-[0.98]"
                            }`}
                        >
                            {isLoading && (
                                <div className="absolute inset-0 bg-background/5 dark:bg-foreground/5 animate-pulse" />
                            )}
                            <div className="flex items-center justify-center gap-4 relative z-10">
                                {isLoading ? (
                                    <>
                                        <Loader2 className="h-6 w-6 animate-spin text-indigo-400" />
                                        <span className="font-black uppercase tracking-[0.2em] text-xs">Analysing Engines...</span>
                                    </>
                                ) : (
                                    <>
                                        <Activity className="h-6 w-6 text-indigo-500 group-hover/btn:text-white transition-all transform group-hover/btn:scale-110 group-hover/btn:rotate-12" />
                                        <span className="font-black uppercase tracking-[0.2em] text-xs">Run Comparative Audit</span>
                                    </>
                                )}
                            </div>
                        </Button>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
