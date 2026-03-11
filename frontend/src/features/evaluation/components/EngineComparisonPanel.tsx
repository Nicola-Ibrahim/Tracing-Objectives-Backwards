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
        <Card className="border-slate-200">
            <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                <CardTitle className="text-lg font-semibold flex items-center gap-2">
                    <Activity className="h-5 w-5 text-indigo-600" />
                    Diagnostic Controller
                </CardTitle>
                <CardDescription>Select dataset and engines for comparative analysis.</CardDescription>
            </CardHeader>
            <CardContent className="p-6 space-y-6">
                <div className="space-y-2">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-tight">Dataset Scope</label>
                    <Select onValueChange={setSelectedDataset} value={selectedDataset}>
                        <SelectTrigger>
                            <SelectValue placeholder="Select dataset" />
                        </SelectTrigger>
                        <SelectContent>
                            {datasets.map(d => (
                                <SelectItem key={d} value={d}>{d}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                {selectedDataset && (
                    <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-tight">Available Engines</label>
                        {fetchingEngines ? (
                            <div className="flex items-center justify-center p-4 border rounded-lg border-dashed">
                                <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
                            </div>
                        ) : availableEngines.length === 0 ? (
                            <div className="p-4 border rounded-lg border-dashed text-center text-sm text-slate-500">
                                No trained engines found for this dataset.
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 gap-2">
                                {availableEngines.map((e) => {
                                    const isSelected = selectedEngines.some(se => se.solver_type === e.solver_type && se.version === e.version);
                                    return (
                                        <div
                                            key={`${e.solver_type}_v${e.version}`}
                                            onClick={() => toggleEngine(e)}
                                            className={`flex items-center justify-between p-3 border rounded-lg cursor-pointer transition-all ${isSelected ? "bg-indigo-50 border-indigo-200" : "hover:border-slate-300 bg-white"
                                                }`}
                                        >
                                            <div className="flex items-center gap-3">
                                                <Checkbox checked={isSelected} onCheckedChange={() => toggleEngine(e)} />
                                                <div>
                                                    <span className="text-sm font-semibold">{e.solver_type}</span>
                                                    <span className="text-[10px] text-slate-500 ml-2 italic">v{e.version}</span>
                                                </div>
                                            </div>
                                            <Badge variant="outline" className="text-[10px]">Trained</Badge>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}

                {selectedEngines.length > 0 && (
                    <div className="pt-4 border-t border-slate-100 flex flex-col gap-4">
                        <div className="flex flex-wrap gap-2">
                            {selectedEngines.map(e => (
                                <Badge key={`${e.solver_type}_v${e.version}`} variant="secondary" className="flex items-center gap-1 bg-indigo-100 text-indigo-700">
                                    {e.solver_type} v{e.version}
                                    <X className="h-3 w-3 cursor-pointer" onClick={() => toggleEngine(e)} />
                                </Badge>
                            ))}
                        </div>

                        <Button
                            disabled={isLoading}
                            onClick={handleDiagnose}
                            className={`w-full py-6 transition-all duration-300 relative overflow-hidden ${
                                isLoading 
                                ? "bg-slate-700 cursor-not-allowed" 
                                : "bg-slate-900 hover:bg-slate-800 shadow-md hover:shadow-lg active:scale-[0.98]"
                            }`}
                        >
                            {isLoading && (
                                <div className="absolute inset-0 bg-white/5 animate-pulse" />
                            )}
                            <div className="flex items-center justify-center gap-3 relative z-10">
                                {isLoading ? (
                                    <>
                                        <Loader2 className="h-5 w-5 animate-spin text-indigo-400" />
                                        <span className="font-medium tracking-wide">Analysing Engines...</span>
                                    </>
                                ) : (
                                    <>
                                        <Activity className="h-5 w-5 text-indigo-400" />
                                        <span className="font-semibold">Run Comparative Diagnosis</span>
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
