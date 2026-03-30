"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { getEngineDetails } from "../api";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { 
    Cpu, 
    Calendar, 
    Layers, 
    Activity, 
    Clock, 
    Database, 
    ArrowRight,
    Loader2,
    CheckCircle2
} from "lucide-react";
import { TrainingHistoryChart } from "./TrainingHistoryChart";
import { motion } from "framer-motion";

interface EngineDetailsInspectorProps {
    engine: {
        dataset_name: string;
        solver_type: string;
        version: number;
    } | null;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function EngineDetailsInspector({
    engine,
    open,
    onOpenChange,
}: EngineDetailsInspectorProps) {
    const { data: details, isLoading } = useQuery({
        queryKey: ["engine-details", engine?.dataset_name, engine?.solver_type, engine?.version],
        queryFn: () => engine ? getEngineDetails(engine.dataset_name, engine.solver_type, engine.version) : null,
        enabled: !!engine && open,
    });

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[900px] p-0 rounded-[2rem] overflow-hidden bg-card border-border max-h-[90vh] overflow-y-auto scrollbar-thin scrollbar-thumb-indigo-500/10 scrollbar-track-transparent">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center h-[500px]">
                        <Loader2 className="h-10 w-10 animate-spin text-indigo-500 mb-4" />
                        <p className="text-muted-foreground font-bold uppercase tracking-[0.2em] text-[10px] font-heading">Retrieving Engine DNA...</p>
                    </div>
                ) : details ? (
                    <div className="flex flex-col">
                        <DialogHeader className="p-10 bg-muted/30 border-b border-border space-y-4 font-sans">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className="p-3 bg-indigo-600 rounded-[1.2rem] shadow-lg shadow-indigo-500/20">
                                        <Cpu className="h-8 w-8 text-white" />
                                    </div>
                                    <div className="space-y-1">
                                        <DialogTitle className="text-3xl font-bold font-heading text-foreground tracking-tight uppercase">
                                            {details.solver_type} <span className="text-indigo-500 text-xl ml-2 font-mono">v{details.version}</span>
                                        </DialogTitle>
                                        <div className="flex items-center gap-3">
                                            <Badge variant="outline" className="bg-indigo-500/5 text-indigo-500 border-indigo-500/20 font-bold text-[9px] uppercase tracking-widest px-3 py-1 rounded-full">
                                                Registry Asset
                                            </Badge>
                                            <span className="text-[10px] text-muted-foreground font-bold flex items-center gap-1.5 uppercase tracking-wider italic font-heading">
                                                <Database className="h-3 w-3" />
                                                {details.dataset_name}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right hidden md:block">
                                    <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-[0.3em] mb-1 font-heading">Authenticated At</p>
                                    <div className="flex items-center gap-2 justify-end text-sm font-medium text-foreground italic uppercase font-mono">
                                        <Calendar className="h-4 w-4 text-indigo-500/40" />
                                        {new Date(details.created_at).toLocaleString()}
                                    </div>
                                </div>
                            </div>
                        </DialogHeader>

                        <div className="p-10 space-y-10">
                            {/* High Density Metadata */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                                <div className="p-6 bg-muted/20 border border-border rounded-[1.5rem] transition-all hover:bg-muted/30">
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <Activity className="h-3 w-3 text-emerald-500" />
                                        Train Size
                                    </p>
                                    <p className="text-3xl font-bold text-foreground tracking-tighter font-mono">
                                        {details.n_train_samples}<span className="text-sm font-medium text-muted-foreground/40 ml-1.5 lowercase font-sans">obs</span>
                                    </p>
                                </div>
                                <div className="p-6 bg-muted/20 border border-border rounded-[1.5rem] transition-all hover:bg-muted/30">
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <Layers className="h-3 w-3 text-amber-500" />
                                        Test Size
                                    </p>
                                    <p className="text-3xl font-bold text-foreground tracking-tighter font-mono">
                                        {details.n_test_samples}<span className="text-sm font-medium text-muted-foreground/40 ml-1.5 lowercase font-sans">obs</span>
                                    </p>
                                </div>
                                <div className="p-6 bg-muted/20 border border-border rounded-[1.5rem] transition-all hover:bg-muted/30">
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <Clock className="h-3 w-3 text-indigo-500" />
                                        Split Ratio
                                    </p>
                                    <p className="text-3xl font-bold text-foreground tracking-tighter font-mono">
                                        {details.split_ratio.toFixed(2)}
                                    </p>
                                </div>
                                <div className="p-6 bg-muted/20 border border-border rounded-[1.5rem] transition-all hover:bg-muted/30">
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                                        Status
                                    </p>
                                    <p className="text-2xl font-bold text-emerald-500 tracking-tighter uppercase italic font-heading">
                                        Stable
                                    </p>
                                </div>
                            </div>

                            <div className="grid md:grid-cols-12 gap-10">
                                {/* Transformation Pipeline */}
                                <div className="md:col-span-5 space-y-6">
                                    <div className="flex items-center gap-3">
                                        <div className="h-px bg-border flex-1" />
                                        <h4 className="text-[11px] font-bold text-muted-foreground uppercase tracking-[0.3em] font-heading">Processing Pipeline</h4>
                                        <div className="h-px bg-border flex-1" />
                                    </div>
                                    <div className="space-y-3">
                                        {details.transform_summary.map((t, i) => (
                                            <motion.div 
                                                key={i}
                                                initial={{ opacity: 0, x: -10 }}
                                                animate={{ opacity: 1, x: 0, transition: { delay: i * 0.1 } }}
                                                className="flex items-center gap-4 group"
                                            >
                                                <div className="h-10 w-10 rounded-full bg-indigo-500/5 border border-indigo-500/20 flex items-center justify-center text-xs font-bold text-indigo-500 shadow-inner font-mono">
                                                    {i + 1}
                                                </div>
                                                <div className="flex-1 p-4 bg-muted/10 border border-border rounded-[1rem] group-hover:bg-muted/20 transition-all">
                                                    <span className="text-sm font-bold text-foreground uppercase tracking-tight font-heading">{t}</span>
                                                </div>
                                                {i < details.transform_summary.length - 1 && (
                                                    <div className="flex flex-col items-center gap-1 opacity-20">
                                                        <ArrowRight className="h-4 w-4 rotate-90" />
                                                    </div>
                                                )}
                                            </motion.div>
                                        ))}
                                    </div>
                                </div>

                                {/* Training History */}
                                <div className="md:col-span-7 space-y-6">
                                    <div className="flex items-center gap-3">
                                        <div className="h-px bg-border flex-1" />
                                        <h4 className="text-[11px] font-bold text-muted-foreground uppercase tracking-[0.3em] font-heading">Learning Telemetry</h4>
                                        <div className="h-px bg-border flex-1" />
                                    </div>
                                    <div className="bg-muted/5 border border-border rounded-[1.5rem] p-6">
                                        {details.training_history?.loss ? (
                                            <TrainingHistoryChart history={details.training_history.loss} />
                                        ) : (
                                            <div className="h-[300px] flex items-center justify-center text-muted-foreground/30 font-bold uppercase tracking-widest text-xs italic">
                                                No historical data available
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : null}
            </DialogContent>
        </Dialog>
    );
}
