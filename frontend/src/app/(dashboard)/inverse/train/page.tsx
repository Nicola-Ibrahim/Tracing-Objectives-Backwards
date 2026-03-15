"use client";

import { useQuery, useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { getDatasets, trainEngine } from "@/features/inverse/api";
import { TrainEngineForm } from "@/features/inverse/components/TrainEngineForm";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
} from "@/components/ui/card";
import { TrainEngineRequest, EngineListItem } from "@/features/inverse/types";
import { TrainingHistoryChart } from "@/features/inverse/components/TrainingHistoryChart";
import { useState } from "react";
import { CheckCircle2, AlertCircle, Cpu, Zap, Activity, Clock } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function TrainEnginePage() {
    const [lastResult, setLastResult] = useState<any>(null);

    const { data: datasets, isLoading: datasetsLoading } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
    });

    const mutation = useMutation({
        mutationFn: (params: TrainEngineRequest) => trainEngine(params),
        onSuccess: (data) => {
            setLastResult({ success: true, data });
        },
        onError: (error: any) => {
            setLastResult({ success: false, error: error.message });
        },
    });

    const datasetNames = datasets?.map((d) => d.name) || [];

    return (
        <div className="space-y-8 max-w-full mx-auto pb-16 px-4 md:px-0">
            <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col gap-2 relative"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-600 rounded-[1rem]">
                        <Cpu className="h-6 w-6 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-foreground via-foreground/90 to-foreground/80 font-heading">
                        Engine Construction
                    </h1>
                </div>
                <p className="text-muted-foreground font-medium ml-12 italic">Train high-fidelity inverse models (GBPI, MDN).</p>
                <div className="absolute -top-10 -right-10 opacity-5 pointer-events-none">
                    <Zap className="h-64 w-64 text-indigo-500 rotate-12" />
                </div>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0, transition: { delay: 0.1 } }}
                    className="md:col-span-5"
                >
                    <Card className="border-border overflow-hidden rounded-[1rem] bg-card transition-all">
                        <CardHeader className="bg-muted/30 border-b border-border py-6 px-8 flex flex-row items-center justify-between space-y-0">
                            <div className="space-y-1">
                                <CardTitle className="text-xl font-bold text-foreground tracking-tight uppercase font-heading">
                                    Configuration
                                </CardTitle>
                                <CardDescription className="text-muted-foreground text-[10px] font-bold uppercase tracking-[0.2em] opacity-70 font-heading">
                                    Solver & Pipeline Setup
                                </CardDescription>
                            </div>
                            <div className="bg-background p-2 rounded-[1rem] border border-border">
                                <Activity className="h-5 w-5 text-indigo-500" />
                            </div>
                        </CardHeader>
                        <CardContent className="p-8">
                            <TrainEngineForm
                                datasets={datasetNames}
                                isLoading={mutation.isPending || datasetsLoading}
                                onSubmit={async (params) => { mutation.mutate(params); }}
                            />
                        </CardContent>
                    </Card>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0, transition: { delay: 0.2 } }}
                    className="h-full md:col-span-7"
                >
                    <Card className="border-border overflow-hidden rounded-[1rem] bg-card transition-all h-full">
                        <CardHeader className="bg-muted/30 border-b border-border py-6 px-8 flex flex-row items-center justify-between space-y-0">
                            <div className="space-y-1">
                                <CardTitle className="text-xl font-bold text-foreground tracking-tight uppercase font-heading">
                                    Telemetry
                                </CardTitle>
                                <CardDescription className="text-muted-foreground text-[10px] font-bold uppercase tracking-[0.2em] opacity-70 font-heading">
                                    Results & Metrics
                                </CardDescription>
                            </div>
                            <div className="bg-background p-2 rounded-[1rem] border border-border">
                                <Clock className="h-5 w-5 text-emerald-500" />
                            </div>
                        </CardHeader>
                        <CardContent className="p-8 h-full flex flex-col">
                            {!lastResult ? (
                                <div className="flex-1 flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed border-border rounded-[1rem] bg-muted/10 group transition-all hover:bg-muted/20">
                                    <div className="bg-background p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
                                        <Cpu className="h-8 w-8 text-muted-foreground/30" />
                                    </div>
                                    <span className="text-muted-foreground/50 font-bold uppercase tracking-[0.2em] text-[10px] font-heading">
                                        No active training session
                                    </span>
                                </div>
                            ) : lastResult.success ? (
                                <div className="space-y-6">
                                    <Alert className="bg-emerald-500/10 border-emerald-500/20 text-emerald-500 rounded-[1rem] border-l-4 border-l-emerald-500">
                                        <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                                        <AlertTitle className="font-bold font-heading">Construction Complete</AlertTitle>
                                        <AlertDescription className="font-medium">
                                            Engine version <span className="font-bold text-emerald-600 dark:text-emerald-400 font-mono">{lastResult.data.engine_version}</span> ready for inference.
                                        </AlertDescription>
                                    </Alert>
                                    <div className="grid grid-cols-2 gap-4 text-sm mt-6">
                                        <div className="p-4 bg-muted/30 rounded-[1rem] border border-border">
                                            <p className="text-muted-foreground text-[10px] font-bold uppercase tracking-widest mb-2 font-heading">Duration</p>
                                            <p className="text-xl font-bold text-foreground font-mono">{(lastResult.data.duration_seconds || 0).toFixed(2)}<span className="text-xs text-muted-foreground ml-1">sec</span></p>
                                        </div>
                                        <div className="p-4 bg-muted/30 rounded-[1rem] border border-border">
                                            <p className="text-muted-foreground text-[10px] font-bold uppercase tracking-widest mb-2 font-heading">Populated</p>
                                            <p className="text-xl font-bold text-foreground font-mono">{lastResult.data.n_train_samples}<span className="text-xs text-muted-foreground ml-1">obs</span></p>
                                        </div>
                                    </div>

                                    {lastResult.data.training_history?.loss && (
                                        <div className="mt-8">
                                            <TrainingHistoryChart 
                                                history={lastResult.data.training_history.loss} 
                                            />
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <Alert variant="destructive" className="rounded-[1rem] border-rose-500/20 bg-rose-500/10 border-l-4 border-l-rose-500">
                                    <AlertCircle className="h-4 w-4 text-rose-500" />
                                    <AlertTitle className="font-bold text-rose-500 font-heading">Training Exception</AlertTitle>
                                    <AlertDescription className="font-medium text-rose-800 dark:text-rose-400">{lastResult.error}</AlertDescription>
                                </Alert>
                            )}
                        </CardContent>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
}
