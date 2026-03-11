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
        <div className="space-y-8 max-w-7xl mx-auto pb-16 px-4 md:px-0">
            <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col gap-2 relative"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-200">
                        <Cpu className="h-6 w-6 text-white" />
                    </div>
                    <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-slate-900 via-indigo-900 to-indigo-800 font-sans">
                        Engine Construction
                    </h1>
                </div>
                <p className="text-slate-500 font-medium ml-12">Train high-fidelity inverse models (GBPI, MDN).</p>
                <div className="absolute -top-10 -right-10 opacity-5 pointer-events-none">
                    <Zap className="h-64 w-64 text-indigo-900 rotate-12" />
                </div>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-8">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0, transition: { delay: 0.1 } }}
                >
                    <Card className="border-slate-200/60 shadow-lg shadow-slate-200/40 overflow-hidden md:col-span-1 rounded-3xl bg-white/80 backdrop-blur-sm">
                        <CardHeader className="bg-slate-50/40 border-b border-slate-100 py-6 px-8 flex flex-row items-center justify-between space-y-0">
                            <div className="space-y-1">
                                <CardTitle className="text-xl font-black text-slate-800 tracking-tight">
                                    Configuration
                                </CardTitle>
                                <CardDescription className="text-slate-500 text-xs font-bold uppercase tracking-widest opacity-60">
                                    Solver & Pipeline Setup
                                </CardDescription>
                            </div>
                            <div className="bg-white p-2 rounded-xl shadow-sm border border-slate-100">
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
                >
                    <Card className="border-slate-200/60 shadow-lg shadow-slate-200/40 overflow-hidden md:col-span-1 rounded-3xl bg-white/80 backdrop-blur-sm h-full">
                        <CardHeader className="bg-slate-50/40 border-b border-slate-100 py-6 px-8 flex flex-row items-center justify-between space-y-0">
                            <div className="space-y-1">
                                <CardTitle className="text-xl font-black text-slate-800 tracking-tight">
                                    Telemetry
                                </CardTitle>
                                <CardDescription className="text-slate-500 text-xs font-bold uppercase tracking-widest opacity-60">
                                    Results & Metrics
                                </CardDescription>
                            </div>
                            <div className="bg-white p-2 rounded-xl shadow-sm border border-slate-100">
                                <Clock className="h-5 w-5 text-teal-500" />
                            </div>
                        </CardHeader>
                        <CardContent className="p-8 h-full">
                            {!lastResult ? (
                                <div className="flex flex-col items-center justify-center h-full min-h-[400px] border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/30 group">
                                    <div className="bg-white p-4 rounded-full shadow-md mb-4 group-hover:scale-110 transition-transform">
                                        <Cpu className="h-8 w-8 text-slate-300" />
                                    </div>
                                    <span className="text-slate-400 font-bold uppercase tracking-widest text-xs">
                                        No active training session
                                    </span>
                                </div>
                            ) : lastResult.success ? (
                                <div className="space-y-6">
                                    <Alert className="bg-emerald-50 border-emerald-200 text-emerald-800 rounded-2xl shadow-sm border-l-4 border-l-emerald-500">
                                        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                                        <AlertTitle className="font-bold">Construction Complete</AlertTitle>
                                        <AlertDescription className="font-medium">
                                            Engine version <span className="font-black text-emerald-900">{lastResult.data.engine_version}</span> ready for inference.
                                        </AlertDescription>
                                    </Alert>
                                    <div className="grid grid-cols-2 gap-4 text-sm mt-6">
                                        <div className="p-4 bg-slate-50/50 rounded-2xl border border-slate-100 shadow-inner">
                                            <p className="text-slate-400 text-[10px] font-black uppercase tracking-widest mb-2">Duration</p>
                                            <p className="text-xl font-black text-slate-900">{lastResult.data.duration_seconds.toFixed(2)}<span className="text-xs text-slate-500 ml-1">sec</span></p>
                                        </div>
                                        <div className="p-4 bg-slate-50/50 rounded-2xl border border-slate-100 shadow-inner">
                                            <p className="text-slate-400 text-[10px] font-black uppercase tracking-widest mb-2">Populated</p>
                                            <p className="text-xl font-black text-slate-900">{lastResult.data.n_train_samples}<span className="text-xs text-slate-500 ml-1">obs</span></p>
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
                                <Alert variant="destructive" className="rounded-2xl border-rose-200 bg-rose-50/50 border-l-4 border-l-rose-500">
                                    <AlertCircle className="h-4 w-4 text-rose-600" />
                                    <AlertTitle className="font-bold">Training Exception</AlertTitle>
                                    <AlertDescription className="font-medium text-rose-800">{lastResult.error}</AlertDescription>
                                </Alert>
                            )}
                        </CardContent>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
}
