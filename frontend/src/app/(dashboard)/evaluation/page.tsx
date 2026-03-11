"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { getDatasets } from "@/features/inverse/api";
import { diagnoseEngines } from "@/features/evaluation/api";
import { EngineComparisonPanel } from "@/features/evaluation/components/EngineComparisonPanel";
import { PerformanceChart, MetricBarChart } from "@/features/evaluation/components/Charts";
import { DiagnoseRequest, DiagnoseResponse } from "@/features/evaluation/types";
import { LineChart, AlertCircle, TrendingUp, Info, Table as TableIcon, Loader2, Sparkles, Trophy } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function EvaluationPage() {
    const [result, setResult] = useState<DiagnoseResponse | null>(null);
    const [isLocalLoading, setIsLocalLoading] = useState(false);

    const { data: datasets = [], isLoading: loadingDatasets } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
        select: (data) => data.map((d) => d.name),
    });

    const mutation = useMutation({
        mutationFn: diagnoseEngines,
        onSuccess: (data) => {
            setResult(data);
        },
    });

    const handleDiagnose = async (params: DiagnoseRequest) => {
        setIsLocalLoading(true);
        try {
            // First attempt to retrieve cached results
            const cachedResult = await diagnoseEngines(params);
            setResult(cachedResult);
        } catch (error) {
            // If cache miss (404), trigger new diagnosis
            await mutation.mutateAsync(params);
        } finally {
            setIsLocalLoading(false);
        }
    };

    const isPending = isLocalLoading || mutation.isPending;

    return (
        <div className="space-y-8 max-w-7xl mx-auto pb-16 px-4 md:px-0">
            <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col gap-2 relative"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-200">
                        <LineChart className="h-6 w-6 text-white" />
                    </div>
                    <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-slate-900 via-indigo-900 to-indigo-800 font-sans">
                        Model Evaluation
                    </h1>
                </div>
                <p className="text-slate-500 font-medium ml-12">Benchmark engine calibration and predictive fidelity across high-dimensional objectives.</p>
                <div className="absolute -top-10 -right-10 opacity-5 pointer-events-none">
                    <TrendingUp className="h-64 w-64 text-indigo-900 rotate-12" />
                </div>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-1">
                    <EngineComparisonPanel
                        datasets={datasets}
                        onDiagnose={handleDiagnose}
                        isLoading={isPending}
                    />
                </div>

                <div className="lg:col-span-3 space-y-8 min-h-[600px]">
                    <AnimatePresence mode="wait">
                        {mutation.isError ? (
                            <motion.div
                                key="error"
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                            >
                                <Alert variant="destructive" className="border-rose-200 bg-rose-50/50">
                                    <AlertCircle className="h-4 w-4 text-rose-600" />
                                    <AlertTitle className="text-rose-900 font-bold">Benchmark Failed</AlertTitle>
                                    <AlertDescription className="text-rose-700">
                                        {(mutation.error as any)?.response?.data?.detail || "An unexpected error occurred during model diagnosis."}
                                    </AlertDescription>
                                </Alert>
                            </motion.div>
                        ) : isPending ? (
                            <motion.div 
                                key="loading"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="space-y-8"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                    <div className="h-[380px] bg-white/60 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-sm animate-pulse" />
                                    <div className="h-[380px] bg-white/60 backdrop-blur-sm rounded-2xl border border-slate-200 shadow-sm animate-pulse" />
                                </div>
                                <div className="flex flex-col items-center justify-center p-20 bg-slate-50/30 rounded-2xl border border-dashed border-slate-200">
                                    <Loader2 className="h-10 w-10 animate-spin text-indigo-500 mb-4" />
                                    <span className="text-sm font-bold text-slate-400 uppercase tracking-widest">Processing diagnostics</span>
                                    <div className="mt-2 text-xs text-slate-400 animate-pulse">Computing Kolmogorov-Smirnov statistics...</div>
                                </div>
                            </motion.div>
                        ) : result ? (
                            <motion.div 
                                key="results"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="space-y-8"
                            >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                    <PerformanceChart
                                        title="ECDF (Calibration)"
                                        description="Empirical Cumulative Distribution Function of residuals. Focus on top-left area."
                                        data={result.ecdf}
                                        xAxisLabel="Normalized Residual"
                                        yAxisLabel="Cumulative Prob."
                                    />
                                    <PerformanceChart
                                        title="PIT (Probabilistic Calibration)"
                                        description="Calibration of uncertainty. Perfectly calibrated models align with diagonal."
                                        data={result.pit}
                                        xAxisLabel="PIT Value"
                                        yAxisLabel="Cumulative Frequency"
                                        showIdeal
                                    />
                                </div>

                                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-stretch">
                                    <div className="lg:col-span-3">
                                        <MetricBarChart
                                            title="MACE Comparison"
                                            description="Mean Absolute Calibration Error. Lower is more reliable."
                                            data={result.mace}
                                            yAxisLabel="MACE Score"
                                        />
                                    </div>
                                    <div className="lg:col-span-2">
                                        <Card className="border-slate-200/60 bg-white/80 backdrop-blur-md shadow-lg shadow-slate-200/50 h-full overflow-hidden flex flex-col">
                                            <CardHeader className="py-5 px-6 bg-slate-50/40 border-b border-slate-100 flex flex-row items-center justify-between space-y-0">
                                                <CardTitle className="text-sm font-bold flex items-center gap-2 text-slate-800">
                                                    <Trophy className="h-4 w-4 text-amber-500" />
                                                    Engine Leaderboard
                                                </CardTitle>
                                                <Badge variant="outline" className="bg-white text-[10px] font-bold py-0.5">
                                                    Top Performers
                                                </Badge>
                                            </CardHeader>
                                            <CardContent className="p-0 grow">
                                                <Table>
                                                    <TableHeader className="bg-slate-50/20">
                                                        <TableRow className="hover:bg-transparent border-slate-100 uppercase tracking-tighter">
                                                            <TableHead className="font-bold text-slate-400 text-[10px] h-10 px-6">Rank & Engine</TableHead>
                                                            <TableHead className="text-right font-bold text-slate-400 text-[10px] h-10 px-6">MACE Score</TableHead>
                                                        </TableRow>
                                                    </TableHeader>
                                                    <TableBody>
                                                        {Object.entries(result.mace)
                                                            .sort(([, a], [, b]) => a - b)
                                                            .map(([engine, score], index) => (
                                                                <TableRow key={engine} className="border-slate-50 hover:bg-indigo-50/30 transition-all group">
                                                                    <TableCell className="py-4 px-6">
                                                                        <div className="flex items-center gap-3">
                                                                            <span className={`flex items-center justify-center w-6 h-6 rounded-md text-[10px] font-bold ${
                                                                                index === 0 ? "bg-amber-100 text-amber-600" : "bg-slate-100 text-slate-500"
                                                                            }`}>
                                                                                {index + 1}
                                                                            </span>
                                                                            <div className="flex flex-col">
                                                                                <span className="font-bold text-slate-700 text-sm group-hover:text-indigo-600 transition-colors">{engine}</span>
                                                                                <div className="flex items-center gap-1.5 mt-1">
                                                                                    {score < 0.05 && <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />}
                                                                                    <span className={`text-[10px] font-bold uppercase tracking-tight ${
                                                                                        score < 0.05 ? "text-emerald-600/80" : "text-slate-400"
                                                                                    }`}>
                                                                                        {score < 0.05 ? "Elite precision" : "Baseline"}
                                                                                    </span>
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    </TableCell>
                                                                    <TableCell className="text-right py-4 px-6">
                                                                        <div className="flex flex-col items-end">
                                                                            <span className="font-mono text-sm font-extrabold text-slate-900 tabular-nums">
                                                                                {score.toFixed(4)}
                                                                            </span>
                                                                            <div className="w-16 h-1 bg-slate-100 rounded-full mt-2 overflow-hidden">
                                                                                <div 
                                                                                    className={`h-full rounded-full ${score < 0.05 ? "bg-emerald-400" : "bg-indigo-400"}`}
                                                                                    style={{ width: `${Math.max(5, 100 - score * 500)}%` }}
                                                                                />
                                                                            </div>
                                                                        </div>
                                                                    </TableCell>
                                                                </TableRow>
                                                            ))}
                                                    </TableBody>
                                                </Table>
                                            </CardContent>
                                        </Card>
                                    </div>
                                </div>

                                {result.warnings.length > 0 && (
                                    <Alert className="bg-indigo-50/50 border-indigo-100 text-indigo-900 rounded-2xl shadow-sm">
                                        <div className="flex items-start gap-3">
                                            <div className="bg-white p-1.5 rounded-lg shadow-sm">
                                                <Sparkles className="h-4 w-4 text-indigo-500" />
                                            </div>
                                            <div>
                                                <AlertTitle className="text-indigo-900 font-bold tracking-tight">System Insights</AlertTitle>
                                                <AlertDescription className="mt-2">
                                                    <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
                                                        {result.warnings.map((w, i) => (
                                                            <li key={i} className="flex items-center gap-2 text-xs font-medium text-indigo-800/80">
                                                                <div className="h-1 w-1 bg-indigo-400 rounded-full shrink-0" />
                                                                {w}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </AlertDescription>
                                            </div>
                                        </div>
                                    </Alert>
                                )}
                            </motion.div>
                        ) : (
                            <motion.div 
                                key="empty"
                                initial={{ opacity: 0, scale: 0.98 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="flex flex-col items-center justify-center h-[600px] border-2 border-dashed border-slate-200/60 rounded-3xl bg-slate-50/20 backdrop-blur-[2px] p-10 text-center relative overflow-hidden group"
                            >
                                <div className="absolute inset-0 bg-linear-to-br from-indigo-50/50 via-transparent to-teal-50/30 opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />
                                <div className="bg-white p-6 rounded-3xl shadow-xl shadow-slate-200/50 mb-8 relative z-10 transition-transform group-hover:scale-110 duration-500">
                                    <TrendingUp className="h-10 w-10 text-indigo-600" />
                                </div>
                                <h3 className="text-2xl font-black text-slate-900 mb-3 relative z-10 tracking-tight">Performance Benchmarking</h3>
                                <p className="text-slate-500 max-w-sm mb-10 relative z-10 font-medium leading-relaxed">
                                    Gain deep insights into your model's reliability. Select target datasets and engine candidates to begin the comparative audit.
                                </p>
                                <div className="flex items-center gap-2 text-indigo-600 font-bold text-xs uppercase tracking-widest relative z-10 bg-indigo-50/50 px-4 py-2 rounded-full border border-indigo-100">
                                    <Sparkles className="h-4 w-4" />
                                    Awaiting input
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
