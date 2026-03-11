"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { getDatasets } from "@/features/inverse/api";
import { diagnoseEngines } from "@/features/evaluation/api";
import { EngineComparisonPanel } from "@/features/evaluation/components/EngineComparisonPanel";
import { PerformanceChart, MetricBarChart } from "@/features/evaluation/components/EvaluationCharts";
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
                    <div className="p-2 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-500/20">
                        <LineChart className="h-6 w-6 text-white" />
                    </div>
                    <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-foreground via-foreground/90 to-foreground/80 font-sans">
                        Model Evaluation
                    </h1>
                </div>
                <p className="text-muted-foreground font-medium ml-12 italic">Benchmark engine calibration and predictive fidelity across high-dimensional objectives.</p>
                <div className="absolute -top-10 -right-10 opacity-5 pointer-events-none">
                    <TrendingUp className="h-64 w-64 text-indigo-500 rotate-12" />
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
                                <Alert variant="destructive" className="border-destructive/20 bg-destructive/10">
                                    <AlertCircle className="h-4 w-4 text-destructive" />
                                    <AlertTitle className="text-destructive font-black uppercase tracking-tight">Benchmark Failed</AlertTitle>
                                    <AlertDescription className="text-destructive/80 font-medium italic">
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
                                <div className="grid grid-cols-2 gap-8 mb-8 min-w-0">
                                    <div className="h-[350px] bg-muted/30 backdrop-blur-sm rounded-2xl border border-border shadow-sm animate-pulse min-w-0" />
                                    <div className="h-[350px] bg-muted/30 backdrop-blur-sm rounded-2xl border border-border shadow-sm animate-pulse min-w-0" />
                                </div>
                                <div className="flex flex-col items-center justify-center p-20 bg-muted/5 rounded-3xl border border-dashed border-border transition-all">
                                    <Loader2 className="h-10 w-10 animate-spin text-indigo-500 mb-6" />
                                    <span className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em]">Processing diagnostics</span>
                                    <div className="mt-3 text-[10px] text-muted-foreground/60 animate-pulse font-medium italic tracking-wide">Computing Kolmogorov-Smirnov statistics...</div>
                                </div>
                            </motion.div>
                        ) : result ? (
                            <motion.div 
                                key="results"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="space-y-8"
                            >
                                <div className="grid grid-cols-2 gap-8 min-w-0">
                                    <div className="min-w-0">
                                        <PerformanceChart
                                            title="ECDF (Calibration)"
                                            description="Empirical Cumulative Distribution Function of residuals. Focus on top-left area."
                                            data={result.ecdf}
                                            xAxisLabel="Normalized Residual"
                                            yAxisLabel="Cumulative Prob."
                                        />
                                    </div>
                                    <div className="min-w-0">
                                        <PerformanceChart
                                            title="PIT (Probabilistic Calibration)"
                                            description="Calibration of uncertainty. Perfectly calibrated models align with diagonal."
                                            data={result.pit}
                                            xAxisLabel="PIT Value"
                                            yAxisLabel="Cumulative Frequency"
                                            showIdeal
                                        />
                                    </div>
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
                                        <Card className="border-border bg-card/80 backdrop-blur-md shadow-2xl h-full overflow-hidden flex flex-col rounded-3xl">
                                            <CardHeader className="py-6 px-8 bg-muted/10 border-b border-border flex flex-row items-center justify-between space-y-0">
                                                <CardTitle className="text-xs font-black flex items-center gap-3 text-foreground uppercase tracking-widest">
                                                    <Trophy className="h-4 w-4 text-amber-500" />
                                                    Engine Leaderboard
                                                </CardTitle>
                                                <Badge variant="outline" className="bg-background text-[9px] font-black border-border px-3 rounded-full opacity-60">
                                                    TOP PERFORMERS
                                                </Badge>
                                            </CardHeader>
                                            <CardContent className="p-0 grow">
                                                <Table>
                                                    <TableHeader className="bg-muted/5">
                                                        <TableRow className="hover:bg-transparent border-border/50 uppercase tracking-tighter">
                                                            <TableHead className="font-black text-muted-foreground/40 text-[9px] h-12 px-8 tracking-[0.2em] uppercase">Rank & Engine</TableHead>
                                                            <TableHead className="text-right font-black text-muted-foreground/40 text-[9px] h-12 px-8 tracking-[0.2em] uppercase">MACE Score</TableHead>
                                                        </TableRow>
                                                    </TableHeader>
                                                    <TableBody>
                                                        {Object.entries(result.mace)
                                                            .sort(([, a], [, b]) => a - b)
                                                            .map(([engine, score], index) => (
                                                                <TableRow key={engine} className="border-border/50 hover:bg-muted/30 transition-all group">
                                                                    <TableCell className="py-5 px-8">
                                                                        <div className="flex items-center gap-4">
                                                                            <span className={`flex items-center justify-center w-7 h-7 rounded-lg text-[10px] font-black ${
                                                                                index === 0 ? "bg-amber-500/20 text-amber-500 border border-amber-500/20" : "bg-muted text-muted-foreground/60 border border-border"
                                                                            }`}>
                                                                                {index + 1}
                                                                            </span>
                                                                            <div className="flex flex-col">
                                                                                <span className="font-black text-foreground text-sm group-hover:text-indigo-500 transition-colors tracking-tight uppercase">{engine}</span>
                                                                                <div className="flex items-center gap-2 mt-1">
                                                                                    {score < 0.05 && <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />}
                                                                                    <span className={`text-[9px] font-black uppercase tracking-widest ${
                                                                                        score < 0.05 ? "text-emerald-500" : "text-muted-foreground/40"
                                                                                    }`}>
                                                                                        {score < 0.05 ? "Elite precision" : "Baseline"}
                                                                                    </span>
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    </TableCell>
                                                                    <TableCell className="text-right py-5 px-8">
                                                                        <div className="flex flex-col items-end">
                                                                            <span className="font-mono text-xs font-black text-foreground tabular-nums">
                                                                                {score.toFixed(4)}
                                                                            </span>
                                                                            <div className="w-20 h-1.5 bg-muted rounded-full mt-2.5 overflow-hidden border border-border/10">
                                                                                <div 
                                                                                    className={`h-full rounded-full transition-all duration-1000 ${score < 0.05 ? "bg-emerald-500" : "bg-indigo-500"}`}
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
                                    <Alert className="bg-indigo-500/5 border-indigo-500/10 text-foreground rounded-3xl shadow-2xl">
                                        <div className="flex items-start gap-4">
                                            <div className="bg-indigo-500/10 p-2 rounded-xl border border-indigo-500/20">
                                                <Sparkles className="h-5 w-5 text-indigo-500" />
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <AlertTitle className="text-foreground font-black uppercase tracking-widest text-sm mb-3 opacity-90">System Insights</AlertTitle>
                                                <AlertDescription>
                                                    <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-3">
                                                        {result.warnings.map((w, i) => (
                                                            <li key={i} className="flex items-center gap-3 text-[11px] font-bold text-muted-foreground hover:text-indigo-500 transition-colors group cursor-default">
                                                                <div className="h-1.5 w-1.5 bg-indigo-500/40 rounded-full shrink-0 group-hover:scale-125 transition-transform" />
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
                                className="flex flex-col items-center justify-center h-[600px] border-2 border-dashed border-border rounded-[2.5rem] bg-muted/5 backdrop-blur-[2px] p-10 text-center relative overflow-hidden group"
                            >
                                <div className="absolute inset-0 bg-linear-to-br from-indigo-500/5 via-transparent to-teal-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />
                                <div className="bg-background border border-border p-8 rounded-[2rem] shadow-2xl shadow-indigo-500/5 mb-10 relative z-10 transition-all duration-500 group-hover:scale-110 group-hover:rotate-3">
                                    <TrendingUp className="h-12 w-12 text-indigo-500" />
                                </div>
                                <h3 className="text-3xl font-black text-foreground mb-4 relative z-10 tracking-tight uppercase">Performance Benchmarking</h3>
                                <p className="text-muted-foreground/70 max-w-sm mb-12 relative z-10 font-medium italic leading-relaxed">
                                    Gain deep insights into your model's reliability. Select target datasets and engine candidates to begin the comparative audit.
                                </p>
                                <div className="flex items-center gap-3 text-indigo-500 font-black text-[10px] uppercase tracking-[0.25em] relative z-10 bg-indigo-500/5 px-6 py-3 rounded-2xl border border-indigo-500/20 shadow-lg shadow-indigo-500/10">
                                    <Sparkles className="h-4 w-4 animate-pulse" />
                                    Awaiting Calibration Input
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
