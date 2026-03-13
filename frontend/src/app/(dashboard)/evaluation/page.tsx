"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { getDatasets } from "@/features/inverse/api";
import { diagnoseEngines } from "@/features/evaluation/api";
import { EngineComparisonPanel } from "@/features/evaluation/components/EngineComparisonPanel";
import { PerformanceChart } from "@/features/evaluation/components/EvaluationCharts";
import { DiagnoseRequest, DiagnoseResponse } from "@/features/evaluation/types";
import { 
    LineChart, 
    AlertCircle, 
    TrendingUp, 
    Loader2, 
    Sparkles, 
    Trophy, 
    Target, 
    ShieldCheck, 
    Activity,
    Brain,
    Scale,
    Layers,
    Info
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
    Tooltip, 
    TooltipContent, 
    TooltipProvider, 
    TooltipTrigger 
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const METRIC_DEFINITIONS = {
    MACE: "Mean Absolute Calibration Error: Average discrepancy between nominal and empirical coverage. Lower is better.",
    CRPS: "Standardized Continuous Ranked Probability Score: Evaluates absolute accuracy and uncertainty sharpness. Lower is better.",
    Diversity: "Measures the variety or spread of generated candidates. Higher values indicate better exploration of the solution space.",
    RelWidth: "Relative Interval Width: Average size of prediction intervals relative to data scale. Narrower is more precise."
};

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
            const cachedResult = await diagnoseEngines(params);
            setResult(cachedResult);
        } catch (error) {
            await mutation.mutateAsync(params);
        } finally {
            setIsLocalLoading(false);
        }
    };

    const isPending = isLocalLoading || mutation.isPending;

    return (
        <TooltipProvider>
            <div className="space-y-8 max-w-full mx-auto pb-16 px-4 md:px-0 bg-background/50 min-h-screen transition-colors duration-500">
                <motion.div 
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col gap-2 relative mt-4"
                >
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-linear-to-br from-indigo-600 to-violet-700 rounded-[1rem] shadow-lg shadow-indigo-500/20">
                            <LineChart className="h-6 w-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold tracking-tight text-foreground font-heading uppercase">
                                Model Evaluation
                            </h1>
                            <p className="text-muted-foreground font-medium italic text-sm">Benchmark engine calibration and predictive fidelity across high-dimensional objectives.</p>
                        </div>
                    </div>
                </motion.div>

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                    <div className="lg:col-span-1">
                        <EngineComparisonPanel
                            datasets={datasets}
                            onDiagnose={handleDiagnose}
                            isLoading={isPending}
                        />
                    </div>

                    <div className="lg:col-span-3 space-y-8 min-h-[700px]">
                        <AnimatePresence mode="wait">
                            {mutation.isError ? (
                                <motion.div
                                    key="error"
                                    initial={{ opacity: 0, scale: 0.98 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.98 }}
                                >
                                    <Alert variant="destructive" className="border-destructive/20 bg-destructive/5 rounded-[1rem] p-6">
                                        <div className="flex items-center gap-4">
                                            <div className="p-3 bg-destructive/10 rounded-[1rem]">
                                                <AlertCircle className="h-6 w-6 text-destructive" />
                                            </div>
                                            <div>
                                                <AlertTitle className="text-destructive font-bold uppercase tracking-widest text-xs font-heading">Benchmark Failed</AlertTitle>
                                                <AlertDescription className="text-destructive/80 font-medium italic mt-1">
                                                    {(mutation.error as any)?.response?.data?.detail?.message || "An unexpected error occurred during model diagnosis."}
                                                </AlertDescription>
                                            </div>
                                        </div>
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
                                        <div className="h-[400px] bg-muted/20 backdrop-blur-sm rounded-[1rem] border border-border animate-pulse" />
                                        <div className="h-[400px] bg-muted/20 backdrop-blur-sm rounded-[1rem] border border-border animate-pulse" />
                                    </div>
                                    <div className="flex flex-col items-center justify-center p-24 bg-muted/5 rounded-[1rem] border-2 border-dashed border-border transition-all">
                                        <div className="relative">
                                            <div className="absolute inset-0 blur-xl bg-indigo-500/20 animate-pulse rounded-full" />
                                            <Loader2 className="h-12 w-12 animate-spin text-indigo-500 relative z-10" />
                                        </div>
                                        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-[0.4em] mt-8 font-heading">Processing Diagnostics</span>
                                        <div className="mt-4 text-[11px] text-muted-foreground/50 animate-pulse font-bold italic tracking-wider">Computing objective-space residuals and probabilistic coverage...</div>
                                    </div>
                                </motion.div>
                            ) : result ? (
                                <motion.div 
                                    key="results"
                                    initial={{ opacity: 0, scale: 0.98 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="space-y-8"
                                >
                                    <Tabs defaultValue="decision" className="w-full">
                                        <div className="flex items-center justify-between mb-8 bg-muted/30 p-1.5 rounded-[1rem] border border-border/50 backdrop-blur-sm">
                                            <TabsList className="bg-transparent h-12 p-0 gap-2">
                                                <TabsTrigger 
                                                    value="decision" 
                                                    className="rounded-full px-8 h-10 data-[state=active]:bg-background data-[state=active]:shadow-lg data-[state=active]:text-indigo-500 font-bold uppercase text-[10px] tracking-widest transition-all duration-300 font-heading"
                                                >
                                                    <ShieldCheck className="h-3.5 w-3.5 mr-2" />
                                                    Decision Reliability
                                                </TabsTrigger>
                                                <TabsTrigger 
                                                    value="objective" 
                                                    className="rounded-full px-8 h-10 data-[state=active]:bg-background data-[state=active]:shadow-lg data-[state=active]:text-teal-500 font-bold uppercase text-[10px] tracking-widest transition-all duration-300 font-heading"
                                                >
                                                    <Target className="h-3.5 w-3.5 mr-2" />
                                                    Objective Accuracy
                                                </TabsTrigger>
                                            </TabsList>
                                            <div className="px-6 flex items-center gap-2">
                                                <Activity className="h-3.5 w-3.5 text-indigo-500/50" />
                                                <span className="text-[9px] font-bold text-muted-foreground/40 uppercase tracking-[0.2em] font-heading">Diagnostic Lens: Global Comparison</span>
                                            </div>
                                        </div>

                                        <TabsContent value="decision" className="space-y-8 animate-in fade-in zoom-in duration-500 mt-0">
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 min-w-0">
                                                {/* Calibration Reliability Plot (Distributional Engines only) */}
                                                <div className="min-w-0">
                                                    {Object.keys(result.decision_space.calibration_curves).length > 0 ? (
                                                        <PerformanceChart
                                                            title="Calibration Reliability"
                                                            description="Nominal vs Empirical coverage for distributional engines (INN, MDN)."
                                                            data={result.decision_space.calibration_curves}
                                                            xAxisLabel="Nominal Coverage"
                                                            yAxisLabel="Empirical Coverage"
                                                            showIdeal
                                                        />
                                                    ) : (
                                                        <div className="flex flex-col items-center justify-center p-12 bg-muted/5 rounded-[1rem] border border-border/50 h-[300px] text-center">
                                                            <Activity className="h-8 w-8 text-muted-foreground/20 mb-4" />
                                                            <span className="text-[10px] font-bold text-muted-foreground/30 uppercase tracking-widest font-heading">No Distributional Data</span>
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Uncertainty Profile Plot (ECDF for Interval Engines) */}
                                                <div className="min-w-0">
                                                    {Object.keys(result.decision_space.ecdf).length > 0 ? (
                                                        <PerformanceChart
                                                            title="Uncertainty Profile (ECDF)"
                                                            description="Empirical Cumulative Distribution Function of coverage for interval engines (GBPI)."
                                                            data={result.decision_space.ecdf}
                                                            xAxisLabel="Coverage Level"
                                                            yAxisLabel="Cumulative Density"
                                                        />
                                                    ) : (
                                                        <div className="flex flex-col items-center justify-center p-12 bg-muted/5 rounded-[1rem] border border-border/50 h-[300px] text-center">
                                                            <Layers className="h-8 w-8 text-muted-foreground/20 mb-4" />
                                                            <span className="text-[10px] font-bold text-muted-foreground/30 uppercase tracking-widest font-heading">No Interval Data</span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>

                                            <Card className="border-border bg-card/40 backdrop-blur-xl rounded-[1rem] overflow-hidden border-2">
                                                <CardHeader className="py-8 px-10 border-b border-border/50 bg-muted/10 flex flex-row items-center justify-between">
                                                    <div className="flex items-center gap-4">
                                                        <div className="p-3 bg-indigo-500/10 rounded-[1rem] border border-indigo-500/20">
                                                            <ShieldCheck className="h-5 w-5 text-indigo-500" />
                                                        </div>
                                                        <div>
                                                            <CardTitle className="text-sm font-bold uppercase tracking-widest text-foreground font-heading">Decision Space Metrics</CardTitle>
                                                            <CardDescription className="text-[11px] font-medium italic text-muted-foreground mt-1 tracking-tight">Reliability benchmarks for decision-space predictions.</CardDescription>
                                                        </div>
                                                    </div>
                                                    <Badge variant="outline" className="px-5 py-1.5 rounded-full bg-background/50 text-[10px] font-bold border-border/80 text-muted-foreground/60 tracking-widest uppercase font-heading">
                                                        P(x|y) Evaluation
                                                    </Badge>
                                                </CardHeader>
                                                <CardContent className="p-0">
                                                    <Table>
                                                        <TableHeader className="bg-muted/5">
                                                            <TableRow className="border-border/50 hover:bg-transparent">
                                                                <TableHead className="px-10 h-16 font-bold text-muted-foreground/30 text-[10px] uppercase tracking-[0.2em] font-heading">Engine Instance</TableHead>
                                                                <TableHead className="px-10 h-16 text-right font-bold text-muted-foreground/30 text-[10px] uppercase tracking-[0.2em] font-heading">
                                                                    <div className="flex items-center justify-end gap-2">
                                                                        MACE Error
                                                                        <Tooltip>
                                                                            <TooltipTrigger asChild>
                                                                                <Info className="h-3 w-3 cursor-help text-muted-foreground/40 hover:text-indigo-500 transition-colors" />
                                                                            </TooltipTrigger>
                                                                            <TooltipContent className="max-w-xs bg-card border-border text-foreground shadow-2xl p-3">
                                                                                <p className="font-bold mb-1 uppercase text-[10px] tracking-wider font-heading">Mean Absolute Calibration Error</p>
                                                                                <p className="font-medium text-[10px] leading-relaxed italic">{METRIC_DEFINITIONS.MACE}</p>
                                                                            </TooltipContent>
                                                                        </Tooltip>
                                                                    </div>
                                                                </TableHead>
                                                                <TableHead className="px-10 h-16 text-right font-bold text-muted-foreground/30 text-[10px] uppercase tracking-[0.2em] font-heading">
                                                                    <div className="flex items-center justify-end gap-2">
                                                                        S-CRPS
                                                                        <Tooltip>
                                                                            <TooltipTrigger asChild>
                                                                                <Info className="h-3 w-3 cursor-help text-muted-foreground/40 hover:text-indigo-500 transition-colors" />
                                                                            </TooltipTrigger>
                                                                            <TooltipContent className="max-w-xs bg-card border-border text-foreground shadow-2xl p-3">
                                                                                <p className="font-bold mb-1 uppercase text-[10px] tracking-wider font-heading">Standardized CRPS</p>
                                                                                <p className="font-medium text-[10px] leading-relaxed italic">{METRIC_DEFINITIONS.CRPS}</p>
                                                                            </TooltipContent>
                                                                        </Tooltip>
                                                                    </div>
                                                                </TableHead>
                                                                <TableHead className="px-10 h-16 text-right font-bold text-muted-foreground/30 text-[10px] uppercase tracking-[0.2em] font-heading">
                                                                    <div className="flex items-center justify-end gap-2">
                                                                        Diversity
                                                                        <Tooltip>
                                                                            <TooltipTrigger asChild>
                                                                                <Info className="h-3 w-3 cursor-help text-muted-foreground/40 hover:text-indigo-500 transition-colors" />
                                                                            </TooltipTrigger>
                                                                            <TooltipContent className="max-w-xs bg-card border-border text-foreground shadow-2xl p-3">
                                                                                <p className="font-bold mb-1 uppercase text-[10px] tracking-wider font-heading">Decision Diversity</p>
                                                                                <p className="font-medium text-[10px] leading-relaxed italic">{METRIC_DEFINITIONS.Diversity}</p>
                                                                            </TooltipContent>
                                                                        </Tooltip>
                                                                    </div>
                                                                </TableHead>
                                                                <TableHead className="px-10 h-16 text-right font-bold text-muted-foreground/30 text-[10px] uppercase tracking-[0.2em] font-heading">
                                                                    <div className="flex items-center justify-end gap-2">
                                                                        Rel. Width
                                                                        <Tooltip>
                                                                            <TooltipTrigger asChild>
                                                                                <Info className="h-3 w-3 cursor-help text-muted-foreground/40 hover:text-indigo-500 transition-colors" />
                                                                            </TooltipTrigger>
                                                                            <TooltipContent className="max-w-xs bg-card border-border text-foreground shadow-2xl p-3">
                                                                                <p className="font-bold mb-1 uppercase text-[10px] tracking-wider font-heading">Relative Interval Width</p>
                                                                                <p className="font-medium text-[10px] leading-relaxed italic">{METRIC_DEFINITIONS.RelWidth}</p>
                                                                            </TooltipContent>
                                                                        </Tooltip>
                                                                    </div>
                                                                </TableHead>
                                                            </TableRow>
                                                        </TableHeader>
                                                        <TableBody>
                                                            {Object.entries(result.decision_space.metrics).map(([engine, metrics]) => (
                                                                <TableRow key={engine} className="group border-border/50 hover:bg-indigo-500/5 transition-all duration-300">
                                                                    <TableCell className="px-10 py-6">
                                                                        <div className="flex items-center gap-4">
                                                                            <div className="flex flex-col">
                                                                                <span className="font-bold text-sm text-foreground uppercase tracking-tight group-hover:text-indigo-500 transition-colors font-heading">{engine}</span>
                                                                                <span className="text-[9px] font-bold text-muted-foreground/40 uppercase tracking-widest mt-1 font-heading">
                                                                                    {result.capabilities[engine] === "full_distribution" ? "Distributional Engine" : "Interval Engine"}
                                                                                </span>
                                                                            </div>
                                                                        </div>
                                                                    </TableCell>
                                                                    <TableCell className="px-10 py-6 text-right relative group/item">
                                                                        <span className="font-mono text-xs font-bold text-foreground tabular-nums">{(metrics.mace || metrics.mean_coverage_error || 0).toFixed(4)}</span>
                                                                        <div className="mt-2 text-[9px] font-bold uppercase text-indigo-500 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-1 group-hover:translate-y-0 font-heading">Lower is Better</div>
                                                                    </TableCell>
                                                                    <TableCell className="px-10 py-6 text-right relative">
                                                                        <span className="font-mono text-xs font-semibold text-muted-foreground tabular-nums">{(metrics.mean_crps || 0).toFixed(4)}</span>
                                                                        <div className="mt-2 text-[9px] font-bold uppercase text-indigo-500 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-1 group-hover:translate-y-0 font-heading">Lower is Better</div>
                                                                    </TableCell>
                                                                    <TableCell className="px-10 py-6 text-right relative">
                                                                        <span className="font-mono text-xs font-semibold text-muted-foreground tabular-nums">{(metrics.diversity || 0).toFixed(3)}</span>
                                                                        <div className="mt-2 text-[9px] font-bold uppercase text-violet-500 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-1 group-hover:translate-y-0 font-heading">Higher is Better</div>
                                                                    </TableCell>
                                                                    <TableCell className="px-10 py-6 text-right relative">
                                                                        <span className="font-mono text-xs font-semibold text-muted-foreground tabular-nums">{(metrics.interval_width || 0).toFixed(3)}</span>
                                                                        <div className="mt-2 text-[9px] font-bold uppercase text-indigo-500 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-y-1 group-hover:translate-y-0 font-heading">Lower is Better</div>
                                                                    </TableCell>
                                                                </TableRow>
                                                            ))}
                                                        </TableBody>
                                                    </Table>
                                                </CardContent>
                                            </Card>
                                        </TabsContent>

                                        <TabsContent value="objective" className="space-y-8 animate-in fade-in zoom-in duration-500 mt-0">
                                            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-stretch">
                                                <div className="lg:col-span-3 min-w-0">
                                                    <PerformanceChart
                                                        title="Global Accuracy Profile"
                                                        description="Empirical Cumulative Distribution Function of objective-space residuals."
                                                        data={result.objective_space.ecdf}
                                                        xAxisLabel="Normalized Residual (Distance)"
                                                        yAxisLabel="Cumulative Density"
                                                        xAxisType="log"
                                                    />
                                                </div>
                                                <div className="lg:col-span-2 min-w-0">
                                                    <Card className="border-border bg-card/8 backdrop-blur-md h-full rounded-[1rem] border-2 shadow-xl shadow-teal-500/5">
                                                        <CardHeader className="py-8 px-10 border-b border-border/50 bg-muted/10">
                                                            <div className="flex items-center gap-4">
                                                                <div className="p-3 bg-teal-500/10 rounded-[1rem] border border-teal-500/20">
                                                                    <Target className="h-5 w-5 text-teal-500" />
                                                                </div>
                                                                <div>
                                                                    <CardTitle className="text-xs font-bold uppercase tracking-[0.2em] text-foreground font-heading">Objective Metrics</CardTitle>
                                                                    <CardDescription className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-widest mt-1 font-heading">Accuracy Leaderboard</CardDescription>
                                                                </div>
                                                            </div>
                                                        </CardHeader>
                                                        <CardContent className="p-0">
                                                            <div className="p-8 space-y-6">
                                                                {Object.entries(result.objective_space.metrics)
                                                                    .sort(([, a], [, b]) => a.mean_best_shot - b.mean_best_shot)
                                                                    .map(([engine, metrics], index) => (
                                                                        <div key={engine} className="relative group p-6 rounded-[1.5rem] bg-muted/10 border border-border/50 hover:bg-teal-500/5 hover:border-teal-500/20 transition-all duration-300">
                                                                            <div className="flex items-center justify-between mb-4">
                                                                                <div className="flex items-center gap-4">
                                                                                    <div className={cn(
                                                                                        "w-8 h-8 rounded-[1rem] flex items-center justify-center text-[10px] font-bold border-2 font-mono",
                                                                                        index === 0 ? "bg-teal-500/20 text-teal-600 border-teal-500/30" : "bg-muted text-muted-foreground/40 border-border"
                                                                                    )}>
                                                                                        {index + 1}
                                                                                    </div>
                                                                                    <span className="font-bold text-xs uppercase tracking-tight text-foreground truncate max-w-[120px] font-heading">{engine}</span>
                                                                                </div>
                                                                                <div className="flex flex-col items-end">
                                                                                    <span className="font-mono text-sm font-bold text-foreground tabular-nums">{metrics.mean_best_shot.toFixed(4)}</span>
                                                                                    <span className="text-[8px] font-bold uppercase text-teal-500 tracking-widest mt-1 opacity-0 group-hover:opacity-100 transition-opacity font-heading">Lower is Better</span>
                                                                                </div>
                                                                            </div>
                                                                            
                                                                            <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border/10">
                                                                                <div className="group/metric">
                                                                                    <div className="text-[8px] font-bold text-muted-foreground/50 uppercase tracking-widest mb-1.5 flex items-center justify-between font-heading">
                                                                                        Mean Bias
                                                                                        <span className="text-[6px] text-teal-500 opacity-0 group-hover/metric:opacity-100 transition-opacity">Lower is Better</span>
                                                                                    </div>
                                                                                    <div className="font-mono text-[10px] font-semibold text-foreground">{metrics.mean_bias.toFixed(4)}</div>
                                                                                </div>
                                                                                <div className="group/metric">
                                                                                    <div className="text-[8px] font-bold text-muted-foreground/50 uppercase tracking-widest mb-1.5 flex items-center justify-between font-heading">
                                                                                        Dispersion
                                                                                        <span className="text-[6px] text-teal-500 opacity-0 group-hover/metric:opacity-100 transition-opacity">Lower is Better</span>
                                                                                    </div>
                                                                                    <div className="font-mono text-[10px] font-semibold text-foreground">{metrics.mean_dispersion.toFixed(4)}</div>
                                                                                </div>
                                                                            </div>

                                                                            {index === 0 && (
                                                                                <div className="absolute -top-2 -right-2 p-1.5 bg-background border border-teal-500/30 rounded-full shadow-lg">
                                                                                    <Sparkles className="h-3 w-3 text-teal-500 animate-pulse" />
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    ))}
                                                            </div>
                                                        </CardContent>
                                                    </Card>
                                                </div>
                                            </div>
                                        </TabsContent>
                                    </Tabs>

                                    {result.warnings.length > 0 && (
                                        <Alert className="bg-indigo-500/5 border-indigo-500/10 text-foreground rounded-[1rem] p-8 border-2">
                                            <div className="flex items-start gap-6">
                                                <div className="bg-indigo-500/10 p-4 rounded-[1rem] border border-indigo-500/20">
                                                    <Sparkles className="h-6 w-6 text-indigo-500" />
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <AlertTitle className="text-foreground font-bold uppercase tracking-[0.25em] text-xs mb-4 opacity-90 font-heading">Engine Audit Insights</AlertTitle>
                                                    <AlertDescription>
                                                        <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-4">
                                                            {result.warnings.map((w, i) => (
                                                                <li key={i} className="flex items-center gap-4 text-[11px] font-semibold text-muted-foreground/70 hover:text-indigo-500 transition-colors group cursor-default">
                                                                    <div className="h-2 w-2 bg-indigo-500/30 rounded-full shrink-0 group-hover:scale-125 transition-transform group-hover:bg-indigo-500" />
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
                                    className="flex flex-col items-center justify-center h-[700px] border-[3px] border-dashed border-border rounded-[1rem] bg-muted/5 backdrop-blur-[4px] p-20 text-center relative overflow-hidden group"
                                >
                                    <div className="absolute inset-0 bg-linear-to-br from-indigo-500/5 via-transparent to-teal-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />
                                    
                                    <div className="grid grid-cols-2 gap-6 mb-16 relative z-10 transition-all duration-700 group-hover:scale-105">
                                        <div className="p-8 bg-background border border-border rounded-[1rem] shadow-xl group-hover:shadow-indigo-500/10 group-hover:-translate-y-2 transition-all duration-500">
                                            <ShieldCheck className="h-10 w-10 text-indigo-500" />
                                            <div className="mt-4 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/40 font-heading">Decision Reliability</div>
                                        </div>
                                        <div className="p-8 bg-background border border-border rounded-[1rem] shadow-xl group-hover:shadow-teal-500/10 group-hover:translate-y-2 transition-all duration-500">
                                            <Target className="h-10 w-10 text-teal-500" />
                                            <div className="mt-4 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/40 font-heading">Objective Accuracy</div>
                                        </div>
                                    </div>

                                    <h3 className="text-4xl font-bold text-foreground mb-6 relative z-10 tracking-tighter uppercase max-w-lg leading-tight font-heading">
                                        Global Model Performance Benchmark
                                    </h3>
                                    <p className="text-muted-foreground/70 max-w-md mb-16 relative z-10 font-medium italic text-lg leading-relaxed">
                                        Comparative calibration audit across decision and objective spaces. Select targets to initiate diagnostic compute.
                                    </p>

                                    <div className="flex items-center gap-4 relative z-10">
                                        <div className="flex items-center gap-3 text-indigo-500 font-bold text-[11px] uppercase tracking-[0.3em] bg-indigo-500/5 px-10 py-5 rounded-[1rem] border border-indigo-500/20 shadow-lg shadow-indigo-500/5 group-hover:scale-110 transition-all duration-500 font-heading">
                                            <Sparkles className="h-5 w-5 animate-pulse" />
                                            Awaiting Multi-Engine Selection
                                        </div>
                                    </div>

                                    <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex items-center gap-6 opacity-20 relative z-10">
                                        <Brain className="h-5 w-5" />
                                        <Scale className="h-5 w-5" />
                                        <Activity className="h-5 w-5" />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </TooltipProvider>
    );
}
