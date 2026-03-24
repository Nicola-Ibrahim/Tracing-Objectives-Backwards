"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { getDatasets } from "@/features/inverse/api";
import { EngineComparisonPanel } from "@/features/evaluation/components/EngineComparisonPanel";
import { PerformanceChart, MetricBarChart } from "@/features/evaluation/components/EvaluationCharts";
import { useDiagnosticsSSE } from "@/features/evaluation/hooks/useDiagnosticsSSE";
import {
    LineChart,
    AlertCircle,
    TrendingUp,
    Loader2,
    Sparkles,
    Trophy,
    Target,
    Activity,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
    TooltipProvider,
} from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export default function EvaluationPage() {
    const { 
        startDiagnostic, 
        isEvaluating, 
        progress, 
        statusMessage, 
        error, 
        result 
    } = useDiagnosticsSSE();

    const { data: datasets = [] } = useQuery({
        queryKey: ["datasets"],
        queryFn: getDatasets,
        select: (data) => data.map((d) => d.name),
    });

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
                            onDiagnose={startDiagnostic}
                            isLoading={isEvaluating}
                        />
                    </div>

                    <div className="lg:col-span-3 space-y-8 min-h-[700px]">
                        <AnimatePresence mode="wait">
                            {error && (
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
                                                <AlertTitle className="text-lg font-bold">Diagnostic Failed</AlertTitle>
                                                <AlertDescription className="text-muted-foreground font-medium">
                                                    {error}
                                                </AlertDescription>
                                            </div>
                                        </div>
                                    </Alert>
                                </motion.div>
                            )}

                            {isEvaluating && (
                                <motion.div
                                    key="loading"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="flex flex-col items-center justify-center min-h-[400px] space-y-6 bg-card/50 rounded-[2rem] border border-border/50 backdrop-blur-sm"
                                >
                                    <div className="relative">
                                        <div className="absolute inset-0 bg-indigo-500/20 blur-3xl rounded-full" />
                                        <Loader2 className="h-16 w-16 animate-spin text-indigo-600 relative z-10" />
                                    </div>
                                    <div className="text-center space-y-4 max-w-md w-full px-8">
                                        <h3 className="text-xl font-bold text-foreground">{statusMessage}</h3>
                                        <Progress value={progress} className="h-2 w-full bg-indigo-100 dark:bg-indigo-950/30" />
                                        <p className="text-sm text-muted-foreground font-medium">
                                            Executing asynchronous diagnostics via Celery. This provides safe, long-running compute without blocking your experience.
                                        </p>
                                    </div>
                                </motion.div>
                            )}

                            {result && !isEvaluating && (
                                <motion.div
                                    key="result"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-8"
                                >
                                    {/* Score Summary Cards */}
                                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                                        {result.engines.map((engine) => {
                                            const metrics = result.objective_space.metrics[engine];
                                            const mace = metrics?.["MACE"];
                                            const isBest = mace === Math.min(...result.engines.map(e => result.objective_space.metrics[e]?.["MACE"] || 999));

                                            return (
                                                <Card key={engine} className={cn(
                                                    "overflow-hidden transition-all duration-300 border-border/50 rounded-[1.5rem]",
                                                    isBest ? "ring-2 ring-indigo-500/50 shadow-xl shadow-indigo-500/10" : "hover:border-indigo-500/30"
                                                )}>
                                                    <CardContent className="p-6">
                                                        <div className="flex justify-between items-start mb-4">
                                                            <div className="space-y-1">
                                                                <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest">{result.capabilities[engine]}</p>
                                                                <h4 className="text-xl font-black text-foreground">{engine}</h4>
                                                            </div>
                                                            {isBest && (
                                                                <div className="p-2 bg-indigo-500/10 rounded-full">
                                                                    <Trophy className="h-5 w-5 text-indigo-600" />
                                                                </div>
                                                            )}
                                                        </div>
                                                        <div className="space-y-1">
                                                            <p className="text-3xl font-black text-foreground tracking-tighter">
                                                                {mace !== undefined ? mace.toFixed(4) : "N/A"}
                                                            </p>
                                                            <p className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Mean Calibration Error</p>
                                                        </div>
                                                    </CardContent>
                                                </Card>
                                            );
                                        })}
                                    </div>

                                    {/* Main Results Tabs */}
                                    <Tabs defaultValue="objective" className="w-full">
                                        <div className="flex items-center justify-between mb-6">
                                            <TabsList className="bg-muted/50 p-1 rounded-[1.2rem] border border-border/50">
                                                <TabsTrigger value="objective" className="rounded-[1rem] px-6 data-[state=active]:bg-background data-[state=active]:shadow-sm font-bold">
                                                    <Target className="w-4 h-4 mr-2" />
                                                    Objective Space
                                                </TabsTrigger>
                                                <TabsTrigger value="decision" className="rounded-[1rem] px-6 data-[state=active]:bg-background data-[state=active]:shadow-sm font-bold">
                                                    <Activity className="w-4 h-4 mr-2" />
                                                    Decision Space
                                                </TabsTrigger>
                                            </TabsList>
                                            <div className="flex gap-2">
                                                <Badge variant="outline" className="rounded-full bg-indigo-50/50 dark:bg-indigo-950/20 text-indigo-600 dark:text-indigo-400 border-indigo-200/50 font-bold px-4 py-1">
                                                    <Sparkles className="w-3.h-3 mr-2" />
                                                    AI Assessment Live
                                                </Badge>
                                            </div>
                                        </div>

                                        <TabsContent value="objective" className="mt-0 space-y-8 outline-none">
                                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                                                <PerformanceChart
                                                    title="Empirical Coverage ECDF"
                                                    description="Ideal calibration follows the diagonal line (y=x). Deviations indicate overconfidence or underconfidence."
                                                    data={result.objective_space.ecdf}
                                                    type="ecdf"
                                                />
                                                <PerformanceChart
                                                    title="Calibration Error Curves"
                                                    description="Absolute deviation from ideal calibration across the entire CDF range."
                                                    data={result.objective_space.calibration_curves}
                                                    type="calibration"
                                                />
                                            </div>
                                            
                                            <div className="p-8 bg-card/30 border border-border/50 rounded-[2rem] backdrop-blur-sm">
                                                <div className="flex items-center gap-3 mb-6">
                                                    <div className="p-2 bg-indigo-500/10 rounded-lg">
                                                        <TrendingUp className="w-5 h-5 text-indigo-600" />
                                                    </div>
                                                    <h3 className="text-xl font-bold tracking-tight">Standardized Performance Metrics</h3>
                                                </div>
                                                <MetricBarChart data={result.objective_space.metrics} />
                                            </div>
                                        </TabsContent>

                                        <TabsContent value="decision" className="mt-0 space-y-8 outline-none">
                                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                                                <PerformanceChart
                                                    key="decision-ecdf"
                                                    title="Decision Variable ECDF"
                                                    description="Calibration of the generated candidates in the inverse decision space."
                                                    data={result.decision_space.ecdf}
                                                    type="ecdf"
                                                />
                                                <PerformanceChart
                                                    key="decision-calibration"
                                                    title="Decision Calibration Curves"
                                                    description="Error curves specifically for decision parameters."
                                                    data={result.decision_space.calibration_curves}
                                                    type="calibration"
                                                />
                                            </div>

                                            <div className="p-8 bg-card/30 border border-border/50 rounded-[2rem] backdrop-blur-sm">
                                                <div className="flex items-center gap-3 mb-6">
                                                    <div className="p-2 bg-violet-500/10 rounded-lg">
                                                        <Activity className="w-5 h-5 text-violet-600" />
                                                    </div>
                                                    <h3 className="text-xl font-bold tracking-tight">Decision Space Calibration Metrics</h3>
                                                </div>
                                                <MetricBarChart data={result.decision_space.metrics} />
                                            </div>
                                        </TabsContent>
                                    </Tabs>

                                    {/* Warnings Section */}
                                    {result.warnings.length > 0 && (
                                        <Card className="border-amber-500/20 bg-amber-500/5 rounded-[1.5rem] overflow-hidden">
                                            <CardHeader className="pb-2">
                                                <div className="flex items-center gap-2 text-amber-600">
                                                    <AlertCircle className="h-5 w-5" />
                                                    <CardTitle className="text-sm font-bold uppercase tracking-widest">Convergence Notices</CardTitle>
                                                </div>
                                            </CardHeader>
                                            <CardContent>
                                                <ul className="space-y-1">
                                                    {result.warnings.map((w, i) => (
                                                        <li key={i} className="text-sm text-amber-800/80 font-medium">• {w}</li>
                                                    ))}
                                                </ul>
                                            </CardContent>
                                        </Card>
                                    )}
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {!result && !isEvaluating && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="flex flex-col items-center justify-center min-h-[500px] border-2 border-dashed border-border/50 rounded-[3rem] bg-muted/20"
                            >
                                <div className="p-6 bg-background rounded-full shadow-xl mb-6 ring-8 ring-muted/5 z-0">
                                    <LineChart className="h-12 w-12 text-muted-foreground/40" />
                                </div>
                                <h3 className="text-2xl font-black text-foreground tracking-tight mb-2 uppercase">Awaiting Diagnosis</h3>
                                <p className="text-muted-foreground font-medium text-center max-w-sm px-6 italic">
                                    Select your dataset and engine candidates to begin the high-fidelity calibration audit.
                                </p>
                            </motion.div>
                        )}
                    </div>
                </div>
            </div>
        </TooltipProvider>
    );
}
