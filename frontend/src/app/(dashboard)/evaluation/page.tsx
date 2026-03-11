"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { getDatasets } from "@/features/inverse/api";
import { diagnoseEngines } from "@/features/evaluation/api";
import { EngineComparisonPanel } from "@/features/evaluation/components/EngineComparisonPanel";
import { PerformanceChart, MetricBarChart } from "@/features/evaluation/components/Charts";
import { DiagnoseRequest, DiagnoseResponse } from "@/features/evaluation/types";
import { LineChart, AlertCircle, TrendingUp, Info, Table as TableIcon, Loader2 } from "lucide-react";
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
        <div className="space-y-6 max-w-7xl mx-auto pb-10">
            <div className="flex flex-col gap-1">
                <h1 className="text-3xl font-bold tracking-tight text-slate-900 font-sans leading-8 flex items-center gap-2">
                    <LineChart className="h-7 w-7 text-indigo-600" />
                    Model Evaluation
                </h1>
                <p className="text-slate-500 font-medium">Benchmark and compare multiple inverse mapping engines.</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-1">
                    <EngineComparisonPanel
                        datasets={datasets}
                        onDiagnose={handleDiagnose}
                        isLoading={isPending}
                    />
                </div>

                <div className="lg:col-span-3 space-y-6">
                    {mutation.isError && (
                        <Alert variant="destructive">
                            <AlertCircle className="h-4 w-4" />
                            <AlertTitle>Benchmark Failed</AlertTitle>
                            <AlertDescription>
                                {(mutation.error as any)?.response?.data?.detail || "An unexpected error occurred during model diagnosis."}
                            </AlertDescription>
                        </Alert>
                    )}

                    {isPending ? (
                        <div className="space-y-8 animate-pulse">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="h-[380px] bg-slate-100 rounded-xl border border-slate-200" />
                                <div className="h-[380px] bg-slate-100 rounded-xl border border-slate-200" />
                            </div>
                            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                                <div className="lg:col-span-3 h-[380px] bg-slate-100 rounded-xl border border-slate-200" />
                                <div className="lg:col-span-2 h-[380px] bg-slate-100 rounded-xl border border-slate-200" />
                            </div>
                            <div className="flex items-center justify-center p-10">
                                <div className="flex flex-col items-center gap-3">
                                    <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
                                    <p className="text-sm font-medium text-slate-500 italic">Processing high-dimensional diagnostic data...</p>
                                </div>
                            </div>
                        </div>
                    ) : result ? (
                        <div className="space-y-8 animate-in slide-in-from-bottom-2 fade-in duration-700">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <PerformanceChart
                                    title="ECDF (Calibration)"
                                    description="Empirical Cumulative Distribution Function of residuals. Lower area under curve (closer to top-left) indicates better accuracy."
                                    data={result.ecdf}
                                    xAxisLabel="Normalized Residual"
                                    yAxisLabel="Cumulative Prob."
                                />
                                <PerformanceChart
                                    title="PIT (Probabilistic Calibration)"
                                    description="Calibration of uncertainty. A perfectly calibrated model follows the diagonal ideal line."
                                    data={result.pit}
                                    xAxisLabel="PIT Value"
                                    yAxisLabel="Cumulative Frequency"
                                    showIdeal
                                />
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                                <div className="lg:col-span-3">
                                    <MetricBarChart
                                        title="MACE Comparison"
                                        description="Mean Absolute Calibration Error across models. Lower values represent better matching with ground truth distributions."
                                        data={result.mace}
                                        yAxisLabel="MACE Score"
                                    />
                                </div>
                                <div className="lg:col-span-2">
                                    <Card className="border-slate-200 bg-white shadow-sm h-full overflow-hidden">
                                        <CardHeader className="py-4 bg-slate-50 border-b border-slate-100">
                                            <CardTitle className="text-sm font-bold flex items-center gap-2 text-slate-800">
                                                <TableIcon className="h-4 w-4 text-indigo-500" />
                                                Engine Leaderboard
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent className="p-0">
                                            <Table>
                                                <TableHeader className="bg-slate-50/20">
                                                    <TableRow className="hover:bg-transparent border-slate-100">
                                                        <TableHead className="font-semibold text-slate-600 text-[11px] uppercase tracking-wider">Engine</TableHead>
                                                        <TableHead className="text-right font-semibold text-slate-600 text-[11px] uppercase tracking-wider">MACE</TableHead>
                                                    </TableRow>
                                                </TableHeader>
                                                <TableBody>
                                                    {Object.entries(result.mace)
                                                        .sort(([, a], [, b]) => a - b)
                                                        .map(([engine, score]) => (
                                                            <TableRow key={engine} className="border-slate-100/60 hover:bg-slate-50/30 transition-colors">
                                                                <TableCell className="py-3">
                                                                    <div className="flex flex-col">
                                                                        <span className="font-semibold text-slate-700 text-sm">{engine}</span>
                                                                        <Badge variant={score < 0.05 ? "default" : "secondary"} className={`w-fit text-[9px] h-4 mt-1 px-1.5 ${
                                                                            score < 0.05 ? "bg-emerald-50 text-emerald-600 border-emerald-100" : "bg-slate-100 text-slate-500 border-slate-200"
                                                                        }`}>
                                                                            {score < 0.05 ? "High Precision" : "Standard"}
                                                                        </Badge>
                                                                    </div>
                                                                </TableCell>
                                                                <TableCell className="text-right py-3">
                                                                    <span className="font-mono text-sm font-bold text-indigo-600">
                                                                        {score.toFixed(4)}
                                                                    </span>
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
                                <Alert className="bg-amber-50 border-amber-200 text-amber-800">
                                    <Info className="h-4 w-4 text-amber-600" />
                                    <AlertTitle className="text-amber-900">Diagnosis Notes</AlertTitle>
                                    <AlertDescription>
                                        <ul className="list-disc pl-4 mt-2 space-y-1">
                                            {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
                                        </ul>
                                    </AlertDescription>
                                </Alert>
                            )}
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center min-h-[500px] border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/10 p-10 text-center">
                            <div className="bg-white p-4 rounded-full shadow-sm mb-4">
                                <TrendingUp className="h-8 w-8 text-slate-300" />
                            </div>
                            <h3 className="text-lg font-semibold text-slate-900 mb-2">Comparative Benchmark</h3>
                            <p className="text-slate-500 max-w-sm mb-6">Select multiple trained engines on the left to compare their calibration and predictive performance.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
