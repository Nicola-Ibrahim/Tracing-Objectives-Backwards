"use client";

import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { getDatasets } from "@/features/inverse/api";
import { diagnoseEngines } from "@/features/evaluation/api";
import { EngineComparisonPanel } from "@/features/evaluation/components/EngineComparisonPanel";
import { PerformanceChart } from "@/features/evaluation/components/Charts";
import { DiagnoseRequest, DiagnoseResponse } from "@/features/evaluation/types";
import { LineChart, AlertCircle, TrendingUp, Info, Table as TableIcon } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function EvaluationPage() {
    const [result, setResult] = useState<DiagnoseResponse | null>(null);

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
        await mutation.mutateAsync(params);
    };

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
                        isLoading={mutation.isPending}
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

                    {result ? (
                        <div className="space-y-6 animate-in fade-in duration-500">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <PerformanceChart
                                    title="ECDF (Calibration)"
                                    description="Empirical Cumulative Distribution Function of residuals."
                                    data={result.ecdf}
                                    xAxisLabel="Normalized Residual"
                                    yAxisLabel="Cumulative Prob."
                                />
                                <PerformanceChart
                                    title="PIT (Probabilistic Calibration)"
                                    description="Probability Integral Transform for uncertainty depth."
                                    data={result.pit}
                                    xAxisLabel="PIT Value"
                                    yAxisLabel="Cumulative Frequency"
                                    showIdeal
                                />
                            </div>

                            <Card className="border-slate-200">
                                <CardHeader className="py-4 border-b">
                                    <CardTitle className="text-sm font-semibold flex items-center gap-2">
                                        <TableIcon className="h-4 w-4 text-slate-400" />
                                        MACE (Mean Absolute Calibration Error)
                                    </CardTitle>
                                </CardHeader>
                                <CardContent className="p-0">
                                    <Table>
                                        <TableHeader className="bg-slate-50/50">
                                            <TableRow>
                                                <TableHead>Engine Identifier</TableHead>
                                                <TableHead className="text-right">MACE Score</TableHead>
                                                <TableHead className="text-center">Performance Status</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {Object.entries(result.mace).map(([engine, score]) => (
                                                <TableRow key={engine}>
                                                    <TableCell className="font-medium text-slate-700">{engine}</TableCell>
                                                    <TableCell className="text-right font-mono text-sm">{score.toFixed(4)}</TableCell>
                                                    <TableCell className="text-center">
                                                        <Badge variant={score < 0.05 ? "default" : "secondary"} className={score < 0.05 ? "bg-emerald-100 text-emerald-700 hover:bg-emerald-100 border-emerald-200" : ""}>
                                                            {score < 0.05 ? "Highly Calibrated" : "Minor Drift"}
                                                        </Badge>
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </CardContent>
                            </Card>

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
