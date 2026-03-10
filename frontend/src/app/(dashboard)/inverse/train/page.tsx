"use client";

import { useQuery, useMutation } from "@tanstack/react-query";
import { getDatasets, trainEngine } from "@/features/inverse/api";
import { TrainEngineForm } from "@/features/inverse/components/TrainEngineForm";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
} from "@/components/ui/card";
import { TrainEngineRequest } from "@/features/inverse/types";
import { useState } from "react";
import { CheckCircle2, AlertCircle } from "lucide-react";
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
        <div className="space-y-6 max-w-7xl mx-auto">
            <div className="flex flex-col gap-1">
                <h1 className="text-3xl font-bold tracking-tight text-slate-900 font-sans">
                    Inverse Engine Construction
                </h1>
                <p className="text-slate-500 font-medium">
                    Train high-fidelity inverse models (GBPI, MDN).
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
                <Card className="border-slate-200 shadow-sm overflow-hidden md:col-span-1">
                    <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                        <CardTitle className="text-lg font-semibold text-slate-800">
                            Configuration
                        </CardTitle>
                        <CardDescription className="text-slate-500 text-sm">
                            Define solver type and transformation pipeline.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="p-8">
                        <TrainEngineForm
                            datasets={datasetNames}
                            isLoading={mutation.isPending || datasetsLoading}
                            onSubmit={async (params) => { mutation.mutate(params); }}
                        />
                    </CardContent>
                </Card>

                <Card className="border-slate-200 shadow-sm overflow-hidden md:col-span-1">
                    <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                        <CardTitle className="text-lg font-semibold text-slate-800">
                            Training Status & Results
                        </CardTitle>
                        <CardDescription className="text-slate-500 text-sm">
                            Real-time telemetry and summary.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="p-8">
                        {!lastResult ? (
                            <div className="flex flex-col items-center justify-center min-h-[300px] border-2 border-dashed border-slate-200 rounded-xl bg-slate-50/30">
                                <span className="text-slate-400 font-medium italic">
                                    No training job initiated.
                                </span>
                            </div>
                        ) : lastResult.success ? (
                            <div className="space-y-4">
                                <Alert className="bg-emerald-50 border-emerald-200 text-emerald-800">
                                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                                    <AlertTitle>Success</AlertTitle>
                                    <AlertDescription>
                                        Engine trained successfully with Version {lastResult.data.engine_version}.
                                    </AlertDescription>
                                </Alert>
                                <div className="grid grid-cols-2 gap-4 text-sm mt-6">
                                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                                        <p className="text-slate-500 mb-1">Duration</p>
                                        <p className="font-bold">{lastResult.data.duration_seconds.toFixed(2)}s</p>
                                    </div>
                                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                                        <p className="text-slate-500 mb-1">Samples</p>
                                        <p className="font-bold">{lastResult.data.n_train_samples}</p>
                                    </div>
                                </div>

                                {lastResult.data.training_history && Object.keys(lastResult.data.training_history).length > 0 && (
                                    <div className="mt-6">
                                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-tight mb-3">Solver History & Artifacts</h3>
                                        <div className="space-y-2">
                                            {Object.entries(lastResult.data.training_history).map(([key, value]: [string, any]) => (
                                                <div key={key} className="flex justify-between items-center p-2 bg-white border border-slate-100 rounded-md text-xs">
                                                    <span className="text-slate-500 capitalize">{key.replace("_", " ")}</span>
                                                    <span className="font-mono font-bold text-indigo-600 text-right max-w-[60%] truncate">
                                                        {Array.isArray(value) 
                                                            ? `[${value.length} epochs / steps]` 
                                                            : typeof value === "number" ? value.toFixed(4) : String(value)}
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <Alert variant="destructive">
                                <AlertCircle className="h-4 w-4" />
                                <AlertTitle>Error</AlertTitle>
                                <AlertDescription>{lastResult.error}</AlertDescription>
                            </Alert>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
