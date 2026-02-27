"use client";

import React, { useState } from "react";
import { useDataset } from "@/components/DatasetContext";
import ChartJSWrapper, { ChartDataset, ChartDataPoint } from "@/components/ChartJSWrapper";
import DatasetSelector from "@/components/DatasetSelector";
import { Card, Button, Input } from "@/components/ui";
import { useToast } from "@/components/ui/ToastContext";
import { generateCandidates, trainDataset } from "@/lib/apiClient";
import { Dataset, GenerationRequest, GenerationResponse, Vector } from "@/types/api";

export default function GeneratorContainer() {
    const {
        datasets,
        selectedDataset,
        baselineData,
        isLoading,
        selectDataset,
        refreshDatasetData,
        ranges
    } = useDataset();

    const { showToast } = useToast();

    // Generation State
    const [targetObj1, setTargetObj1] = useState<number>(0);
    const [targetObj2, setTargetObj2] = useState<number>(0);
    const [nSamples, setNSamples] = useState<number>(50);
    const [trustRadius, setTrustRadius] = useState<number>(0.05);
    const [isGenerating, setIsGenerating] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [result, setResult] = useState<GenerationResponse | null>(null);

    // Clear results when dataset changes
    React.useEffect(() => {
        setResult(null);
    }, [selectedDataset]);

    const handleTrain = async () => {
        if (!selectedDataset) return;
        setIsTraining(true);
        try {
            await trainDataset(selectedDataset);
            showToast("Model trained successfully. You can now generate candidates.", "success");
            // Refresh dataset info to update is_trained status
            await refreshDatasetData();
        } catch (err: any) {
            showToast(err.message, "error");
        } finally {
            setIsTraining(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsGenerating(true);
        setResult(null);

        const request: GenerationRequest = {
            dataset_name: selectedDataset,
            target_objective: [targetObj1, targetObj2],
            n_samples: nSamples,
            trust_radius: trustRadius,
        };

        try {
            const data = await generateCandidates(request);
            setResult(data);
            showToast("Candidates generated successfully", "success");
        } catch (err: any) {
            showToast(err.message, "error");
        } finally {
            setIsGenerating(false);
        }
    };

    // Prepare Objective Space Plot Data
    const objectiveDatasets: ChartDataset[] = [];
    if (baselineData) {
        const baselinePoints: ChartDataPoint[] = [];
        const paretoPoints: ChartDataPoint[] = [];

        baselineData.y.forEach((o: Vector, i: number) => {
            const pt = { x: o[0], y: o[1] };
            if (baselineData.is_pareto?.[i]) {
                paretoPoints.push(pt);
            } else {
                baselinePoints.push(pt);
            }
        });

        objectiveDatasets.push({
            label: "Baseline Data",
            data: baselinePoints,
            backgroundColor: "rgba(203, 213, 225, 0.4)",
        });

        if (paretoPoints.length > 0) {
            objectiveDatasets.push({
                label: "Pareto Front",
                data: paretoPoints,
                backgroundColor: "#10b981",
                pointRadius: 6,
            });
        }
    }

    if (result) {
        if (result.anchor_indices && baselineData) {
            const anchorsY = result.anchor_indices.map((idx) => baselineData.y[idx]);
            objectiveDatasets.push({
                label: "Context Anchors",
                data: anchorsY.map((v: Vector) => ({ x: v[0], y: v[1] })),
                backgroundColor: "#f59e0b", // Amber-500
                pointRadius: 10,
                pointHoverRadius: 12,
            });
        }

        objectiveDatasets.push({
            label: "Generated Candidates",
            data: result.candidate_objectives.map((o: Vector) => ({ x: o[0], y: o[1] })),
            backgroundColor: "#4f46e5", // Indigo-600
            pointRadius: 8,
            pointHoverRadius: 10,
        });

        objectiveDatasets.push({
            label: "Target Objective",
            data: [{ x: result.target_objective[0], y: result.target_objective[1] }],
            backgroundColor: "#ef4444",
            pointRadius: 12,
            pointHoverRadius: 14,
            pointStyle: 'rectRot',
        });

        // Highlight Winner Candidate in Gold
        objectiveDatasets.push({
            label: "Winner Candidate",
            data: [{ x: result.winner_point[0], y: result.winner_point[1] }],
            backgroundColor: "#fbbf24", // Amber-400 (Gold)
            pointRadius: 14,
            pointHoverRadius: 16,
            pointStyle: 'star',
            borderColor: "#d97706",
            borderWidth: 2,
        });

        // Vector Line: Target -> Winner
        objectiveDatasets.push({
            label: "Optimal Vector",
            data: [
                { x: result.target_objective[0], y: result.target_objective[1] },
                { x: result.winner_point[0], y: result.winner_point[1] }
            ],
            backgroundColor: "transparent",
            borderColor: "#ef4444", // Red matching target
            showLine: true,
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
        });
    }

    // Prepare Decision Space Plot Data
    const decisionDatasets: ChartDataset[] = [];
    if (baselineData) {
        const baselinePoints: ChartDataPoint[] = [];
        const paretoPoints: ChartDataPoint[] = [];

        baselineData.X.forEach((x: Vector, i: number) => {
            const pt = { x: x[0], y: x[1] };
            if (baselineData.is_pareto?.[i]) {
                paretoPoints.push(pt);
            } else {
                baselinePoints.push(pt);
            }
        });

        decisionDatasets.push({
            label: "Baseline Designs",
            data: baselinePoints,
            backgroundColor: "rgba(148, 163, 184, 0.4)",
        });

        if (paretoPoints.length > 0) {
            decisionDatasets.push({
                label: "Pareto Optimal Decisions",
                data: paretoPoints,
                backgroundColor: "#059669",
                pointRadius: 6,
            });
        }
    }

    if (result) {
        if (result.anchor_indices && baselineData) {
            const anchorsX = result.anchor_indices.map((idx) => baselineData.X[idx]);
            if (anchorsX.length >= 3) {
                const closedAnchors = [...anchorsX, anchorsX[0]];
                decisionDatasets.push({
                    label: "Support Simplex",
                    data: closedAnchors.map((v: Vector) => ({ x: v[0], y: v[1] })),
                    backgroundColor: "transparent",
                    borderColor: "#f59e0b",
                    showLine: true,
                    borderWidth: 2,
                    pointRadius: 0,
                });
            }
            decisionDatasets.push({
                label: "Context Anchors",
                data: anchorsX.map((v: Vector) => ({ x: v[0], y: v[1] })),
                backgroundColor: "#f59e0b",
                pointRadius: 10,
            });
        }
        decisionDatasets.push({
            label: "Candidate Designs",
            data: result.candidate_decisions.map((x: Vector) => ({ x: x[0], y: x[1] })),
            backgroundColor: "#4f46e5", // Indigo-600
            pointRadius: 8,
            pointHoverRadius: 10,
        });

        // Highlight Winner Decision in Decision Space
        decisionDatasets.push({
            label: "Winner Design",
            data: [{ x: result.winner_decision[0], y: result.winner_decision[1] }],
            backgroundColor: "#fbbf24",
            pointRadius: 12,
            pointStyle: 'star',
            borderColor: "#d97706",
            borderWidth: 2,
        });
    }

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight mb-1">Candidate Generator</h1>
                    <p className="text-slate-500 text-sm font-medium">Define targets and generate optimized designs.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div className="lg:col-span-3 space-y-6">
                    <Card title="Source Selection">
                        <DatasetSelector
                            datasets={datasets}
                            selectedDataset={selectedDataset}
                            onSelect={selectDataset}
                            isLoading={isLoading}
                            className="w-full"
                        />
                    </Card>

                    <Card title="Optimization Targets">
                        {baselineData && !baselineData.is_trained ? (
                            <div className="p-5 bg-indigo-50/50 backdrop-blur-sm rounded-2xl border border-indigo-100 flex flex-col items-center text-center space-y-4">
                                <div className="p-3 bg-white rounded-xl shadow-sm">
                                    <div className="w-10 h-10 text-indigo-500">
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
                                        </svg>
                                    </div>
                                </div>
                                <div className="space-y-1">
                                    <h3 className="text-sm font-bold text-slate-800">Model Not Trained</h3>
                                    <p className="text-[11px] text-slate-500 font-medium px-2">
                                        This dataset requires surrogate model fitting before candidate generation.
                                    </p>
                                </div>
                                <Button
                                    onClick={handleTrain}
                                    className="w-full bg-indigo-600 hover:bg-indigo-700 shadow-lg shadow-indigo-100 py-5"
                                    disabled={isTraining}
                                >
                                    {isTraining ? (
                                        <>
                                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                                            Fitting Surrogates...
                                        </>
                                    ) : "Initialize Training"}
                                </Button>
                            </div>
                        ) : (
                            <form onSubmit={handleSubmit} className="space-y-6">
                                <div className="space-y-4">
                                    <label className="text-sm font-semibold text-slate-700 ml-1">
                                        Goal Coordinates
                                    </label>
                                    <div className="grid grid-cols-2 gap-4">
                                        <Input
                                            label="Objective 1"
                                            type="number"
                                            step="0.01"
                                            value={targetObj1}
                                            onChange={(e) => setTargetObj1(parseFloat(e.target.value))}
                                        />
                                        <Input
                                            label="Objective 2"
                                            type="number"
                                            step="0.01"
                                            value={targetObj2}
                                            onChange={(e) => setTargetObj2(parseFloat(e.target.value))}
                                        />
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <Input
                                        label="Batch Size"
                                        type="number"
                                        value={nSamples}
                                        onChange={(e) => setNSamples(parseInt(e.target.value))}
                                    />
                                    <Input
                                        label="Search Radius"
                                        type="number"
                                        step="0.01"
                                        value={trustRadius}
                                        onChange={(e) => setTrustRadius(parseFloat(e.target.value))}
                                    />
                                </div>

                                <Button
                                    type="submit"
                                    className="w-full py-6 text-lg shadow-xl shadow-indigo-200"
                                    disabled={isGenerating || !selectedDataset}
                                >
                                    {isGenerating ? (
                                        <>
                                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                                            Optimizing...
                                        </>
                                    ) : "Start Optimization"}
                                </Button>
                            </form>
                        )}
                    </Card>

                    {result && (
                        <Card title="Optimization Insight">
                            <div className="space-y-4">
                                <div className="p-4 bg-amber-50 rounded-2xl border border-amber-100 space-y-2">
                                    <div className="flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                                        <span className="text-xs font-bold text-amber-900 uppercase tracking-widest">Optimal Match Found</span>
                                    </div>
                                    <div className="space-y-1">
                                        <p className="text-lg font-black text-slate-900 leading-tight">
                                            Winner: [{result.winner_point[0].toFixed(3)}, {result.winner_point[1].toFixed(3)}]
                                        </p>
                                        <p className="text-[10px] text-amber-700 font-bold uppercase tracking-tight">
                                            Error: {result.residual_errors[result.winner_index].toFixed(5)}
                                        </p>
                                    </div>
                                </div>

                                <div className="flex justify-between items-center text-sm p-3 bg-slate-50 rounded-xl border border-slate-100">
                                    <span className="text-slate-500 font-medium">Trajectory</span>
                                    <span className={`px-2.5 py-1 rounded-lg text-xs font-bold uppercase tracking-wider ${result.pathway === "coherent"
                                        ? "bg-green-100 text-green-700"
                                        : "bg-amber-100 text-amber-700"
                                        }`}>
                                        {result.pathway}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center text-sm p-3 bg-slate-50 rounded-xl border border-slate-100">
                                    <span className="text-slate-500 font-medium">Model Validity</span>
                                    <span className={`font-bold ${result.is_inside_mesh ? "text-green-600" : "text-amber-600"}`}>
                                        {result.is_inside_mesh ? "Verified (Inside Mesh)" : "Extrapolated (Outside Mesh)"}
                                    </span>
                                </div>
                            </div>
                        </Card>
                    )}
                </div>

                <div className="lg:col-span-9 space-y-6">
                    <div className="grid grid-cols-1 2xl:grid-cols-2 gap-8">
                        <div className="space-y-3">
                            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 px-1">Candidate Allocation (X)</h2>
                            <ChartJSWrapper
                                title="Design Parameters"
                                datasets={decisionDatasets}
                                xAxisTitle="Dimension 1"
                                yAxisTitle="Dimension 2"
                                xRange={ranges.decX || undefined}
                                yRange={ranges.decY || undefined}
                            />
                        </div>
                        <div className="space-y-3">
                            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 px-1">Performance Results (y)</h2>
                            <ChartJSWrapper
                                title="Success Metrics"
                                datasets={objectiveDatasets}
                                xAxisTitle="Objective 1"
                                yAxisTitle="Objective 2"
                                xRange={ranges.objX || undefined}
                                yRange={ranges.objY || undefined}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
