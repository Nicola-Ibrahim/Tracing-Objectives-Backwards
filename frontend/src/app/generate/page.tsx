"use client";

import React, { useEffect, useState } from "react";
import { fetchDatasets, generateCandidates, fetchDatasetData } from "@/lib/apiClient";
import { GenerationRequest, GenerationResponse, DatasetDetailResponse, Vector } from "@/types/api";
import ChartJSWrapper, { ChartDataset, ChartDataPoint } from "@/components/ChartJSWrapper";
import DatasetSelector from "@/components/DatasetSelector";
import DatasetGenerator from "@/components/DatasetGenerator";
import Tabs from "@/components/ui/Tabs";
import { Card, Button, Input } from "@/components/ui";
import { useToast } from "@/components/ui/ToastContext";

export default function GeneratePage() {

    // Navigation State
    const [activeTab, setActiveTab] = useState("data-hub");

    // Dataset State
    const [datasets, setDatasets] = useState<string[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [baselineData, setBaselineData] = useState<DatasetDetailResponse | null>(null);
    const [coordinateLoading, setCoordinateLoading] = useState(false);

    // Generation State
    const [targetObj1, setTargetObj1] = useState<number>(0);
    const [targetObj2, setTargetObj2] = useState<number>(0);
    const [nSamples, setNSamples] = useState<number>(50);
    const [trustRadius, setTrustRadius] = useState<number>(0.05);
    const [isGenerating, setIsGenerating] = useState(false);
    const [result, setResult] = useState<GenerationResponse | null>(null);

    // Axis Range State
    const [objXRange, setObjXRange] = useState<[number, number] | null>(null);
    const [objYRange, setObjYRange] = useState<[number, number] | null>(null);
    const [decXRange, setDecXRange] = useState<[number, number] | null>(null);
    const [decYRange, setDecYRange] = useState<[number, number] | null>(null);

    const { showToast } = useToast();

    const refreshDatasets = async (selectName?: string) => {
        try {
            const data = await fetchDatasets();
            setDatasets(data);
            if (selectName) {
                setSelectedDataset(selectName);
                setActiveTab("data-hub");
            } else if (data.length > 0 && !selectedDataset) {
                setSelectedDataset(data[0]);
            }
        } catch (err: any) {
            showToast(err.message, "error");
        }
    };

    useEffect(() => {
        refreshDatasets();
    }, [showToast]);

    useEffect(() => {
        if (!selectedDataset) {
            setBaselineData(null);
            setResult(null);
            return;
        }

        setCoordinateLoading(true);
        fetchDatasetData(selectedDataset)
            .then((data) => {
                setBaselineData(data);
                // Reset result when dataset changes to avoid mismatched overlays
                setResult(null);

                const padding = 0.05;

                if (data.bounds.obj_0 && data.bounds.obj_1) {
                    const o0 = data.bounds.obj_0;
                    const o1 = data.bounds.obj_1;
                    const span0 = o0[1] - o0[0];
                    const span1 = o1[1] - o1[0];
                    setObjXRange([o0[0] - (span0 || 1) * padding, o0[1] + (span0 || 1) * padding]);
                    setObjYRange([o1[0] - (span1 || 1) * padding, o1[1] + (span1 || 1) * padding]);
                }

                if (data.X.length > 0) {
                    const x0 = data.X.map((v: Vector) => v[0]).filter(v => typeof v === 'number' && !isNaN(v));
                    const x1 = data.X.map((v: Vector) => v[1]).filter(v => typeof v === 'number' && !isNaN(v));

                    if (x0.length > 0 && x1.length > 0) {
                        const minX0 = Math.min(...x0);
                        const maxX0 = Math.max(...x0);
                        const minX1 = Math.min(...x1);
                        const maxX1 = Math.max(...x1);
                        const spanX0 = maxX0 - minX0;
                        const spanX1 = maxX1 - minX1;
                        setDecXRange([minX0 - (spanX0 || 1) * padding, maxX0 + (spanX0 || 1) * padding]);
                        setDecYRange([minX1 - (spanX1 || 1) * padding, maxX1 + (spanX1 || 1) * padding]);
                    }
                }
            })
            .catch((err) => showToast(err.message, "error"))
            .finally(() => setCoordinateLoading(false));
    }, [selectedDataset, showToast]);

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
            backgroundColor: "#4f46e5", // Indigo-600 (consistent with objective space)
            pointRadius: 8,
            pointHoverRadius: 10,
        });
    }

    const tabs = [
        { id: "data-hub", label: "Data Hub" },
        { id: "generator", label: "Candidate Generator" },
    ];

    return (
        <div className="space-y-8 max-w-full mx-auto px-4 py-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight mb-1">Experiment Hub</h1>
                    <p className="text-slate-500 text-sm font-medium">Trace objectives and optimize designs.</p>
                </div>
                <div className="glass-panel p-1.5 flex items-center bg-white/30">
                    <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
                </div>
            </div>

            {activeTab === "data-hub" ? (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                        {/* Control Side Panel */}
                        <div className="lg:col-span-1 space-y-6">
                            <Card title="Source Selection">
                                <DatasetSelector
                                    datasets={datasets}
                                    selectedDataset={selectedDataset}
                                    onSelect={setSelectedDataset}
                                    isLoading={coordinateLoading}
                                    className="w-full"
                                />
                            </Card>

                            <Card>
                                <DatasetGenerator onSuccess={(name) => refreshDatasets(name)} />
                            </Card>
                        </div>

                        {/* Exploration Plots Section */}
                        <div className="lg:col-span-3">
                            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                                <div className="space-y-3">
                                    <div className="flex items-center justify-between px-1">
                                        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400">Decision Space (X)</h2>
                                        {baselineData && (
                                            <span className="text-[11px] font-bold text-indigo-500 bg-indigo-50 px-2 py-0.5 rounded-full">
                                                {baselineData.X.length} SAMPLES
                                            </span>
                                        )}
                                    </div>
                                    <ChartJSWrapper
                                        title="Parameter Space"
                                        datasets={decisionDatasets}
                                        xAxisTitle="Dimension 1"
                                        yAxisTitle="Dimension 2"
                                        xRange={decXRange || undefined}
                                        yRange={decYRange || undefined}
                                    />
                                </div>

                                <div className="space-y-3">
                                    <div className="flex items-center justify-between px-1">
                                        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400">Objective Space (y)</h2>
                                        {baselineData && (
                                            <span className="text-[11px] font-bold text-emerald-500 bg-emerald-50 px-2 py-0.5 rounded-full uppercase">
                                                Pareto: {baselineData.is_pareto?.filter(Boolean).length}
                                            </span>
                                        )}
                                    </div>
                                    <ChartJSWrapper
                                        title="Performance Space"
                                        datasets={objectiveDatasets}
                                        xAxisTitle="Objective 1"
                                        yAxisTitle="Objective 2"
                                        xRange={objXRange || undefined}
                                        yRange={objYRange || undefined}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="lg:col-span-1 space-y-6">
                        <Card title="Source Selection">
                            <DatasetSelector
                                datasets={datasets}
                                selectedDataset={selectedDataset}
                                onSelect={setSelectedDataset}
                                isLoading={coordinateLoading}
                                className="w-full"
                            />
                        </Card>

                        <Card title="Optimization Targets">
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
                        </Card>

                        {result && (
                            <Card title="Optimization Insight">
                                <div className="space-y-4">
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

                    <div className="lg:col-span-2 space-y-6">
                        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                            <div className="space-y-3">
                                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 px-1">Candidate Allocation (X)</h2>
                                <ChartJSWrapper
                                    title="Design Parameters"
                                    datasets={decisionDatasets}
                                    xAxisTitle="Dimension 1"
                                    yAxisTitle="Dimension 2"
                                    xRange={decXRange || undefined}
                                    yRange={decYRange || undefined}
                                />
                            </div>
                            <div className="space-y-3">
                                <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 px-1">Performance Results (y)</h2>
                                <ChartJSWrapper
                                    title="Success Metrics"
                                    datasets={objectiveDatasets}
                                    xAxisTitle="Objective 1"
                                    yAxisTitle="Objective 2"
                                    xRange={objXRange || undefined}
                                    yRange={objYRange || undefined}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
