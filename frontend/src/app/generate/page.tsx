"use client";

import React, { useEffect, useState } from "react";
import { fetchDatasets, generateCandidates, fetchDatasetData } from "@/lib/apiClient";
import { GenerationRequest, GenerationResponse, DatasetDetailResponse, Vector } from "@/types/api";
import PlotlyWrapper from "@/components/PlotlyWrapper";
import DatasetSelector from "@/components/DatasetSelector";
import { Card, Button, Input } from "@/components/ui";
import { useToast } from "@/components/ui/ToastContext";

export default function GeneratePage() {
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

    // Load available datasets on mount
    useEffect(() => {
        fetchDatasets()
            .then((data) => {
                setDatasets(data);
                if (data.length > 0 && !selectedDataset) {
                    setSelectedDataset(data[0]);
                }
            })
            .catch((err) => showToast(err.message, "error"));
    }, [showToast, selectedDataset]);

    // Load coordinates when dataset selection changes
    useEffect(() => {
        if (!selectedDataset) return;

        setCoordinateLoading(true);
        fetchDatasetData(selectedDataset)
            .then((data) => {
                setBaselineData(data);

                // Calculate ranges with 5% padding
                const padding = 0.05;

                // Objective bounds
                if (data.bounds.obj_0 && data.bounds.obj_1) {
                    const o0 = data.bounds.obj_0;
                    const o1 = data.bounds.obj_1;
                    const span0 = o0[1] - o0[0];
                    const span1 = o1[1] - o1[0];
                    // Fallback to [0,1] if span is zero
                    setObjXRange([o0[0] - (span0 || 1) * padding, o0[1] + (span0 || 1) * padding]);
                    setObjYRange([o1[0] - (span1 || 1) * padding, o1[1] + (span1 || 1) * padding]);
                }

                // Decision bounds
                if (data.X.length > 0) {
                    const x0 = data.X.map((v: Vector) => v[0]);
                    const x1 = data.X.map((v: Vector) => v[1]);
                    const minX0 = Math.min(...x0);
                    const maxX0 = Math.max(...x0);
                    const minX1 = Math.min(...x1);
                    const maxX1 = Math.max(...x1);
                    const spanX0 = maxX0 - minX0;
                    const spanX1 = maxX1 - minX1;
                    setDecXRange([minX0 - (spanX0 || 1) * padding, maxX0 + (spanX0 || 1) * padding]);
                    setDecYRange([minX1 - (spanX1 || 1) * padding, maxX1 + (spanX1 || 1) * padding]);
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
    const objectivePlotData: any[] = [];
    if (baselineData) {
        objectivePlotData.push({
            x: baselineData.y.map((o: Vector) => o[0]),
            y: baselineData.y.map((o: Vector) => o[1]),
            mode: "markers",
            type: "scattergl",
            name: "Baseline Data",
            marker: { color: "#cbd5e1", size: 4, opacity: 0.5 },
        });
    }
    if (result) {
        objectivePlotData.push({
            x: result.candidate_objectives.map((o: Vector) => o[0]),
            y: result.candidate_objectives.map((o: Vector) => o[1]),
            mode: "markers",
            type: "scatter",
            name: "Generated Candidates",
            marker: {
                color: "#6366f1",
                size: 10,
                opacity: 0.9,
                line: { width: 1.5, color: "white" }
            },
        });
        objectivePlotData.push({
            x: [result.target_objective[0]],
            y: [result.target_objective[1]],
            mode: "markers",
            type: "scatter",
            name: "Target Objective",
            marker: {
                color: "#ef4444",
                size: 16,
                symbol: "cross",
                line: { width: 2, color: "white" },
            },
        });
    }

    // Prepare Decision Space Plot Data
    const decisionPlotData: any[] = [];
    if (baselineData) {
        decisionPlotData.push({
            x: baselineData.X.map((x: Vector) => x[0]),
            y: baselineData.X.map((x: Vector) => x[1]),
            mode: "markers",
            type: "scattergl",
            name: "Baseline Designs",
            marker: { color: "#94a3b8", size: 4, opacity: 0.5 },
        });
    }
    if (result) {
        // Find anchors if available
        if (result.anchor_indices && baselineData) {
            const anchorsX = result.anchor_indices.map((idx) => baselineData.X[idx]);

            // Add dashed lines between anchors (simplex edges)
            if (anchorsX.length >= 3) {
                const closedAnchors = [...anchorsX, anchorsX[0]];
                decisionPlotData.push({
                    x: closedAnchors.map((v: Vector) => v[0]),
                    y: closedAnchors.map((v: Vector) => v[1]),
                    mode: "lines",
                    type: "scatter",
                    name: "Support Simplex",
                    line: { dash: "dash", width: 2, color: "#f59e0b" },
                    hoverinfo: "none",
                    showlegend: false,
                });
            }

            decisionPlotData.push({
                x: anchorsX.map((v: Vector) => v[0]),
                y: anchorsX.map((v: Vector) => v[1]),
                mode: "markers",
                type: "scatter",
                name: "Context Anchors",
                marker: {
                    color: "#f59e0b",
                    size: 14,
                    symbol: "diamond-open",
                    line: { width: 2.5 }
                },
            });
        }

        decisionPlotData.push({
            x: result.candidate_decisions.map((x: Vector) => x[0]),
            y: result.candidate_decisions.map((x: Vector) => x[1]),
            mode: "markers",
            type: "scatter",
            name: "Candidate Designs",
            marker: {
                color: "#8b5cf6",
                size: 10,
                opacity: 0.9,
                line: { width: 1.5, color: "white" }
            },
        });
    }

    return (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
            {/* Sidebar Controls */}
            <div className="lg:col-span-1 space-y-6">
                <Card title="Generation Parameters">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <DatasetSelector
                            datasets={datasets}
                            selectedDataset={selectedDataset}
                            onSelect={setSelectedDataset}
                            isLoading={coordinateLoading}
                        />

                        <div className="space-y-4 pt-4 border-t border-slate-100">
                            <label className="text-sm font-semibold text-slate-700 ml-1">
                                Target Objective
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
                                label="Number of Samples"
                                type="number"
                                value={nSamples}
                                onChange={(e) => setNSamples(parseInt(e.target.value))}
                            />
                            <Input
                                label="Trust Radius"
                                type="number"
                                step="0.01"
                                value={trustRadius}
                                onChange={(e) => setTrustRadius(parseFloat(e.target.value))}
                            />
                        </div>

                        <Button
                            type="submit"
                            className="w-full py-6 text-lg shadow-lg shadow-primary/20"
                            disabled={isGenerating || !selectedDataset}
                        >
                            {isGenerating ? "Generating..." : "Generate Candidates"}
                        </Button>
                    </form>
                </Card>

                {result && (
                    <Card title="Generation Insight">
                        <div className="space-y-3">
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-500">Pathway</span>
                                <span className={`px-2 py-0.5 rounded-full font-medium ${result.pathway === "coherent"
                                    ? "bg-green-50 text-green-600"
                                    : "bg-amber-50 text-amber-600"
                                    }`}>
                                    {result.pathway}
                                </span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-500">Candidates</span>
                                <span className="font-semibold text-slate-700">
                                    {result.candidate_objectives.length}
                                </span>
                            </div>
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-500">Feasibility</span>
                                <span className={result.is_inside_mesh ? "text-green-600" : "text-amber-600"}>
                                    {result.is_inside_mesh ? "Inside Mesh" : "Outside Mesh"}
                                </span>
                            </div>
                        </div>
                    </Card>
                )}
            </div>

            {/* Main Content Area - Dual Plots */}
            <div className="lg:col-span-3 space-y-8">
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                    {/* Decision Space Plot */}
                    <Card title="Decision Space (X)">
                        <div className="h-[450px]">
                            <PlotlyWrapper
                                data={decisionPlotData}
                                layout={{
                                    autosize: true,
                                    margin: { l: 40, r: 20, t: 40, b: 40 },
                                    xaxis: {
                                        title: { text: "Dimension 1" },
                                        gridcolor: "#f1f5f9",
                                        range: decXRange || undefined
                                    },
                                    yaxis: {
                                        title: { text: "Dimension 2" },
                                        gridcolor: "#f1f5f9",
                                        range: decYRange || undefined
                                    },
                                    plot_bgcolor: "white",
                                    paper_bgcolor: "white",
                                    showlegend: true,
                                    legend: { orientation: "h", y: -0.2 }
                                }}
                                style={{ width: "100%", height: "100%" }}
                                useResizeHandler={true}
                            />
                        </div>
                    </Card>

                    {/* Objective Space Plot */}
                    <Card title="Objective Space (y)">
                        <div className="h-[450px]">
                            <PlotlyWrapper
                                data={objectivePlotData}
                                layout={{
                                    autosize: true,
                                    margin: { l: 40, r: 20, t: 40, b: 40 },
                                    xaxis: {
                                        title: { text: "Objective 1" },
                                        gridcolor: "#f1f5f9",
                                        range: objXRange || undefined
                                    },
                                    yaxis: {
                                        title: { text: "Objective 2" },
                                        gridcolor: "#f1f5f9",
                                        range: objYRange || undefined
                                    },
                                    plot_bgcolor: "white",
                                    paper_bgcolor: "white",
                                    showlegend: true,
                                    legend: { orientation: "h", y: -0.2 }
                                }}
                                style={{ width: "100%", height: "100%" }}
                                useResizeHandler={true}
                            />
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
}
