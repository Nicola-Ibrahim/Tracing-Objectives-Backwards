"use client";

import React from "react";
import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    ScatterController,
} from "chart.js";
import { Scatter } from "react-chartjs-2";
import { CandidateGenerationResponse } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Target, Info, CheckCircle2, Network, Activity, BarChart2 } from "lucide-react";

ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    ScatterController
);

interface CandidateExplorerProps {
    data?: CandidateGenerationResponse;
    backgroundX?: number[][];
    backgroundY?: number[][];
}

export function CandidateExplorer({ data, backgroundX, backgroundY }: CandidateExplorerProps) {
    // Prep data for Objective Space Scatter
    const objectiveDatasets: any[] = [];

    // 1. Background Dataset Points (if provided)
    if (backgroundY) {
        objectiveDatasets.push({
            label: "Dataset Population",
            data: backgroundY.map((obj) => ({ x: obj[0], y: obj[1] })),
            backgroundColor: "rgba(226, 232, 240, 0.4)", // slate-200
            pointRadius: 2,
            order: 10,
        });
    }

    if (data) {
        // 2. Candidate Objectives
        objectiveDatasets.push({
            label: "Candidate Objectives",
            data: data.candidate_objectives.map((obj) => ({ x: obj[0], y: obj[1] })),
            backgroundColor: "rgba(99, 102, 241, 0.5)",
            pointRadius: 4,
            order: 5,
        });

        // 3. Line from Target to Best Fit (Dashed)
        objectiveDatasets.push({
            label: "Target Alignment",
            data: [
                { x: data.target_objective[0], y: data.target_objective[1] },
                { x: data.best_candidate_objective[0], y: data.best_candidate_objective[1] }
            ],
            borderColor: "rgba(79, 70, 229, 0.4)", // indigo-600 with alpha
            borderDash: [5, 5],
            showLine: true,
            pointRadius: 0,
            fill: false,
            borderWidth: 2,
            order: 2,
        });

        // 4. GBPI Simplex (Triangle)
        if (data.metadata?.vertices_indices && data.metadata.vertices_indices.length === 3 && backgroundY) {
            const v1 = backgroundY[data.metadata.vertices_indices[0]];
            const v2 = backgroundY[data.metadata.vertices_indices[1]];
            const v3 = backgroundY[data.metadata.vertices_indices[2]];

            if (v1 && v2 && v3) {
                objectiveDatasets.push({
                    label: "Local Simplex",
                    data: [
                        { x: v1[0], y: v1[1] },
                        { x: v2[0], y: v2[1] },
                        { x: v3[0], y: v3[1] },
                        { x: v1[0], y: v1[1] } // Close the loop
                    ],
                    borderColor: "rgba(245, 158, 11, 0.8)", // amber-500
                    borderWidth: 2,
                    showLine: true,
                    pointRadius: 4,
                    pointBackgroundColor: "white",
                    fill: "rgba(245, 158, 11, 0.1)",
                    order: 3,
                });
            }
        }

        // 5. Best Fit
        objectiveDatasets.push({
            label: "Best Fit",
            data: [{ x: data.best_candidate_objective[0], y: data.best_candidate_objective[1] }],
            backgroundColor: "rgba(239, 68, 68, 1)", // red-500
            pointRadius: 10,
            pointStyle: "star",
            order: 1,
        });

        // 5. Target
        objectiveDatasets.push({
            label: "Target",
            data: [{ x: data.target_objective[0], y: data.target_objective[1] }],
            backgroundColor: "rgba(34, 197, 94, 1)", // green-500
            pointRadius: 8,
            pointStyle: "crossRot",
            borderWidth: 1,
            borderColor: "rgba(34, 197, 94, 1)",
            order: 0,
        });
    }

    const objectiveData = { datasets: objectiveDatasets };

    // Prep data for Decision Space (Dimensions reduced to first 2 for preview if > 2)
    const decisionDatasets: any[] = [];

    if (backgroundX) {
        decisionDatasets.push({
            label: "Dataset Population",
            data: backgroundX.map((dec) => ({ x: dec[0], y: dec[1] || 0 })),
            backgroundColor: "rgba(226, 232, 240, 0.4)",
            pointRadius: 2,
            order: 10,
        });
    }

    if (data) {
        decisionDatasets.push({
            label: "Candidate Decisions",
            data: data.candidate_decisions.map((dec) => ({ x: dec[0], y: dec[1] || 0 })),
            backgroundColor: "rgba(245, 158, 11, 0.5)", // amber-500
            pointRadius: 4,
            order: 5,
        });

        decisionDatasets.push({
            label: "Best Decision",
            data: [{ x: data.best_candidate_decision[0], y: data.best_candidate_decision[1] || 0 }],
            backgroundColor: "rgba(239, 68, 68, 1)",
            pointRadius: 10,
            pointStyle: "star",
            order: 1,
        });

        // 4. GBPI Simplex (X-Space)
        if (data.metadata?.vertices_indices && data.metadata.vertices_indices.length === 3 && backgroundX) {
            const x1 = backgroundX[data.metadata.vertices_indices[0]];
            const x2 = backgroundX[data.metadata.vertices_indices[1]];
            const x3 = backgroundX[data.metadata.vertices_indices[2]];

            if (x1 && x2 && x3) {
                decisionDatasets.push({
                    label: "Local Simplex (X)",
                    data: [
                        { x: x1[0], y: x1[1] || 0 },
                        { x: x2[0], y: x2[1] || 0 },
                        { x: x3[0], y: x3[1] || 0 },
                        { x: x1[0], y: x1[1] || 0 }
                    ],
                    borderColor: "rgba(245, 158, 11, 0.6)",
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    showLine: true,
                    pointRadius: 0,
                    order: 2,
                });
            }
        }
    }

    const decisionData = { datasets: decisionDatasets };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                grid: { color: "rgba(0,0,0,0.03)" },
                ticks: { font: { size: 10 } }
            },
            y: {
                grid: { color: "rgba(0,0,0,0.03)" },
                ticks: { font: { size: 10 } }
            },
        },
        plugins: {
            legend: {
                display: true,
                position: "bottom" as const,
                labels: {
                    boxWidth: 8,
                    usePointStyle: true,
                    font: { size: 10 }
                }
            },
            tooltip: {
                backgroundColor: "rgba(15, 23, 42, 0.9)", // slate-900
                padding: 10,
                titleFont: { size: 12 },
                bodyFont: { size: 11 },
            }
        },
        animation: {
            duration: 1000,
            easing: "easeOutQuart" as const
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
                {data ? (
                    <div className="flex items-center gap-3">
                        <Badge variant="secondary" className="bg-indigo-50 text-indigo-700 hover:bg-indigo-50 border-indigo-100 py-1.5 px-3">
                            <Target className="h-3.5 w-3.5 mr-1.5" />
                            Target: {data.target_objective.map(v => v.toFixed(3)).join(", ")}
                        </Badge>
                        <Badge variant="outline" className="py-1.5 px-3 text-slate-500 font-medium">
                            {data.candidate_decisions.length} Candidates Generated
                        </Badge>
                    </div>
                ) : (
                    <div className="flex items-center gap-3">
                        <Badge variant="outline" className="bg-slate-50 text-slate-500 border-slate-200 py-1.5 px-3">
                            <Info className="h-3.5 w-3.5 mr-1.5" />
                            Preview Mode: Visualize dataset population to select targets
                        </Badge>
                    </div>
                )}
                {data && (
                    <div className="flex items-center text-xs font-medium text-slate-400 gap-1 bg-slate-50 px-2 py-1 rounded">
                        <Info className="h-3.5 w-3.5" />
                        <span>Inference via {data.solver_type}</span>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="border-slate-200 shadow-sm overflow-hidden flex flex-col bg-white">
                    <CardHeader className="bg-slate-50/50 py-3 border-b border-slate-100">
                        <CardTitle className="text-sm font-bold text-slate-700 flex items-center justify-between">
                            Objective Space (f1, f2)
                            <span className="text-[10px] font-normal text-slate-400">Y-Space Visualization</span>
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="h-[400px] pt-6 grow">
                        <Scatter data={objectiveData} options={options} />
                    </CardContent>
                </Card>

                <Card className="border-slate-200 shadow-sm overflow-hidden flex flex-col bg-white">
                    <CardHeader className="bg-slate-50/50 py-3 border-b border-slate-100">
                        <CardTitle className="text-sm font-bold text-slate-700 flex items-center justify-between">
                            Decision Space (x1, x2 preview)
                            <span className="text-[10px] font-normal text-slate-400">X-Space Visualization</span>
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="h-[400px] pt-6 grow">
                        <Scatter data={decisionData} options={options} />
                    </CardContent>
                </Card>
            </div>

            {data && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Common Best Result Card */}
                    <Card className="border-indigo-100 bg-white shadow-sm overflow-hidden relative group/winner">
                        <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500" />
                        <CardContent className="py-4">
                            <div className="flex items-start gap-3">
                                <CheckCircle2 className="h-5 w-5 text-indigo-500 mt-1" />
                                <div className="grow">
                                    <h4 className="font-bold text-slate-800 text-sm">Optimal Candidate Selection</h4>
                                    <p className="text-xs text-slate-500 mb-3">Best approximation candidate identified at relative index #{data.best_index}.</p>
                                    <div className="flex flex-col gap-4 mt-6">
                                        {/* Objective Alignment - Stacked Layout */}
                                        <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex flex-col gap-3 group/row hover:bg-white hover:shadow-md transition-all duration-300 min-w-0 overflow-hidden">
                                            <div className="flex items-center gap-3">
                                                <div className="bg-white p-2 rounded-lg shadow-sm group-hover/row:bg-indigo-50 transition-colors">
                                                    <Target className="h-4 w-4 text-indigo-500" />
                                                </div>
                                                <div className="min-w-0">
                                                    <span className="text-[10px] uppercase font-black text-indigo-400 block tracking-tight">Objective Alignment</span>
                                                    <span className="text-[10px] text-slate-400 block truncate">Distance to Target profile</span>
                                                </div>
                                            </div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                <div className="flex items-center justify-between bg-white px-3 py-2 rounded-lg border border-indigo-50 shadow-sm min-w-0">
                                                    <span className="text-[9px] text-indigo-300 font-bold uppercase shrink-0">f1</span>
                                                    <span className="font-mono text-xs font-bold text-indigo-900 break-all ml-4 text-right">
                                                        {data.best_candidate_objective[0].toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 })}
                                                    </span>
                                                </div>
                                                <div className="flex items-center justify-between bg-white px-3 py-2 rounded-lg border border-indigo-50 shadow-sm min-w-0">
                                                    <span className="text-[9px] text-indigo-300 font-bold uppercase shrink-0">f2</span>
                                                    <span className="font-mono text-xs font-bold text-indigo-900 break-all ml-4 text-right">
                                                        {data.best_candidate_objective[1].toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 })}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Residual Error - Stacked Layout */}
                                        <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex flex-col gap-3 group/row hover:bg-white hover:shadow-md transition-all duration-300 min-w-0 overflow-hidden">
                                            <div className="flex items-center gap-3">
                                                <div className="bg-white p-2 rounded-lg shadow-sm group-hover/row:bg-indigo-50 transition-colors">
                                                    <Activity className="h-4 w-4 text-slate-400 group-hover/row:text-indigo-400" />
                                                </div>
                                                <div className="min-w-0">
                                                    <span className="text-[10px] uppercase font-black text-slate-400 group-hover/row:text-indigo-400 block tracking-tight">Residual Error</span>
                                                    <span className="text-[10px] text-slate-400 block truncate">Cumulative Approximation Penalty</span>
                                                </div>
                                            </div>
                                            <div className="bg-white px-4 py-2.5 rounded-lg border border-slate-100 font-mono text-sm font-bold text-slate-700 shadow-sm break-all text-center">
                                                {data.best_candidate_residual.toFixed(10)}
                                            </div>
                                        </div>

                                        {/* Decision Vectors - Stacked Layout */}
                                        <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex flex-col gap-3 group/row hover:bg-white hover:shadow-md transition-all duration-300 min-w-0 overflow-hidden">
                                            <div className="flex items-center gap-3">
                                                <div className="bg-white p-2 rounded-lg shadow-sm group-hover/row:bg-indigo-50 transition-colors">
                                                    <Info className="h-4 w-4 text-slate-400 group-hover/row:text-indigo-400" />
                                                </div>
                                                <div className="min-w-0">
                                                    <span className="text-[10px] uppercase font-black text-slate-400 group-hover/row:text-indigo-400 block tracking-tight">Decision Vectors</span>
                                                    <span className="text-[10px] text-slate-400 block truncate">Feature Space Sample (First 5 dims)</span>
                                                </div>
                                            </div>
                                            <div className="w-full flex flex-col gap-3">
                                                <div className="bg-white p-2.5 rounded-lg border border-slate-100 shadow-sm overflow-hidden">
                                                    <div className="font-mono text-[10px] text-slate-500 whitespace-normal break-all leading-relaxed px-1">
                                                        {`[ ${data.best_candidate_decision.slice(0, 5).map(v => v.toFixed(3)).join(", ")} ... ]`}
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-3">
                                                    <div className="grow h-1.5 bg-slate-100 rounded-full overflow-hidden shrink-0">
                                                        <div
                                                            className="h-full bg-indigo-500 rounded-full shadow-[0_0_8px_rgba(99,102,241,0.5)]"
                                                            style={{ width: `${Math.min(100, Math.max(5, (1 - data.best_candidate_residual) * 100))}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-[10px] font-bold text-indigo-400 whitespace-nowrap">Confidence</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Solver Specific Insights */}
                    {data.solver_type === "GBPI" ? (
                        <Card className="border-slate-200 bg-white shadow-sm">
                            <CardContent className="py-4">
                                <div className="flex items-start gap-3">
                                    <Network className="h-5 w-5 text-slate-400 mt-1" />
                                    <div className="grow">
                                        <h4 className="font-bold text-slate-800 text-sm italic">GBPI Neighborhood Analysis</h4>
                                        <div className="mt-3 grid grid-cols-2 gap-3">
                                            <div className="space-y-1">
                                                <span className="text-[10px] uppercase text-slate-400 block">Pathway</span>
                                                <Badge variant="outline" className={`text-[10px] py-0 px-2 ${data.metadata?.pathway === 'coherent' ? 'text-green-600 border-green-100 bg-green-50' : 'text-amber-600 border-amber-100 bg-amber-50'}`}>
                                                    {data.metadata?.pathway || "Unknown"}
                                                </Badge>
                                            </div>
                                            <div className="space-y-1">
                                                <span className="text-[10px] uppercase text-slate-400 block">Geometry</span>
                                                <span className="text-xs font-medium text-slate-700">
                                                    {data.metadata?.is_simplex_found ? "Simplex Anchor" : "KNN Fallback"}
                                                </span>
                                            </div>
                                            {data.metadata?.anchor_distances && (
                                                <div className="col-span-2 space-y-1 pt-1 border-t border-slate-50">
                                                    <span className="text-[10px] uppercase text-slate-400 block">Local Tightness (Avg Dist)</span>
                                                    <div className="flex gap-1 flex-wrap">
                                                        {data.metadata.anchor_distances.map((d: number, i: number) => (
                                                            <span key={i} className="text-[10px] font-mono bg-slate-100 px-1 rounded text-slate-500">
                                                                {d.toFixed(3)}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ) : (data.metadata?.log_likelihood ? (
                        <Card className="border-slate-200 bg-white shadow-sm">
                            <CardContent className="py-4">
                                <div className="flex items-start gap-3">
                                    <Activity className="h-5 w-5 text-indigo-400 mt-1" />
                                    <div className="grow">
                                        <h4 className="font-bold text-slate-800 text-sm italic">Generative Likelihood Analysis</h4>
                                        <div className="mt-3 space-y-3">
                                            <div className="flex items-center justify-between text-xs">
                                                <span className="text-slate-400 uppercase text-[10px]">Log-Likelihood Range</span>
                                                <div className="font-mono text-slate-700 flex gap-2">
                                                    <span>Min: {Math.min(...data.metadata.log_likelihood).toFixed(2)}</span>
                                                    <span>Max: {Math.max(...data.metadata.log_likelihood).toFixed(2)}</span>
                                                </div>
                                            </div>
                                            <div className="bg-slate-50 p-2 rounded border border-slate-100">
                                                <div className="flex items-center justify-between mb-1">
                                                    <span className="text-[10px] text-slate-400 uppercase">Density Distribution</span>
                                                    <BarChart2 className="h-3 w-3 text-slate-300" />
                                                </div>
                                                <div className="flex items-end gap-0.5 h-8">
                                                    {(() => {
                                                        const values = data.metadata.log_likelihood;
                                                        const min = Math.min(...values);
                                                        const max = Math.max(...values);
                                                        const bins = new Array(15).fill(0);
                                                        values.forEach((v: number) => {
                                                            const idx = Math.min(Math.floor(((v - min) / (max - min || 1)) * 15), 14);
                                                            bins[idx]++;
                                                        });
                                                        const maxBin = Math.max(...bins);
                                                        return bins.map((b, i) => (
                                                            <div key={i} className="bg-indigo-200 rounded-t-sm w-full" style={{ height: `${(b / (maxBin || 1)) * 100}%` }} />
                                                        ));
                                                    })()}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ) : null)}
                </div>
            )}
        </div>
    );
}
