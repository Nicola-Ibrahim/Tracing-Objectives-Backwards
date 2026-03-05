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
import { Target, Info, CheckCircle2 } from "lucide-react";

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
                { x: data.best_objective[0], y: data.best_objective[1] }
            ],
            borderColor: "rgba(79, 70, 229, 0.4)", // indigo-600 with alpha
            borderDash: [5, 5],
            showLine: true,
            pointRadius: 0,
            fill: false,
            borderWidth: 2,
            order: 2,
        });

        // 4. Best Fit
        objectiveDatasets.push({
            label: "Best Fit",
            data: [{ x: data.best_objective[0], y: data.best_objective[1] }],
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
            data: [{ x: data.best_decision[0], y: data.best_decision[1] || 0 }],
            backgroundColor: "rgba(239, 68, 68, 1)",
            pointRadius: 10,
            pointStyle: "star",
            order: 1,
        });
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
                <Card className="border-indigo-100 bg-indigo-50/10 shadow-none border-dashed">
                    <CardContent className="py-4">
                        <div className="flex items-start gap-3">
                            <CheckCircle2 className="h-5 w-5 text-indigo-500 mt-1" />
                            <div className="grow">
                                <h4 className="font-bold text-slate-800 text-sm">Optimal Candidate Selection</h4>
                                <p className="text-xs text-slate-500 mb-3">Best approximation candidate identified at relative index #{data.best_index}.</p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="bg-white/80 p-2.5 rounded-lg border border-indigo-50 shadow-sm">
                                        <span className="text-[10px] uppercase font-black text-indigo-400 block mb-1 tracking-tighter">Objective Accuracy</span>
                                        <div className="font-mono text-[11px] text-slate-700 flex gap-2">
                                            <span className="bg-indigo-50 px-1.5 rounded">f1: {data.best_objective[0].toFixed(4)}</span>
                                            <span className="bg-indigo-50 px-1.5 rounded">f2: {data.best_objective[1].toFixed(4)}</span>
                                        </div>
                                    </div>
                                    <div className="bg-white/80 p-2.5 rounded-lg border border-slate-100 shadow-sm">
                                        <span className="text-[10px] uppercase font-black text-slate-400 block mb-1 tracking-tighter">Decision Summary</span>
                                        <div className="font-mono text-[11px] text-slate-600 overflow-x-auto whitespace-nowrap pb-1">
                                            [{data.best_decision.slice(0, 5).map(v => v.toFixed(3)).join(", ")}...]
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
