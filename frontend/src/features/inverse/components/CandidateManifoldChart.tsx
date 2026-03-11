"use client";

import React, { useMemo } from "react";
import { CandidateGenerationResponse } from "../types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Target, Info, CheckCircle2, Network, Activity } from "lucide-react";
import { BasePlot } from "@/components/ui/BasePlot";

interface CandidateManifoldChartProps {
    data?: CandidateGenerationResponse;
    backgroundX?: number[][];
    backgroundY?: number[][];
}

export function CandidateManifoldChart({ data, backgroundX, backgroundY }: CandidateManifoldChartProps) {

    const objectiveTraces = useMemo(() => {
        const traces: any[] = [];

        // 1. Background Population
        if (backgroundY) {
            traces.push({
                x: backgroundY.map(obj => obj[0]),
                y: backgroundY.map(obj => obj[1]),
                mode: 'markers',
                type: 'scatter',
                name: "Reference Data",
                marker: { color: 'rgba(203, 213, 225, 0.4)', size: 6 },
                hoverinfo: 'skip'
            });
        }

        if (data) {
            // 2. Candidates
            traces.push({
                x: data.candidate_objectives.map(obj => obj[0]),
                y: data.candidate_objectives.map(obj => obj[1]),
                mode: 'markers',
                type: 'scatter',
                name: "Generated Candidates",
                marker: { color: 'rgba(99, 102, 241, 0.5)', size: 10 },
                hovertemplate: 'Candidate<br>f1: %{x:.4f}<br>f2: %{y:.4f}<extra></extra>',
                hoverlabel: { bgcolor: 'white', bordercolor: '#6366f1', font: { family: 'Inter, sans-serif' } }
            });

            // 3. Target Alignment Line
            traces.push({
                x: [data.target_objective[0], data.best_candidate_objective[0]],
                y: [data.target_objective[1], data.best_candidate_objective[1]],
                mode: 'lines',
                type: 'scatter',
                name: "Alignment",
                line: { color: 'rgba(79, 70, 229, 0.3)', width: 2, dash: 'dot' },
                hoverinfo: 'skip'
            });

            // 4. Simplex
            if (data.metadata?.vertices_indices && data.metadata.vertices_indices.length === 3 && backgroundY) {
                const v = data.metadata.vertices_indices.map((idx: number) => backgroundY[idx]);
                if (v.every(Boolean)) {
                    traces.push({
                        x: [...v.map((p: number[]) => p[0]), v[0][0]],
                        y: [...v.map((p: number[]) => p[1]), v[0][1]],
                        mode: 'lines+markers',
                        type: 'scatter',
                        name: "Local Simplex",
                        fill: 'toself',
                        fillcolor: 'rgba(245, 158, 11, 0.05)',
                        line: { color: 'rgba(245, 158, 11, 0.6)', width: 2 },
                        marker: { size: 6, color: 'white', line: { width: 1, color: '#f59e0b' } },
                        hoverinfo: 'skip'
                    });
                }
            }

            // 5. Best Fit
            traces.push({
                x: [data.best_candidate_objective[0]],
                y: [data.best_candidate_objective[1]],
                mode: 'markers',
                type: 'scatter',
                name: "Best Candidate",
                marker: {
                    symbol: 'star',
                    size: 18,
                    color: '#ef4444',
                    line: { width: 1.5, color: 'white' }
                },
                hovertemplate: 'Best Selection<br>f1: %{x:.6f}<br>f2: %{y:.6f}<extra></extra>',
                hoverlabel: { bgcolor: 'white', bordercolor: '#ef4444', font: { family: 'Inter, sans-serif' } }
            });

            // 6. Target
            traces.push({
                x: [data.target_objective[0]],
                y: [data.target_objective[1]],
                mode: 'markers',
                type: 'scatter',
                name: "Target",
                marker: {
                    symbol: 'x-thin',
                    size: 16,
                    color: '#22c55e',
                    line: { width: 3, color: '#22c55e' }
                },
                hovertemplate: 'Target Profile<br>f1: %{x:.4f}<br>f2: %{y:.4f}<extra></extra>',
                hoverlabel: { bgcolor: 'white', bordercolor: '#22c55e', font: { family: 'Inter, sans-serif' } }
            });
        }
        return traces;
    }, [data, backgroundY]);

    const decisionTraces = useMemo(() => {
        const traces: any[] = [];

        if (backgroundX) {
            traces.push({
                x: backgroundX.map(dec => dec[0]),
                y: backgroundX.map(dec => dec[1] || 0),
                mode: 'markers',
                type: 'scatter',
                name: "Reference Data",
                marker: { color: 'rgba(203, 213, 225, 0.4)', size: 6 },
                hoverinfo: 'skip'
            });
        }

        if (data) {
            traces.push({
                x: data.candidate_decisions.map(dec => dec[0]),
                y: data.candidate_decisions.map(dec => dec[1] || 0),
                mode: 'markers',
                type: 'scatter',
                name: "Generated Decisions",
                marker: { color: 'rgba(245, 158, 11, 0.4)', size: 10 },
                hovertemplate: 'Decision Sample<br>x1: %{x:.4f}<br>x2: %{y:.4f}<extra></extra>',
                hoverlabel: { bgcolor: 'white', bordercolor: '#f59e0b', font: { family: 'Inter, sans-serif' } }
            });

            traces.push({
                x: [data.best_candidate_decision[0]],
                y: [data.best_candidate_decision[1] || 0],
                mode: 'markers',
                type: 'scatter',
                name: "Best Decision",
                marker: { symbol: 'star', size: 18, color: '#ef4444', line: { width: 1.5, color: 'white' } },
                hovertemplate: 'Optimal Features<extra></extra>',
                hoverlabel: { bgcolor: 'white', bordercolor: '#ef4444', font: { family: 'Inter, sans-serif' } }
            });

            if (data.metadata?.vertices_indices && data.metadata.vertices_indices.length === 3 && backgroundX) {
                const x = data.metadata.vertices_indices.map((idx: number) => backgroundX[idx]);
                if (x.every(Boolean)) {
                    traces.push({
                        x: [...x.map((p: number[]) => p[0]), x[0][0]],
                        y: [...x.map((p: number[]) => p[1]), x[0][1]],
                        mode: 'lines',
                        type: 'scatter',
                        name: "Simplex Bound",
                        line: { color: 'rgba(245, 158, 11, 0.4)', width: 1.5, dash: 'dot' },
                        hoverinfo: 'skip'
                    });
                }
            }
        }
        return traces;
    }, [data, backgroundX]);

    const layoutX = { xaxis: { title: { text: "Obj 1" } }, yaxis: { title: { text: "Obj 2" } } };
    const layoutY = { xaxis: { title: { text: "Feature 1" } }, yaxis: { title: { text: "Feature 2" } } };

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
                {data ? (
                    <div className="flex items-center gap-3">
                        <Badge variant="secondary" className="bg-indigo-50 text-indigo-700 hover:bg-indigo-50 border-indigo-100 py-1.5 px-3">
                            <Target className="h-3.5 w-3.5 mr-1.5" />
                            Target: {data.target_objective.map(v => v.toFixed(3)).join(", ")}
                        </Badge>
                        <Badge variant="outline" className="py-1.5 px-3 text-slate-500 font-medium bg-white">
                            {data.candidate_decisions.length} Candidates Identified
                        </Badge>
                    </div>
                ) : (
                    <div className="flex items-center gap-3">
                        <Badge variant="outline" className="bg-slate-50 text-slate-400 border-slate-200 py-1.5 px-3 uppercase text-[10px] font-black tracking-widest">
                            <Info className="h-3.5 w-3.5 mr-1.5" />
                            Selection Mode: Define targets on manifold
                        </Badge>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <BasePlot
                    title="Objective Manifold (f1, f2)"
                    description="Y-Space mapping of reference data and candidates"
                    data={objectiveTraces}
                    layout={layoutX}
                    contentClassName="h-[500px]"
                    headerExtra={<Badge variant="outline" className="text-[8px] border-slate-200 text-slate-400 px-1.5 py-0 leading-tight">Y-Space</Badge>}
                />

                <BasePlot
                    title="Decision Geometry (x1, x2)"
                    description="X-Space mapping of latent features"
                    data={decisionTraces}
                    layout={layoutY}
                    contentClassName="h-[500px]"
                    headerExtra={<Badge variant="outline" className="text-[8px] border-slate-200 text-slate-400 px-1.5 py-0 leading-tight">X-Space</Badge>}
                />
            </div>

            {data && (
                /* ... (keeping the rest of the winner card and metadata insights as is) ... */
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="border-indigo-100 bg-white shadow-xl shadow-indigo-500/5 overflow-hidden relative group/winner rounded-[2.5rem]">
                        <div className="absolute top-0 left-0 w-1.5 h-full bg-indigo-500" />
                        <CardContent className="py-6 px-8">
                            <div className="flex items-start gap-4">
                                <div className="bg-indigo-500 p-2 rounded-xl text-white shadow-lg shadow-indigo-200">
                                    <CheckCircle2 className="h-5 w-5" />
                                </div>
                                <div className="grow">
                                    <h4 className="font-bold text-slate-800 text-lg tracking-tight">Optimal Solution Vector</h4>
                                    <p className="text-xs font-medium text-slate-400 mb-6">Best candidate approximation identified at index #{data.best_index}.</p>
                                    <div className="flex flex-col gap-4 mt-6">
                                        <div className="bg-slate-50/50 p-5 rounded-3xl border border-slate-100 flex flex-col gap-4 group/row hover:bg-white hover:shadow-xl transition-all duration-500">
                                            <div className="flex items-center gap-3">
                                                <div className="bg-white p-2.5 rounded-xl shadow-sm group-hover/row:bg-indigo-50 transition-colors">
                                                    <Target className="h-4 w-4 text-indigo-500" />
                                                </div>
                                                <div>
                                                    <span className="text-[10px] uppercase font-black text-indigo-400 block tracking-widest">Objective Alignment</span>
                                                    <span className="text-[10px] text-slate-400 block font-medium">Distance to Manifold target profile</span>
                                                </div>
                                            </div>
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="flex items-center justify-between bg-white px-4 py-3 rounded-2xl border border-indigo-50 shadow-sm">
                                                    <span className="text-[10px] text-indigo-300 font-black uppercase">f1</span>
                                                    <span className="font-mono text-sm font-bold text-indigo-900">
                                                        {data.best_candidate_objective[0].toFixed(6)}
                                                    </span>
                                                </div>
                                                <div className="flex items-center justify-between bg-white px-4 py-3 rounded-2xl border border-indigo-50 shadow-sm">
                                                    <span className="text-[10px] text-indigo-300 font-black uppercase">f2</span>
                                                    <span className="font-mono text-sm font-bold text-indigo-900">
                                                        {data.best_candidate_objective[1].toFixed(6)}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-slate-50/50 p-5 rounded-3xl border border-slate-100 flex flex-col gap-4 group/row hover:bg-white hover:shadow-xl transition-all duration-500">
                                            <div className="flex items-center gap-3">
                                                <div className="bg-white p-2.5 rounded-xl shadow-sm group-hover/row:bg-indigo-50 transition-colors">
                                                    <Activity className="h-4 w-4 text-slate-400 group-hover/row:text-teal-400" />
                                                </div>
                                                <div>
                                                    <span className="text-[10px] uppercase font-black text-slate-400 group-hover/row:text-teal-400 block tracking-widest">Residual Tolerance</span>
                                                    <span className="text-[10px] text-slate-400 block font-medium">L2 Approximation Error</span>
                                                </div>
                                            </div>
                                            <div className="bg-white px-6 py-3 rounded-2xl border border-slate-100 font-mono text-sm font-bold text-slate-800 shadow-sm text-center">
                                                {data.best_candidate_residual.toExponential(4)}
                                            </div>
                                        </div>

                                        <div className="bg-slate-50/50 p-5 rounded-3xl border border-slate-100 flex flex-col gap-4 group/row hover:bg-white hover:shadow-xl transition-all duration-500">
                                            <div className="flex items-center gap-3">
                                                <div className="bg-white p-2.5 rounded-xl shadow-sm group-hover/row:bg-indigo-50 transition-colors">
                                                    <Network className="h-4 w-4 text-slate-400 group-hover/row:text-indigo-400" />
                                                </div>
                                                <div>
                                                    <span className="text-[10px] uppercase font-black text-slate-400 group-hover/row:text-indigo-400 block tracking-widest">Feature Vectors</span>
                                                    <span className="text-[10px] text-slate-400 block font-medium">High-dim parameter sample</span>
                                                </div>
                                            </div>
                                            <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm">
                                                <div className="font-mono text-[10px] text-slate-500 break-all leading-relaxed bg-slate-50 p-2 rounded-lg mb-4">
                                                    {`[ ${data.best_candidate_decision.slice(0, 5).map(v => v.toFixed(3)).join(", ")} ... ]`}
                                                </div>
                                                <div className="flex items-center gap-4">
                                                    <div className="grow h-2 bg-slate-100 rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-indigo-500 rounded-full shadow-[0_0_12px_rgba(99,102,241,0.6)]"
                                                            style={{ width: `${Math.min(100, Math.max(10, (1 - data.best_candidate_residual) * 100))}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-[10px] font-black text-indigo-500 tracking-widest uppercase">Stability</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    <div className="flex flex-col gap-6">
                        {data.solver_type === "GBPI" ? (
                            <Card className="border-slate-200/60 bg-white shadow-lg rounded-[2.5rem]">
                                <CardContent className="py-8 px-8">
                                    <div className="flex items-start gap-4">
                                        <div className="bg-slate-100 p-2 rounded-xl text-slate-500">
                                            <Network className="h-5 w-5" />
                                        </div>
                                        <div className="grow">
                                            <h4 className="font-bold text-slate-800 text-sm uppercase tracking-widest mb-4">Geometric Analysis</h4>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="space-y-1 bg-slate-50/50 p-3 rounded-2xl border border-slate-100">
                                                    <span className="text-[9px] font-black uppercase text-slate-400 block">Propagation</span>
                                                    <Badge variant="outline" className={`text-[10px] font-bold border-0 shadow-none p-0 ${data.metadata?.pathway === 'coherent' ? 'text-teal-600' : 'text-amber-600'}`}>
                                                        {data.metadata?.pathway || "Asynchronous"}
                                                    </Badge>
                                                </div>
                                                <div className="space-y-1 bg-slate-50/50 p-3 rounded-2xl border border-slate-100">
                                                    <span className="text-[9px] font-black uppercase text-slate-400 block">Topology</span>
                                                    <span className="text-[10px] font-bold text-slate-700">
                                                        {data.metadata?.is_simplex_found ? "Simplex Anchor" : "KNN Interpolation"}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        ) : (data.metadata?.log_likelihood ? (
                            <Card className="border-slate-200/60 bg-white shadow-lg rounded-[2.5rem]">
                                <CardContent className="py-8 px-8">
                                    <div className="flex items-start gap-4">
                                        <div className="bg-indigo-500 p-2 rounded-xl text-white">
                                            <Activity className="h-5 w-5" />
                                        </div>
                                        <div className="grow">
                                            <h4 className="font-bold text-slate-800 text-sm uppercase tracking-widest mb-4">Generative Density</h4>
                                            <div className="space-y-4">
                                                <div className="flex items-center justify-between text-[10px] font-bold text-slate-400 uppercase">
                                                    <span>Log-Likelihood Manifold</span>
                                                    <div className="flex gap-4">
                                                        <span>Min: {Math.min(...data.metadata.log_likelihood).toFixed(2)}</span>
                                                        <span>Max: {Math.max(...data.metadata.log_likelihood).toFixed(2)}</span>
                                                    </div>
                                                </div>
                                                <div className="bg-slate-50/50 p-4 rounded-3xl border border-slate-100">
                                                    <div className="flex items-end gap-1 h-12">
                                                        {(() => {
                                                            const values = data.metadata.log_likelihood;
                                                            const min = Math.min(...values);
                                                            const max = Math.max(...values);
                                                            const bins = new Array(20).fill(0);
                                                            values.forEach((v: number) => {
                                                                const idx = Math.min(Math.floor(((v - min) / (max - min || 1)) * 20), 19);
                                                                bins[idx]++;
                                                            });
                                                            const maxBin = Math.max(...bins);
                                                            return bins.map((b, i) => (
                                                                <div key={i} className="bg-indigo-500/30 rounded-full w-full hover:bg-indigo-500 transition-colors" style={{ height: `${(b / (maxBin || 1)) * 100}%` }} />
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
                </div>
            )}
        </div>
    );
}

// Keep old name for compatibility
export { CandidateManifoldChart as CandidateExplorer };
