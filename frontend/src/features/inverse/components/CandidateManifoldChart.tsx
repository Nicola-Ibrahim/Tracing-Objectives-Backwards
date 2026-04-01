"use client";

import React, { useMemo } from "react";
import { CandidateGenerationResponse } from "../types";
import { Card, CardContent } from "@/components/ui/card";
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
        const traces: Record<string, unknown>[] = [];

        // 1. Background Population
        if (backgroundY) {
            traces.push({
                x: backgroundY.map(obj => obj[0]),
                y: backgroundY.map(obj => obj[1]),
                mode: 'markers',
                type: 'scatter',
                name: "Reference Data",
                marker: { color: 'rgba(203, 213, 225, 0.4)', size: 8 },
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
                marker: { color: 'rgba(99, 102, 241, 0.5)', size: 14 },
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
                line: { color: 'rgba(79, 70, 229, 0.3)', width: 3, dash: 'dot' },
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
                        line: { color: 'rgba(245, 158, 11, 0.6)', width: 3 },
                        marker: { size: 8, color: 'white', line: { width: 1.5, color: '#f59e0b' } },
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
                    size: 24,
                    color: '#ef4444',
                    line: { width: 2, color: 'white' }
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
                    size: 24,
                    color: '#22c55e',
                    line: { width: 4, color: '#22c55e' }
                },
                hovertemplate: 'Target Profile<br>f1: %{x:.4f}<br>f2: %{y:.4f}<extra></extra>',
                hoverlabel: { bgcolor: 'white', bordercolor: '#22c55e', font: { family: 'Inter, sans-serif' } }
            });
        }
        return traces;
    }, [data, backgroundY]);

    const decisionTraces = useMemo(() => {
        const traces: Record<string, unknown>[] = [];

        if (backgroundX) {
            traces.push({
                x: backgroundX.map(dec => dec[0]),
                y: backgroundX.map(dec => dec[1] || 0),
                mode: 'markers',
                type: 'scatter',
                name: "Reference Data",
                marker: { color: 'rgba(203, 213, 225, 0.4)', size: 8 },
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
                marker: { color: 'rgba(245, 158, 11, 0.4)', size: 14 },
                hovertemplate: 'Decision Sample<br>x1: %{x:.4f}<br>x2: %{y:.4f}<extra></extra>',
                hoverlabel: { bgcolor: 'white', bordercolor: '#f59e0b', font: { family: 'Inter, sans-serif' } }
            });

            traces.push({
                x: [data.best_candidate_decision[0]],
                y: [data.best_candidate_decision[1] || 0],
                mode: 'markers',
                type: 'scatter',
                name: "Best Decision",
                marker: { symbol: 'star', size: 24, color: '#ef4444', line: { width: 2, color: 'white' } },
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
                        line: { color: 'rgba(245, 158, 11, 0.4)', width: 2.5, dash: 'dot' },
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
                        <Badge variant="outline" className="bg-indigo-500/10 text-indigo-500 border-indigo-500/20 py-1.5 px-3 font-bold">
                            <Target className="h-3.5 w-3.5 mr-1.5" />
                            Target: {data.target_objective.map(v => v.toFixed(3)).join(", ")}
                        </Badge>
                        <Badge variant="outline" className="py-1.5 px-3 text-muted-foreground font-bold uppercase tracking-widest text-[10px] bg-background border-border font-heading">
                            {data.candidate_decisions.length} Candidates Identified
                        </Badge>
                    </div>
                ) : (
                    <div className="flex items-center gap-3">
                        <Badge variant="outline" className="bg-muted text-muted-foreground border-border py-1.5 px-3 uppercase text-[10px] font-bold tracking-widest font-heading">
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
                    headerExtra={<Badge variant="outline" className="text-[10px] border-border text-muted-foreground px-2 py-0.5 leading-tight font-bold uppercase tracking-widest font-heading">Y-Space</Badge>}
                />

                <BasePlot
                    title="Decision Geometry (x1, x2)"
                    description="X-Space mapping of latent features"
                    data={decisionTraces}
                    layout={layoutY}
                    headerExtra={<Badge variant="outline" className="text-[10px] border-border text-muted-foreground px-2 py-0.5 leading-tight font-bold uppercase tracking-widest font-heading">X-Space</Badge>}
                />
            </div>

            {data && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="border-border bg-card overflow-hidden relative group/winner rounded-[1rem] transition-all">
                        <div className="absolute top-0 left-0 w-1.5 h-full bg-indigo-500" />
                        <CardContent className="py-8 pl-10 pr-10">
                            <div className="flex items-start gap-4">
                                <div className="bg-indigo-500 p-3 rounded-[1rem] text-white">
                                    <CheckCircle2 className="h-6 w-6" />
                                </div>
                                <div className="grow">
                                    <h4 className="font-bold text-foreground text-xl tracking-tight uppercase font-heading">Optimal Solution Vector</h4>
                                    <p className="text-[10px] font-bold text-muted-foreground mb-8 uppercase tracking-widest opacity-60 font-heading">Best candidate approximation identified at index <span className="font-mono text-indigo-500 font-bold">#{data.best_index}</span>.</p>

                                    <div className="flex flex-col gap-5 mt-6">
                                        <div className="bg-muted/30 p-6 rounded-[1rem] border border-border flex flex-col gap-5 group/row hover:bg-background transition-all duration-500">
                                            <div className="flex items-center gap-4">
                                                <div className="bg-background p-3 rounded-[1rem] group-hover/row:bg-indigo-500/10 border border-border transition-colors">
                                                    <Target className="h-5 w-5 text-indigo-500" />
                                                </div>
                                                <div>
                                                    <span className="text-[10px] uppercase font-bold text-indigo-500 block tracking-widest font-heading">Objective Alignment</span>
                                                    <span className="text-[10px] text-muted-foreground block font-bold italic opacity-60">Distance to Manifold target profile</span>
                                                </div>
                                            </div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                                <div className="flex flex-col gap-1">
                                                    <span className="text-[9px] text-indigo-500/70 font-bold uppercase tracking-[0.15em] font-heading">f1 Objective</span>
                                                    <span className="font-mono text-sm font-bold text-foreground break-all leading-tight">
                                                        {data.best_candidate_objective[0].toFixed(6)}
                                                    </span>
                                                </div>
                                                <div className="flex flex-col gap-1">
                                                    <span className="text-[9px] text-indigo-500/70 font-bold uppercase tracking-[0.15em] font-heading">f2 Objective</span>
                                                    <span className="font-mono text-sm font-bold text-foreground break-all leading-tight">
                                                        {data.best_candidate_objective[1].toFixed(6)}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="bg-muted/30 p-6 rounded-[1rem] border border-border flex flex-col gap-5 group/row hover:bg-background transition-all duration-500">
                                            <div className="flex items-center gap-4">
                                                <div className="bg-background p-3 rounded-[1rem] border border-border transition-colors group-hover/row:bg-rose-500/10">
                                                    <Activity className="h-5 w-5 text-muted-foreground group-hover/row:text-rose-500" />
                                                </div>
                                                <div>
                                                    <span className="text-[10px] uppercase font-bold text-muted-foreground group-hover/row:text-rose-500 block tracking-widest transition-colors font-heading">Residual Tolerance</span>
                                                    <span className="text-[10px] text-muted-foreground block font-bold italic opacity-60">L2 Approximation Error</span>
                                                </div>
                                            </div>
                                            <div className="bg-background/50 px-6 py-4 rounded-[1rem] border border-border font-mono text-sm font-bold text-foreground text-center">
                                                {data.best_candidate_residual.toExponential(4)}
                                            </div>
                                        </div>

                                        <div className="bg-muted/30 p-6 rounded-[1rem] border border-border flex flex-col gap-5 group/row hover:bg-background transition-all duration-500">
                                            <div className="flex items-center gap-4">
                                                <div className="bg-background p-3 rounded-[1rem] border border-border transition-colors group-hover/row:bg-indigo-500/10">
                                                    <Network className="h-5 w-5 text-muted-foreground group-hover/row:text-indigo-500" />
                                                </div>
                                                <div>
                                                    <span className="text-[10px] uppercase font-bold text-muted-foreground group-hover/row:text-indigo-500 block tracking-widest transition-colors font-heading">Feature Vectors</span>
                                                    <span className="text-[10px] text-muted-foreground block font-bold italic opacity-60">High-dim parameter sample</span>
                                                </div>
                                            </div>
                                            <div className="bg-background p-4 rounded-[1rem] border border-border">
                                                <div className="font-mono text-[10px] text-muted-foreground break-all leading-relaxed bg-muted/50 p-2 rounded-[1rem] mb-4 border border-border/50">
                                                    {`[ ${data.best_candidate_decision.slice(0, 5).map(v => v.toFixed(3)).join(", ")} ... ]`}
                                                </div>
                                                <div className="flex items-center gap-4">
                                                    <div className="grow h-2 bg-muted rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-indigo-500 rounded-full"
                                                            style={{ width: `${Math.min(100, Math.max(10, (1 - data.best_candidate_residual) * 100))}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-[9px] font-bold text-indigo-500 tracking-widest uppercase font-heading">Stability</span>
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
                            <Card className="border-border bg-card rounded-[1rem] transition-all">
                                <CardContent className="py-8 px-10">
                                    <div className="flex items-start gap-4">
                                        <div className="bg-muted p-2.5 rounded-[1rem] text-muted-foreground">
                                            <Network className="h-5 w-5" />
                                        </div>
                                        <div className="grow">
                                            <h4 className="font-bold text-foreground text-xs uppercase tracking-[0.2em] mb-4 font-heading">Geometric Analysis</h4>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="space-y-1 bg-muted/30 p-4 rounded-[1rem] border border-border hover:bg-background transition-colors">
                                                    <span className="text-[9px] font-bold uppercase text-muted-foreground block tracking-widest font-heading">Propagation</span>
                                                    <Badge variant="outline" className={`text-[10px] font-bold uppercase border-0 shadow-none p-0 font-heading ${data.metadata?.pathway === 'coherent' ? 'text-emerald-500' : 'text-amber-500'}`}>
                                                        {data.metadata?.pathway || "Asynchronous"}
                                                    </Badge>
                                                </div>
                                                <div className="space-y-1 bg-muted/30 p-4 rounded-[1rem] border border-border hover:bg-background transition-colors">
                                                    <span className="text-[9px] font-bold uppercase text-muted-foreground block tracking-widest font-heading">Topology</span>
                                                    <span className="text-[10px] font-bold text-foreground uppercase font-heading">
                                                        {data.metadata?.is_simplex_found ? "Simplex Anchor" : "KNN Interpolation"}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        ) : (data.metadata?.log_likelihood ? (
                            <Card className="border-border bg-card rounded-[1rem] transition-all">
                                <CardContent className="py-8 px-10">
                                    <div className="flex items-start gap-4">
                                        <div className="bg-indigo-500 p-2.5 rounded-[1rem] text-white">
                                            <Activity className="h-5 w-5" />
                                        </div>
                                        <div className="grow">
                                            <h4 className="font-bold text-foreground text-xs uppercase tracking-[0.2em] mb-4 font-heading">Generative Density</h4>
                                            <div className="space-y-4">
                                                <div className="flex items-center justify-between text-[10px] font-bold text-muted-foreground uppercase tracking-widest font-heading">
                                                    <span>Log-Likelihood Manifold</span>
                                                    <div className="flex gap-4">
                                                        <span>Min: {Math.min(...data.metadata.log_likelihood).toFixed(2)}</span>
                                                        <span>Max: {Math.max(...data.metadata.log_likelihood).toFixed(2)}</span>
                                                    </div>
                                                </div>
                                                <div className="bg-muted/30 p-6 rounded-[1rem] border border-border group transition-all hover:bg-background">
                                                    <div className="flex items-end gap-1.5 h-16">
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
                                                                <div key={i} className="bg-indigo-500/20 rounded-full w-full hover:bg-indigo-500 transition-all duration-300" style={{ height: `${(b / (maxBin || 1)) * 100}%` }} />
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
