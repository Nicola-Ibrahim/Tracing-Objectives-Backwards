"use client";

import { useEffect, useState } from "react";
import { fetchDatasets, generateCandidates } from "@/lib/apiClient";
import { GenerationRequest, GenerationResponse } from "@/types/api";
import PlotlyWrapper from "@/components/PlotlyWrapper";
import { Card, Button, Input } from "@/components/ui";
import { useToast } from "@/components/ui/ToastContext";

export default function GeneratePage() {
    const [datasets, setDatasets] = useState<string[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [targetObj1, setTargetObj1] = useState<number>(0);
    const [targetObj2, setTargetObj2] = useState<number>(0);
    const [nSamples, setNSamples] = useState<number>(50);
    const [trustRadius, setTrustRadius] = useState<number>(0.05);

    const [result, setResult] = useState<GenerationResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const { showToast } = useToast();

    useEffect(() => {
        fetchDatasets()
            .then((data) => {
                setDatasets(data);
                if (data.length > 0) setSelectedDataset(data[0]);
            })
            .catch((err) => showToast(err.message, "error"));
    }, [showToast]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
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
            setLoading(false);
        }
    };

    const plotData: any[] = result
        ? [
            {
                x: result.candidate_objectives.map((o) => o[0]),
                y: result.candidate_objectives.map((o) => o[1]),
                mode: "markers",
                type: "scatter",
                name: "Generated Candidates",
                marker: { color: "#6366f1", size: 8, opacity: 0.7 },
            },
            {
                x: [result.target_objective[0]],
                y: [result.target_objective[1]],
                mode: "markers",
                type: "scatter",
                name: "Target Objective",
                marker: {
                    color: "#ef4444",
                    size: 14,
                    symbol: "cross",
                    line: { width: 2, color: "white" },
                },
            },
        ]
        : [];

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <section className="lg:col-span-1">
                <Card title="Generation Parameters">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="space-y-1.5">
                            <label className="text-sm font-semibold text-slate-700 ml-1">Dataset Source</label>
                            <select
                                value={selectedDataset}
                                onChange={(e) => setSelectedDataset(e.target.value)}
                                className="w-full px-4 py-2.5 rounded-xl border border-slate-200 bg-white/50 focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all text-sm"
                            >
                                {datasets.map((name) => (
                                    <option key={name} value={name}>
                                        {name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <Input
                                label="Target Obj 1"
                                type="number"
                                step="any"
                                value={targetObj1}
                                onChange={(e) => setTargetObj1(Number(e.target.value))}
                            />
                            <Input
                                label="Target Obj 2"
                                type="number"
                                step="any"
                                value={targetObj2}
                                onChange={(e) => setTargetObj2(Number(e.target.value))}
                            />
                        </div>

                        <Input
                            label="Sample Count"
                            type="number"
                            value={nSamples}
                            onChange={(e) => setNSamples(Number(e.target.value))}
                        />

                        <Input
                            label="Trust Radius"
                            type="number"
                            step="0.01"
                            value={trustRadius}
                            onChange={(e) => setTrustRadius(Number(e.target.value))}
                        />

                        <Button type="submit" className="w-full py-4 text-lg" isLoading={loading}>
                            Generate Candidates
                        </Button>
                    </form>
                </Card>
            </section>

            <section className="lg:col-span-2">
                <Card
                    className="h-full min-h-[600px] flex flex-col relative overflow-hidden"
                    title="Result Visualization"
                >
                    {result ? (
                        <div className="flex-1 flex flex-col animate-in fade-in duration-700">
                            <div className="flex flex-wrap gap-3 mb-8">
                                <div className="px-4 py-1.5 bg-indigo-50 text-indigo-700 border border-indigo-100 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                                    Pathway: {result.pathway}
                                </div>
                                {result.is_inside_mesh && (
                                    <div className="px-4 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                        Inside Mesh
                                    </div>
                                )}
                            </div>

                            <div className="flex-1 bg-slate-50/50 rounded-2xl border border-slate-100 p-4">
                                <PlotlyWrapper
                                    data={plotData}
                                    layout={{
                                        xaxis: { title: { text: "Objective 1" }, gridcolor: "#f1f5f9", zeroline: false },
                                        yaxis: { title: { text: "Objective 2" }, gridcolor: "#f1f5f9", zeroline: false },
                                    }}
                                />
                            </div>
                        </div>
                    ) : (
                        <div className="flex-1 flex flex-col items-center justify-center text-slate-300 space-y-6">
                            <div className="w-24 h-24 bg-slate-50 rounded-full flex items-center justify-center border border-slate-100 shadow-inner">
                                <svg
                                    className="w-10 h-10 opacity-30"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth="1.5"
                                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                                    />
                                </svg>
                            </div>
                            <p className="font-medium text-slate-400">Adjust parameters and hit generate</p>
                        </div>
                    )}
                </Card>
            </section>
        </div>
    );
}
