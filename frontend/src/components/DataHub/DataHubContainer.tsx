"use client";

import React from "react";
import { useDataset } from "@/components/DatasetContext";
import ChartJSWrapper, { ChartDataset, ChartDataPoint } from "@/components/ChartJSWrapper";
import DatasetSelector from "@/components/DatasetSelector";
import DatasetGenerator from "@/components/DatasetGenerator";
import { Card, Button } from "@/components/ui";
import { useRouter } from "next/navigation";
import { Vector } from "@/types/api";

export default function DataHubContainer() {
    const {
        datasets,
        selectedDataset,
        baselineData,
        isLoading,
        selectDataset,
        refreshDatasets,
        ranges
    } = useDataset();

    const router = useRouter();

    const launchOptimizer = () => {
        if (selectedDataset) {
            router.push(`/generate?dataset=${selectedDataset}`);
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

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight mb-1">Data Hub</h1>
                    <p className="text-slate-500 text-sm font-medium">Manage datasets and build benchmarks.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Control Side Panel */}
                <div className="lg:col-span-1 space-y-6">
                    <Card title="Source Selection">
                        <DatasetSelector
                            datasets={datasets}
                            selectedDataset={selectedDataset}
                            onSelect={selectDataset}
                            isLoading={isLoading}
                            className="w-full"
                        />
                        <Button
                            className="w-full mt-4 bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-200"
                            onClick={launchOptimizer}
                            disabled={!selectedDataset || isLoading}
                        >
                            Launch Optimizer
                        </Button>
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
                                xRange={ranges.decX || undefined}
                                yRange={ranges.decY || undefined}
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
