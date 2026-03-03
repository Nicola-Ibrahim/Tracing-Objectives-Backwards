"use client";

import React, { useMemo } from "react";
import ChartJSWrapper from "../ChartJSWrapper";

interface DistanceGateProps {
    distances: number[];
    tau: number;
}

const DistanceGate: React.FC<DistanceGateProps> = ({ distances = [], tau = 0 }) => {
    if (!distances || distances.length === 0) {
        return (
            <div className="w-full h-[250px] glass-panel p-6 flex flex-col items-center justify-center text-center space-y-2">
                <div className="w-10 h-10 rounded-full bg-slate-50 flex items-center justify-center">
                    <svg className="w-5 h-5 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                </div>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">No Coherence Data</p>
            </div>
        );
    }

    const filteredData = useMemo(() => {
        const validIndices = distances.map((d: number, i: number) => isFinite(d) ? i : -1).filter(i => i !== -1);
        const filteredDistances = validIndices.map(i => distances[i]);
        return {
            labels: validIndices.map(i => `Dist ${i + 1}`),
            data: filteredDistances,
        };
    }, [distances]);

    const datasets = useMemo(() => ([
        {
            label: "Anchor Distances",
            data: filteredData.data,
            backgroundColor: filteredData.data.map((d: number) =>
                d <= (tau || 0) ? "rgba(34, 197, 94, 0.6)" : "rgba(239, 68, 68, 0.6)"
            ),
        }
    ]), [filteredData, tau]);

    return (
        <div className="w-full">
            <ChartJSWrapper
                type="bar"
                title="Coherence Gate"
                labels={filteredData.labels}
                datasets={datasets as any}
                yAxisTitle="Distance"
                lineAnnotation={tau ? {
                    value: tau,
                    label: `Limit (τ = ${tau.toFixed(3)})`
                } : undefined}
            // xAxisTitle doesn't apply nicely to category but we'll leave it
            />
        </div>
    );
};

export default DistanceGate;
