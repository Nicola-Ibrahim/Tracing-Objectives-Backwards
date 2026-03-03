"use client";

import React, { useMemo } from "react";
import ChartJSWrapper, { ChartDataset } from "../ChartJSWrapper";

interface ZoomedScatterPlotProps {
    title: string;
    datasets: ChartDataset[];
    sigma: number;
    targetPoint: { x: number; y: number };
}

const ZoomedScatterPlot: React.FC<ZoomedScatterPlotProps> = ({
    title,
    datasets,
    sigma,
    targetPoint,
}) => {
    const { rings, ranges } = useMemo(() => {
        // Use the passed sigma (already in RAW units)
        const stdDev = sigma || 0.05; // Fallback if sigma is 0

        // Defined σ-rings
        const rings = [
            { radius: stdDev, color: 'rgba(16, 185, 129, 0.4)', label: '1σ (Success)', borderDash: [] },
            { radius: stdDev * 2, color: 'rgba(245, 158, 11, 0.3)', label: '2σ (Marginal)', borderDash: [5, 5] },
            { radius: stdDev * 3, color: 'rgba(239, 68, 68, 0.2)', label: '3σ (Limit)', borderDash: [10, 5] },
        ];

        // Zoom range centered on target
        // We zoom to show up to 3.5σ to see the outer limit clearly
        const padding = stdDev * 3.5;
        const zoomRange = {
            x: [targetPoint.x - padding, targetPoint.x + padding] as [number, number],
            y: [targetPoint.y - padding, targetPoint.y + padding] as [number, number],
        };

        return { rings, ranges: zoomRange };
    }, [sigma, targetPoint]);

    return (
        <div className="w-full h-full flex justify-center">
            <div className="w-full max-w-[550px]">
                <ChartJSWrapper
                    title={title}
                    datasets={datasets}
                    xAxisTitle="Objective 1"
                    yAxisTitle="Objective 2"
                    xRange={ranges.x}
                    yRange={ranges.y}
                    height="h-[550px] aspect-square"
                    circleAnnotations={rings.map(r => ({
                        center: targetPoint,
                        radius: r.radius,
                        color: r.color,
                        label: r.label,
                        borderDash: r.borderDash
                    }))}
                />
            </div>
        </div>
    );
};

export default ZoomedScatterPlot;
