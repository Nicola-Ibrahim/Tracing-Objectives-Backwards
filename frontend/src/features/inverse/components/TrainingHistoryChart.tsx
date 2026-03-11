import React from 'react';
import { BasePlot } from "@/components/ui/BasePlot";
import { Activity } from "lucide-react";

interface TrainingHistoryData {
    epochs: number[];
    train_loss: number[];
    val_loss: number[];
    [key: string]: any;
}

interface TrainingHistoryChartProps {
    history: TrainingHistoryData;
    title?: string;
}

export const TrainingHistoryChart: React.FC<TrainingHistoryChartProps> = ({ 
    history, 
    title = "Convergence Profile" 
}) => {
    if (!history || !history.epochs || history.epochs.length === 0) {
        return null;
    }

    const { epochs, train_loss, val_loss, ...extras } = history;

    const traces: any[] = [
        {
            x: epochs,
            y: train_loss,
            name: 'Training Loss',
            type: 'scatter',
            mode: 'lines',
            line: { 
                color: 'rgb(79, 70, 229)', 
                width: 2.5,
            },
        },
        {
            x: epochs,
            y: val_loss,
            name: 'Validation Loss',
            type: 'scatter',
            mode: 'lines',
            line: { 
                color: 'rgb(13, 148, 136)', 
                width: 2.5,
                dash: 'dot',
            },
        }
    ];

    // Handle extra metrics like recon and kl for CVAE
    if (extras.train_recon) {
        traces.push({
            x: epochs,
            y: extras.train_recon,
            name: 'Reconstruction (Train)',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(79, 70, 229, 0.4)', width: 1.5, dash: 'dash' },
            visible: 'legendonly'
        });
    }

    if (extras.train_kl) {
        traces.push({
            x: epochs,
            y: extras.train_kl,
            name: 'KL Divergence (Train)',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(225, 29, 72, 0.4)', width: 1.5, dash: 'dash' },
            visible: 'legendonly'
        });
    }

    const layout = {
        xaxis: {
            title: { text: 'Epoch', font: { size: 10, weight: 'bold' } },
            gridcolor: '#f1f5f9',
            zerolinecolor: '#e2e8f0',
            tickfont: { size: 9 },
        },
        yaxis: {
            title: { text: 'Loss Magnitude', font: { size: 10, weight: 'bold' } },
            gridcolor: '#f1f5f9',
            zerolinecolor: '#e2e8f0',
            tickfont: { size: 9 },
            autorange: true,
        },
        margin: { t: 40, r: 40, b: 140, l: 60 },
        hovermode: 'x unified',
    };

    return (
        <BasePlot
            title={title}
            description="Training vs Validation loss convergence"
            data={traces}
            layout={layout}
            headerIcon={<Activity className="h-4 w-4 text-indigo-500" />}
            contentClassName="h-[450px]"
        />
    );
};
