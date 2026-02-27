"use client";

import React, { useState } from "react";
import { Card, Button, Input } from "./ui";
import { DatasetGenerationRequest } from "@/types/api";
import { generateDataset } from "@/lib/apiClient";
import { useToast } from "./ui/ToastContext";

interface DatasetGeneratorProps {
    onSuccess: (name: string) => void;
}

const DatasetGenerator: React.FC<DatasetGeneratorProps> = ({ onSuccess }) => {
    const [functionId, setFunctionId] = useState<number>(1);
    const [popSize, setPopSize] = useState<number>(200);
    const [nVar, setNVar] = useState<number>(2);
    const [generations, setGenerations] = useState<number>(0);
    const [datasetName, setDatasetName] = useState<string>("");
    const [isGenerating, setIsGenerating] = useState(false);

    const { showToast } = useToast();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!datasetName) {
            showToast("Please provide a dataset name", "error");
            return;
        }

        setIsGenerating(true);
        const request: DatasetGenerationRequest = {
            function_id: functionId,
            population_size: popSize,
            n_var: nVar,
            generations: generations,
            dataset_name: datasetName,
        };

        try {
            await generateDataset(request);
            showToast(`Dataset ${datasetName} generated!`, "success");
            onSuccess(datasetName);
            // Reset name
            setDatasetName("");
        } catch (err: any) {
            showToast(err.message, "error");
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <Card title="COCOEX Generator">
            <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                    <label className="text-sm font-medium text-slate-700">Function ID (1-55)</label>
                    <input
                        type="number"
                        min={1}
                        max={55}
                        value={functionId}
                        onChange={(e) => setFunctionId(parseInt(e.target.value))}
                        className="w-full p-2 bg-slate-50 border border-slate-200 rounded-md text-sm"
                    />
                </div>

                <Input
                    label="Population Size"
                    type="number"
                    value={popSize}
                    onChange={(e) => setPopSize(parseInt(e.target.value))}
                />

                <div className="grid grid-cols-2 gap-4">
                    <Input
                        label="Dimensions"
                        type="number"
                        value={nVar}
                        onChange={(e) => setNVar(parseInt(e.target.value))}
                    />
                    <Input
                        label="Opt. Gen (0=Random)"
                        type="number"
                        value={generations}
                        onChange={(e) => setGenerations(parseInt(e.target.value))}
                    />
                </div>

                <Input
                    label="Custom Name"
                    placeholder="e.g. my_benchmark_f1"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                />

                <Button
                    type="submit"
                    className="w-full mt-2"
                    disabled={isGenerating}
                >
                    {isGenerating ? "Generating..." : "Generate Dataset"}
                </Button>
            </form>
        </Card>
    );
};

export default DatasetGenerator;
