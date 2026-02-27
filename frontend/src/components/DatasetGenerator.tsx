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
        <div className="space-y-4">
            <h3 className="text-md font-bold text-slate-800 px-1">Benchmark Factory</h3>
            <form onSubmit={handleSubmit} className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                        <label className="text-xs font-semibold text-slate-500 ml-1">COCOEX Function</label>
                        <input
                            type="number"
                            min={1}
                            max={55}
                            value={functionId}
                            onChange={(e) => setFunctionId(parseInt(e.target.value))}
                            className="w-full px-3 py-2 bg-white/50 border border-slate-200 rounded-xl text-sm focus:bg-white focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                        />
                    </div>
                    <Input
                        label="Population"
                        type="number"
                        value={popSize}
                        onChange={(e) => setPopSize(parseInt(e.target.value))}
                        className="py-2"
                    />
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <Input
                        label="Dimensions"
                        type="number"
                        value={nVar}
                        onChange={(e) => setNVar(parseInt(e.target.value))}
                        className="py-2"
                    />
                    <Input
                        label="Opt. Gen"
                        type="number"
                        value={generations}
                        onChange={(e) => setGenerations(parseInt(e.target.value))}
                        className="py-2"
                    />
                </div>

                <Input
                    label="Dataset Alias"
                    placeholder="e.g. f1_2d_100"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    className="py-2"
                />

                <Button
                    type="submit"
                    className="w-full py-3 mt-2 shadow-md shadow-primary/10"
                    disabled={isGenerating}
                >
                    {isGenerating ? "Building..." : "Build Benchmark"}
                </Button>
            </form>
        </div>
    );
};

export default DatasetGenerator;
