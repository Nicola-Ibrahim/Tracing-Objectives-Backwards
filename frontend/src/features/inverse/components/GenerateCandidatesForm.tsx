"use client";

import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { generateCandidatesSchema, GenerateCandidatesFormValues } from "./generate-schema";
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
    FormDescription,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { CandidateGenerationRequest, EngineListItem } from "../types";
import { Play, Loader2, Info } from "lucide-react";
import { listEnginesForDataset } from "../api";
import { Badge } from "@/components/ui/badge";

interface GenerateCandidatesFormProps {
    datasets: string[];
    onSubmit: (params: CandidateGenerationRequest) => Promise<void>;
    isLoading?: boolean;
    onDatasetChange?: (name: string) => void;
}

export function GenerateCandidatesForm({
    datasets,
    onSubmit,
    isLoading = false,
    onDatasetChange,
}: GenerateCandidatesFormProps) {
    const [engines, setEngines] = useState<EngineListItem[]>([]);
    const [fetchingEngines, setFetchingEngines] = useState(false);

    const form = useForm<GenerateCandidatesFormValues>({
        resolver: zodResolver(generateCandidatesSchema) as any,
        defaultValues: {
            dataset_name: "",
            engine_id: "",
            objective1: 0.5,
            objective2: 0.5,
            num_candidates: 50,
        },
    });

    const datasetName = form.watch("dataset_name");
    const engineId = form.watch("engine_id");
    const selectedEngine = engines.find(e => `${e.solver_type}_v${e.version}` === engineId);

    useEffect(() => {
        if (datasetName) {
            setFetchingEngines(true);
            listEnginesForDataset(datasetName)
                .then(setEngines)
                .catch((err) => {
                    console.error("Failed to fetch engines", err);
                    setEngines([]);
                })
                .finally(() => setFetchingEngines(false));

            // Notify parent of dataset change
            onDatasetChange?.(datasetName);

            // Reset engine selection when dataset changes
            form.setValue("engine_id", "");
        } else {
            setEngines([]);
            form.setValue("engine_id", "");
            onDatasetChange?.("");
        }
    }, [datasetName, form, onDatasetChange]);

    const handleInternalSubmit = async (values: GenerateCandidatesFormValues) => {
        const [solver_type, versionStr] = values.engine_id.split("_v");
        const target_objective = [values.objective1, values.objective2];

        const request: CandidateGenerationRequest = {
            dataset_name: values.dataset_name,
            solver_type,
            version: parseInt(versionStr),
            target_objective,
            num_candidates: values.num_candidates,
            // Hidden defaults
            trust_radius: 0.05,
            concentration_factor: 10.0,
        };

        await onSubmit(request);
    };

    return (
        <Form {...(form as any)}>
            <form
                onSubmit={form.handleSubmit(handleInternalSubmit as any)}
                className="space-y-6"
            >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FormField
                        control={form.control as any}
                        name="dataset_name"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Reference Dataset</FormLabel>
                                <Select
                                    onValueChange={field.onChange}
                                    defaultValue={field.value}
                                >
                                    <FormControl>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select dataset" />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                        {datasets.map((d) => (
                                            <SelectItem key={d} value={d}>
                                                {d}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                                <FormMessage />
                            </FormItem>
                        )}
                    />

                    <FormField
                        control={form.control as any}
                        name="engine_id"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Inference Engine</FormLabel>
                                <Select
                                    onValueChange={field.onChange}
                                    value={field.value}
                                    disabled={!datasetName || fetchingEngines}
                                >
                                    <FormControl>
                                        <SelectTrigger>
                                            <SelectValue
                                                placeholder={fetchingEngines ? "Loading..." : "Select engine"}
                                            />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                        {engines.map((e) => (
                                            <SelectItem key={`${e.solver_type}_v${e.version}`} value={`${e.solver_type}_v${e.version}`}>
                                                <div className="flex items-center gap-2">
                                                    <span className="font-medium">{e.solver_type}</span>
                                                    <Badge variant="outline" className="text-[10px] px-1 h-4">v{e.version}</Badge>
                                                </div>
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <FormField
                        control={form.control as any}
                        name="objective1"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Objective 1 (f1)</FormLabel>
                                <FormControl>
                                    <Input type="number" step="0.01" {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control as any}
                        name="objective2"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Objective 2 (f2)</FormLabel>
                                <FormControl>
                                    <Input type="number" step="0.01" {...field} />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>

                <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
                    <FormField
                        control={form.control as any}
                        name="num_candidates"
                        render={({ field }) => (
                            <FormItem className="max-w-[150px]">
                                <FormLabel className="text-[10px] font-bold uppercase text-slate-500">Cohort Size</FormLabel>
                                <FormControl>
                                    <Input type="number" {...field} className="bg-white" />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <div className="text-[10px] italic text-slate-400 mt-6 bg-slate-50 px-2 py-1 rounded">
                        Using optimized sampler
                    </div>
                </div>

                <Button
                    type="submit"
                    disabled={isLoading || !form.watch("engine_id")}
                    className="w-full bg-slate-900 hover:bg-slate-800 text-white font-semibold py-6 mt-4"
                >
                    {isLoading ? (
                        <Loader2 className="h-5 w-5 animate-spin mr-2" />
                    ) : (
                        <Play className="h-5 w-5 mr-2" fill="currentColor" />
                    )}
                    Generate Inverse Candidates
                </Button>
            </form>
        </Form>
    );
}
