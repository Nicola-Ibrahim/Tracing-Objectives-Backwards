"use client";

import React, { useEffect } from "react";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQuery } from "@tanstack/react-query";
import { generateCandidatesSchema, GenerateCandidatesFormValues } from "./generate-schema";
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
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
import { CandidateGenerationRequest } from "../types";
import { Play, Loader2 } from "lucide-react";
import { listEnginesForDataset } from "../api";
import { Badge } from "@/components/ui/badge";
import { Resolver } from "react-hook-form";

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
    const form = useForm<GenerateCandidatesFormValues>({
        resolver: zodResolver(generateCandidatesSchema) as Resolver<GenerateCandidatesFormValues>,
        defaultValues: {
            dataset_name: "",
            engine_id: "",
            objective1: 420,
            objective2: 1400,
            n_samples: 5,
        },
    });

    const datasetName = useWatch({
        control: form.control,
        name: "dataset_name",
        defaultValue: ""
    });

    const engineId = useWatch({
        control: form.control,
        name: "engine_id",
        defaultValue: ""
    });

    const { data: engines = [], isFetching: fetchingEngines } = useQuery({
        queryKey: ["engines", datasetName],
        queryFn: () => listEnginesForDataset(datasetName),
        enabled: !!datasetName,
    });

    useEffect(() => {
        if (datasetName) {
            onDatasetChange?.(datasetName);
            form.setValue("engine_id", "");
        } else {
            onDatasetChange?.("");
            form.setValue("engine_id", "");
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
            n_samples: values.n_samples,
        };

        await onSubmit(request);
    };

    return (
        <Form {...form}>
            <form
                onSubmit={form.handleSubmit(handleInternalSubmit)}
                className="space-y-6"
            >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FormField
                        control={form.control}
                        name="dataset_name"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="text-xs font-bold uppercase text-muted-foreground">Reference Dataset</FormLabel>
                                <Select
                                    onValueChange={field.onChange}
                                    value={field.value || ""}
                                >
                                    <FormControl>
                                        <SelectTrigger className="bg-background border-border text-foreground">
                                            <SelectValue placeholder="Select dataset" />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent className="bg-popover border-border text-popover-foreground">
                                        {datasets.map((d) => (
                                            <SelectItem key={d} value={d} className="hover:bg-muted focus:bg-muted cursor-pointer">
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
                        control={form.control}
                        name="engine_id"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="text-xs font-bold uppercase text-muted-foreground">Inference Engine</FormLabel>
                                <Select
                                    onValueChange={field.onChange}
                                    value={field.value}
                                    disabled={!datasetName || fetchingEngines}
                                >
                                    <FormControl>
                                        <SelectTrigger className="bg-background border-border text-foreground">
                                            <SelectValue
                                                placeholder={fetchingEngines ? "Loading..." : "Select engine"}
                                            />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent className="bg-popover border-border text-popover-foreground">
                                        {engines.map((e) => (
                                            <SelectItem key={`${e.solver_type}_v${e.version}`} value={`${e.solver_type}_v${e.version}`} className="hover:bg-muted focus:bg-muted cursor-pointer">
                                                <div className="flex items-center gap-2">
                                                    <span className="font-medium">{e.solver_type}</span>
                                                    <Badge variant="indigo" className="text-[10px] px-1 h-4">v{e.version}</Badge>
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
                        control={form.control}
                        name="objective1"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="text-xs font-bold uppercase text-muted-foreground">Objective 1 (f1)</FormLabel>
                                <FormControl>
                                    <Input type="number" step="0.01" {...field} className="bg-background border-border text-foreground focus:ring-indigo-500/10 focus:border-indigo-500" />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                    <FormField
                        control={form.control}
                        name="objective2"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="text-xs font-bold uppercase text-muted-foreground">Objective 2 (f2)</FormLabel>
                                <FormControl>
                                    <Input type="number" step="0.01" {...field} className="bg-background border-border text-foreground focus:ring-indigo-500/10 focus:border-indigo-500" />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>

                <div className="pt-4 border-t border-border flex items-center justify-between">
                    <FormField
                        control={form.control}
                        name="n_samples"
                        render={({ field }) => (
                            <FormItem className="max-w-[200px]">
                                <FormLabel className="text-[10px] font-bold uppercase text-muted-foreground">Number of Candidates</FormLabel>
                                <FormControl>
                                    <Input type="number" {...field} className="bg-background border-border text-foreground focus:ring-indigo-500/10 focus:border-indigo-500" />
                                </FormControl>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>

                <Button
                    type="submit"
                    disabled={isLoading || !engineId}
                    className="w-full bg-foreground text-background hover:opacity-90 font-bold h-12 transition-all active:scale-[0.98] group"
                >
                    {isLoading ? (
                        <Loader2 className="h-5 w-5 animate-spin mr-2 text-indigo-500" />
                    ) : (
                        <Play className="h-5 w-5 mr-2 text-indigo-500 group-hover:scale-110 transition-transform" fill="currentColor" />
                    )}
                    Generate Inverse Candidates
                </Button>
            </form>
        </Form>
    );
}
