"use client";

import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQuery } from "@tanstack/react-query";
import { getSolvers } from "../api";
import { trainEngineSchema, TrainEngineFormValues } from "./schema";
import { DynamicConfigForm } from "@/components/common/DynamicConfigForm";
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { TrainEngineRequest, SolverSchema } from "../types";
import { BrainCircuit, Loader2, Settings2, Sparkles } from "lucide-react";

interface TrainEngineFormProps {
    datasets: string[];
    onSubmit: (params: TrainEngineRequest) => Promise<void>;
    isLoading?: boolean;
}

export function TrainEngineForm({
    datasets,
    onSubmit,
    isLoading = false,
}: TrainEngineFormProps) {
    const [dynamicParams, setDynamicParams] = useState<Record<string, any>>({});
    const [isDynamicValid, setIsDynamicValid] = useState(true);
    const [selectedSolver, setSelectedSolver] = useState<SolverSchema | null>(null);

    const { data: discovery, isLoading: isLoadingDiscovery } = useQuery({
        queryKey: ["solvers-discovery"],
        queryFn: getSolvers,
    });

    const form = useForm<TrainEngineFormValues>({
        resolver: zodResolver(trainEngineSchema) as any,
        defaultValues: {
            dataset_name: "",
            solver_type: "GBPI",
        },
    });

    const solvers = discovery?.solvers || [];

    useEffect(() => {
        if (solvers.length > 0 && !selectedSolver) {
            const defaultSolver = solvers.find(s => s.type === "GBPI") || solvers[0];
            setSelectedSolver(defaultSolver);
            form.setValue("solver_type", defaultSolver.type);
        }
    }, [solvers, form]);

    const handleSolverChange = (solverId: string) => {
        const solver = solvers.find(s => s.type === solverId);
        if (solver) {
            setSelectedSolver(solver);
            form.setValue("solver_type", solverId);
        }
    };

    const handleInternalSubmit = async (values: TrainEngineFormValues) => {
        const request: TrainEngineRequest = {
            dataset_name: values.dataset_name,
            solver: {
                type: values.solver_type,
                params: dynamicParams,
            },
            transforms: [
                { target: "decisions", type: "standard" },
                { target: "objectives", type: "min_max" },
            ],
        };

        await onSubmit(request);
    };

    if (isLoadingDiscovery) {
        return (
            <div className="flex flex-col items-center justify-center py-10">
                <Loader2 className="h-8 w-8 animate-spin text-indigo-500 mb-2" />
                <p className="text-sm text-muted-foreground font-medium tracking-wide">Reflecting backend solvers...</p>
            </div>
        );
    }

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
                                <FormLabel className="text-[10px] font-black uppercase text-muted-foreground tracking-widest">Source Dataset</FormLabel>
                                <Select
                                    onValueChange={field.onChange}
                                    value={field.value || ""}
                                >
                                    <FormControl>
                                        <SelectTrigger className="bg-background border-border text-foreground">
                                            <SelectValue placeholder="Select a dataset" />
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
                        name="solver_type"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel className="text-[10px] font-black uppercase text-muted-foreground tracking-widest">Architecture Strategy</FormLabel>
                                <Select
                                    onValueChange={handleSolverChange}
                                    value={field.value || ""}
                                >
                                    <FormControl>
                                        <SelectTrigger className="bg-background border-border text-foreground">
                                            <SelectValue placeholder="Select a solver" />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent className="bg-popover border-border text-popover-foreground">
                                        {solvers.map((s) => (
                                            <SelectItem key={s.type} value={s.type} className="hover:bg-muted focus:bg-muted cursor-pointer">
                                                {s.name}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>

                {selectedSolver && (
                    <div className="pt-6 border-t border-border">
                        <div className="flex items-center gap-2 mb-6">
                            <Sparkles className="h-4 w-4 text-amber-500" />
                            <h3 className="text-xs font-black text-foreground uppercase tracking-[0.2em]">
                                Hyperparameters for {selectedSolver.name}
                            </h3>
                        </div>

                        <DynamicConfigForm
                            parameters={selectedSolver.parameters}
                            onChange={(vals, valid) => {
                                setDynamicParams(vals);
                                setIsDynamicValid(valid);
                            }}
                        />
                    </div>
                )}

                <Button
                    type="submit"
                    disabled={isLoading || !isDynamicValid}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600 text-white font-bold h-14 rounded-[1rem] transition-all active:scale-[0.98] mt-4"
                >
                    {isLoading ? (
                        <Loader2 className="h-5 w-5 animate-spin mr-2" />
                    ) : (
                        <BrainCircuit className="h-5 w-5 mr-2" />
                    )}
                    Launch Engine Training
                </Button>
            </form>
        </Form>
    );
}
