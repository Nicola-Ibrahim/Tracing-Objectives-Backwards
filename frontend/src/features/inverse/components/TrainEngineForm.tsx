"use client";

import React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { trainEngineSchema, TrainEngineFormValues } from "./schema";
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
import { TrainEngineRequest } from "../types"; // Assuming split_ratio is removed from this imported type
import { BrainCircuit, Loader2, Settings2 } from "lucide-react";
import { z } from "zod";

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
    const form = useForm<TrainEngineFormValues>({
        resolver: zodResolver(trainEngineSchema) as any,
        defaultValues: {
            dataset_name: "",
            solver_type: "GBPI",
            gbpi_params: {
                n_neighbors: 10,
                trust_radius: 0.1,
                concentration_factor: 1.0,
            },
            mdn_params: {
                n_hidden: 64,
                n_mixtures: 5,
                epochs: 100,
                lr: 0.001,
            },
        },
    });

    const solverType = form.watch("solver_type");

    const handleInternalSubmit = async (values: TrainEngineFormValues) => {
        // Map internal flat values to API format
        const solverParams =
            values.solver_type === "GBPI" ? values.gbpi_params : values.mdn_params;

        const request: TrainEngineRequest = {
            dataset_name: values.dataset_name,
            solver: {
                type: values.solver_type,
                params: solverParams || {},
            },
            // Default transforms if not provided in simple form
            transforms: [
                { target: "decisions", type: "standard" },
                { target: "objectives", type: "min_max" },
            ],
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
                                <FormLabel>Dataset</FormLabel>
                                <Select
                                    onValueChange={field.onChange}
                                    defaultValue={field.value}
                                >
                                    <FormControl>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select a dataset" />
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
                        control={form.control}
                        name="solver_type"
                        render={({ field }) => (
                            <FormItem>
                                <FormLabel>Solver Strategy</FormLabel>
                                <Select
                                    onValueChange={field.onChange}
                                    defaultValue={field.value}
                                >
                                    <FormControl>
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select a solver" />
                                        </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                        <SelectItem value="GBPI">GBPI (Geometric)</SelectItem>
                                        <SelectItem value="MDN">MDN (Probabilistic)</SelectItem>
                                    </SelectContent>
                                </Select>
                                <FormMessage />
                            </FormItem>
                        )}
                    />
                </div>

                <div className="pt-2 border-t border-slate-100">
                    <h3 className="text-sm font-semibold text-slate-900 mb-4 flex items-center gap-2">
                        <Settings2 className="h-4 w-4" />
                        Hyperparameters for {solverType}
                    </h3>

                    {solverType === "GBPI" && (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <FormField
                                control={form.control}
                                name="gbpi_params.n_neighbors"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Neighbors (K)</FormLabel>
                                        <FormControl>
                                            <Input type="number" {...field} />
                                        </FormControl>
                                        <FormDescription>Local context size.</FormDescription>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="gbpi_params.trust_radius"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Trust Radius</FormLabel>
                                        <FormControl>
                                            <Input type="number" step="0.01" {...field} />
                                        </FormControl>
                                        <FormDescription>Local fidelity bound.</FormDescription>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="gbpi_params.concentration_factor"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Concentration</FormLabel>
                                        <FormControl>
                                            <Input type="number" step="0.1" {...field} />
                                        </FormControl>
                                        <FormDescription>Dirichlet factor.</FormDescription>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                        </div>
                    )}

                    {solverType === "MDN" && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <FormField
                                control={form.control}
                                name="mdn_params.n_mixtures"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Mixtures</FormLabel>
                                        <FormControl>
                                            <Input type="number" {...field} />
                                        </FormControl>
                                        <FormDescription>Number of Gaussian components.</FormDescription>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="mdn_params.epochs"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Epochs</FormLabel>
                                        <FormControl>
                                            <Input type="number" {...field} />
                                        </FormControl>
                                        <FormDescription>Maximum training iterations.</FormDescription>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                        </div>
                    )}
                </div>

                <Button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-6"
                >
                    {isLoading ? (
                        <Loader2 className="h-5 w-5 animate-spin mr-2" />
                    ) : (
                        <BrainCircuit className="h-5 w-5 mr-2" />
                    )}
                    Initiliaze Engine Construction
                </Button>
            </form>
        </Form>
    );
}
