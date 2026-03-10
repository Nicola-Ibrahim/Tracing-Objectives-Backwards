"use client";

import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useQuery } from "@tanstack/react-query";
import { getGenerators } from "../api";
import { DynamicConfigForm } from "@/components/common/DynamicConfigForm";
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
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Loader2, PlusCircle, Wand2 } from "lucide-react";
import { GeneratorSchema } from "../types";

const generateDatasetSchema = z.object({
    dataset_name: z.string().min(3, "Name must be at least 3 characters"),
    generator_type: z.string().min(1, "Generator type is required"),
    split_ratio: z.coerce.number().min(0).max(0.5).default(0.2),
    random_state: z.coerce.number().default(42),
});

type GenerateDatasetFormValues = z.infer<typeof generateDatasetSchema>;

interface GenerateDatasetFormProps {
    onSubmit: (values: any) => void;
    isLoading?: boolean;
}

export function GenerateDatasetForm({ onSubmit, isLoading = false }: GenerateDatasetFormProps) {
    const [dynamicParams, setDynamicParams] = useState<Record<string, any>>({});
    const [isDynamicValid, setIsDynamicValid] = useState(true);
    const [selectedGenerator, setSelectedGenerator] = useState<GeneratorSchema | null>(null);

    const { data: discovery, isLoading: isLoadingDiscovery } = useQuery({
        queryKey: ["generators-discovery"],
        queryFn: getGenerators,
    });

    const form = useForm<GenerateDatasetFormValues>({
        resolver: zodResolver(generateDatasetSchema) as any,
        defaultValues: {
            dataset_name: "",
            generator_type: "coco_pymoo",
            split_ratio: 0.2,
            random_state: 42,
        },
    });

    const generators = discovery?.generators || [];

    useEffect(() => {
        if (generators.length > 0) {
            const defaultGen = generators.find(g => g.type === "coco_pymoo") || generators[0];
            setSelectedGenerator(defaultGen);
            form.setValue("generator_type", defaultGen.type);
        }
    }, [generators, form]);

    const handleGeneratorChange = (genId: string) => {
        const gen = generators.find(g => g.type === genId);
        if (gen) {
            setSelectedGenerator(gen);
            form.setValue("generator_type", genId);
        }
    };

    const handleFormSubmit = (values: GenerateDatasetFormValues) => {
        onSubmit({
            ...values,
            params: dynamicParams,
        });
    };

    if (isLoadingDiscovery) {
        return (
            <div className="flex flex-col items-center justify-center py-10">
                <Loader2 className="h-8 w-8 animate-spin text-indigo-600 mb-2" />
                <p className="text-sm text-slate-500 font-medium tracking-wide">Initializing generators...</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <Form {...form}>
                <form onSubmit={form.handleSubmit(handleFormSubmit)} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <FormField
                            control={form.control}
                            name="dataset_name"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-xs font-bold uppercase text-slate-500">Dataset Identification</FormLabel>
                                    <FormControl>
                                        <Input {...field} placeholder="e.g. baseline_exp_01" className="bg-white" />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />

                        <FormField
                            control={form.control}
                            name="generator_type"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-xs font-bold uppercase text-slate-500">Engine Type</FormLabel>
                                    <Select
                                        onValueChange={handleGeneratorChange}
                                        value={field.value || ""}
                                    >
                                        <FormControl>
                                            <SelectTrigger className="bg-white">
                                                <SelectValue placeholder="Select generator" />
                                            </SelectTrigger>
                                        </FormControl>
                                        <SelectContent>
                                            {generators.map((gen) => (
                                                <SelectItem key={gen.type} value={gen.type}>
                                                    {gen.name}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                    </div>

                    {selectedGenerator && (
                        <div className="space-y-4 pt-4 border-t border-slate-100">
                            <div className="flex items-center gap-2 mb-2">
                                <Wand2 className="h-4 w-4 text-indigo-500" />
                                <h4 className="text-sm font-bold text-slate-700 uppercase tracking-tight">
                                    {selectedGenerator.name} Configuration
                                </h4>
                            </div>
                            <DynamicConfigForm
                                parameters={selectedGenerator.parameters}
                                onChange={(vals, valid) => {
                                    setDynamicParams(vals);
                                    setIsDynamicValid(valid);
                                }}
                            />
                        </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-100">
                        <FormField
                            control={form.control}
                            name="split_ratio"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-xs font-bold uppercase text-slate-500">Test Split Ratio (0.0 - 0.5)</FormLabel>
                                    <FormControl>
                                        <Input {...field} type="number" step="0.05" className="bg-white" />
                                    </FormControl>
                                    <FormDescription className="text-[10px] italic">Ratio of data reserved for testing.</FormDescription>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="random_state"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-xs font-bold uppercase text-slate-500">Random Seed</FormLabel>
                                    <FormControl>
                                        <Input {...field} type="number" className="bg-white" />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                    </div>

                    <Button
                        type="submit"
                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold h-11 shadow-lg shadow-indigo-200 transition-all active:scale-[0.98]"
                        disabled={isLoading || !isDynamicValid}
                    >
                        {isLoading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Synthesizing...
                            </>
                        ) : (
                            "Start Generation Pipeline"
                        )}
                    </Button>
                </form>
            </Form>
        </div>
    );
}
