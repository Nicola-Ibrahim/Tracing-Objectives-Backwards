"use client";

import React, { useEffect, useState } from "react";
import { useForm, Resolver, useWatch } from "react-hook-form";
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
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Loader2, Wand2 } from "lucide-react";
import { DatasetGenerationRequest } from "../types";

const generateDatasetSchema = z.object({
    dataset_name: z.string().min(3, "Name must be at least 3 characters"),
    generator_type: z.string().min(1, "Generator type is required"),
    split_ratio: z.coerce.number().min(0).max(0.99).default(0.2),
    random_state: z.coerce.number().default(42),
});

type GenerateDatasetFormValues = z.infer<typeof generateDatasetSchema>;

interface GenerateDatasetFormProps {
    onSubmit: (values: DatasetGenerationRequest) => void;
    isLoading?: boolean;
}

export function GenerateDatasetForm({ onSubmit, isLoading = false }: GenerateDatasetFormProps) {
    const [dynamicParams, setDynamicParams] = useState<Record<string, unknown>>({});
    const [isDynamicValid, setIsDynamicValid] = useState(true);

    const { data: discovery, isLoading: isLoadingDiscovery } = useQuery({
        queryKey: ["generators-discovery"],
        queryFn: getGenerators,
    });

    const form = useForm<GenerateDatasetFormValues>({
        resolver: zodResolver(generateDatasetSchema) as Resolver<GenerateDatasetFormValues>,
        defaultValues: {
            dataset_name: "",
            generator_type: "coco_pymoo",
            split_ratio: 0.2,
            random_state: 42,
        },
    });

    const generators = React.useMemo(() => discovery?.generators || [], [discovery?.generators]);
    const generatorType = useWatch({
        control: form.control,
        name: "generator_type",
        defaultValue: "coco_pymoo"
    });

    const selectedGenerator = React.useMemo(() => 
        generators.find(g => g.type === generatorType) || null,
    [generators, generatorType]);

    useEffect(() => {
        if (generators.length > 0 && !generatorType) {
            const defaultGen = generators.find(g => g.type === "coco_pymoo") || generators[0];
            form.setValue("generator_type", defaultGen.type);
        }
    }, [generators, form, generatorType]);

    const handleGeneratorChange = (genId: string) => {
        form.setValue("generator_type", genId);
    };

    const handleFormSubmit = (values: GenerateDatasetFormValues) => {
        onSubmit({
            ...values,
            params: dynamicParams,
        } as DatasetGenerationRequest);
    };

    if (isLoadingDiscovery) {
        return (
            <div className="flex flex-col items-center justify-center py-10">
                <Loader2 className="h-8 w-8 animate-spin text-indigo-500 mb-2" />
                <p className="text-sm text-muted-foreground font-bold uppercase tracking-widest font-heading">Initializing generators...</p>
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
                                    <FormLabel className="text-[10px] font-bold uppercase text-muted-foreground tracking-widest font-heading">Dataset Identification</FormLabel>
                                    <FormControl>
                                        <Input {...field} placeholder="e.g. baseline_exp_01" className="bg-background border-border text-foreground h-11" />
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
                                    <FormLabel className="text-[10px] font-bold uppercase text-muted-foreground tracking-widest font-heading">Engine Type</FormLabel>
                                    <Select
                                        onValueChange={handleGeneratorChange}
                                        value={field.value || ""}
                                    >
                                        <FormControl>
                                            <SelectTrigger className="bg-background border-border text-foreground h-11">
                                                <SelectValue placeholder="Select generator" />
                                            </SelectTrigger>
                                        </FormControl>
                                        <SelectContent className="bg-popover border-border text-popover-foreground">
                                            {generators.map((gen) => (
                                                <SelectItem key={gen.type} value={gen.type} className="hover:bg-muted focus:bg-muted cursor-pointer font-medium">
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
                        <div className="space-y-6 pt-6 border-t border-border">
                            <div className="flex items-center gap-2 mb-2">
                                <Wand2 className="h-4 w-4 text-indigo-500" />
                                <h4 className="text-[10px] font-bold text-foreground uppercase tracking-[0.2em] font-heading">
                                    {selectedGenerator.name} Configuration
                                </h4>
                            </div>
                            <DynamicConfigForm
                                key={selectedGenerator.type}
                                parameters={selectedGenerator.parameters}
                                onChange={(vals, valid) => {
                                    setDynamicParams(vals);
                                    setIsDynamicValid(valid);
                                }}
                            />
                        </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6 border-t border-border">
                        <FormField
                            control={form.control}
                            name="split_ratio"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-[10px] font-bold uppercase text-muted-foreground tracking-widest font-heading">Test Split Ratio (0.0 - 0.5)</FormLabel>
                                    <FormControl>
                                        <Input {...field} type="number" step="0.05" className="bg-background border-border text-foreground h-11" />
                                    </FormControl>
                                    <FormDescription className="text-[10px] font-medium italic opacity-60">Ratio of data reserved for testing.</FormDescription>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                        <FormField
                            control={form.control}
                            name="random_state"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-[10px] font-bold uppercase text-muted-foreground tracking-widest font-heading">Random Seed</FormLabel>
                                    <FormControl>
                                        <Input {...field} type="number" className="bg-background border-border text-foreground h-11" />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />
                    </div>

                    <Button
                        type="submit"
                        className="w-full bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-500 dark:hover:bg-indigo-600 text-white font-bold h-14 rounded-[1rem] transition-all active:scale-[0.98] mt-4 font-heading"
                        disabled={isLoading || !isDynamicValid}
                    >
                        {isLoading ? (
                            <>
                                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
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
