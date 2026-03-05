"use client";

import React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
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
import { Loader2, PlusCircle } from "lucide-react";

const generateDatasetSchema = z.object({
    dataset_name: z.string().min(3, "Name must be at least 3 characters"),
    function_id: z.coerce.number().min(1).max(24),
    n_var: z.coerce.number().min(2).max(10),
    population_size: z.coerce.number().min(10).max(1000),
    generations: z.coerce.number().min(1).max(100),
    split_ratio: z.coerce.number().min(0).max(0.5),
});

type GenerateDatasetFormValues = z.infer<typeof generateDatasetSchema>;

interface GenerateDatasetFormProps {
    onSubmit: (values: GenerateDatasetFormValues) => void;
    isLoading?: boolean;
}

export function GenerateDatasetForm({ onSubmit, isLoading = false }: GenerateDatasetFormProps) {
    const form = useForm<GenerateDatasetFormValues>({
        resolver: zodResolver(generateDatasetSchema) as any,
        defaultValues: {
            dataset_name: "",
            function_id: 1,
            n_var: 5,
            population_size: 100,
            generations: 20,
            split_ratio: 0.2,
        },
    });

    return (
        <Card className="border-slate-200 shadow-sm bg-white">
            <CardHeader className="bg-slate-50/50 border-b border-slate-100">
                <div className="flex items-center gap-2">
                    <PlusCircle className="h-5 w-5 text-indigo-600" />
                    <CardTitle className="text-lg font-bold text-slate-800">Generate Dataset</CardTitle>
                </div>
                <CardDescription>Configure synthetic dataset parameters (COCOEX benchmark)</CardDescription>
            </CardHeader>
            <CardContent className="p-6">
                <Form {...form}>
                    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                        <FormField
                            control={form.control}
                            name="dataset_name"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-xs font-bold uppercase text-slate-500">Dataset Name</FormLabel>
                                    <FormControl>
                                        <Input {...field} placeholder="e.g. baseline_exp_01" className="bg-white" />
                                    </FormControl>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />

                        <div className="grid grid-cols-2 gap-4">
                            <FormField
                                control={form.control}
                                name="function_id"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel className="text-xs font-bold uppercase text-slate-500">Function ID (1-24)</FormLabel>
                                        <FormControl>
                                            <Input {...field} type="number" className="bg-white" />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="n_var"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel className="text-xs font-bold uppercase text-slate-500">Decision Dims (n_var)</FormLabel>
                                        <FormControl>
                                            <Input {...field} type="number" className="bg-white" />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <FormField
                                control={form.control}
                                name="population_size"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel className="text-xs font-bold uppercase text-slate-500">Pop Size</FormLabel>
                                        <FormControl>
                                            <Input {...field} type="number" className="bg-white" />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="generations"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel className="text-xs font-bold uppercase text-slate-500">Generations</FormLabel>
                                        <FormControl>
                                            <Input {...field} type="number" className="bg-white" />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                        </div>

                        <FormField
                            control={form.control}
                            name="split_ratio"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel className="text-xs font-bold uppercase text-slate-500">Test Split Ratio (0.0 - 0.5)</FormLabel>
                                    <FormControl>
                                        <Input {...field} type="number" step="0.05" className="bg-white" />
                                    </FormControl>
                                    <FormDescription className="text-[10px] italic">Ratio of data reserved for testing (holdout set).</FormDescription>
                                    <FormMessage />
                                </FormItem>
                            )}
                        />

                        <Button
                            type="submit"
                            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold h-11"
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Generating...
                                </>
                            ) : (
                                "Start Generation Pipeline"
                            )}
                        </Button>
                    </form>
                </Form>
            </CardContent>
        </Card>
    );
}
