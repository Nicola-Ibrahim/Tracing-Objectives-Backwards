"use client";

import React from "react";
import { Plus, Trash2, GripVertical, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { TransformationStep, TransformerMetadata } from "../api";
import { TransformationForm } from "./TransformationForm";
import {
    Dialog,
    DialogContent,
    DialogTrigger,
} from "@/components/ui/dialog";

interface ChainBuilderProps {
    title?: string;
    chain: TransformationStep[];
    transformers: TransformerMetadata[];
    onChange: (chain: TransformationStep[]) => void;
}

export function ChainBuilder({ title, chain, transformers, onChange }: ChainBuilderProps) {
    const addStep = (type: string) => {
        const meta = transformers.find((t) => t.type === type);
        if (!meta) return;

        const defaultParams = Object.entries(meta.params).reduce((acc, [name, schema]) => {
            acc[name] = schema.default;
            return acc;
        }, {} as Record<string, any>);

        onChange([...chain, { type, params: defaultParams }]);
    };

    const removeStep = (index: number) => {
        const newChain = [...chain];
        newChain.splice(index, 1);
        onChange(newChain);
    };

    const updateStepParams = (index: number, params: Record<string, any>) => {
        const newChain = [...chain];
        newChain[index] = { ...newChain[index], params };
        onChange(newChain);
    };

    return (
        <div className="space-y-4">
            {title && (
                <div className="flex items-center gap-2 mb-2">
                    <span className="text-[10px] uppercase font-bold text-slate-400 tracking-widest">{title}</span>
                </div>
            )}
            <div className="space-y-3">
                {chain.map((step, idx) => {
                    const meta = transformers.find((t) => t.type === step.type);

                    return (
                        <div
                            key={idx}
                            className="flex flex-col p-3 bg-white border border-slate-200 rounded-2xl shadow-sm hover:border-indigo-200 transition-colors group"
                        >
                            <div className="flex items-center gap-3">
                                <div className="h-6 w-6 rounded-full bg-slate-100 flex items-center justify-center text-[10px] font-bold text-slate-500 shrink-0">
                                    {idx + 1}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs font-bold text-slate-700 truncate capitalize">
                                        {step.type.replace("_", " ")}
                                    </p>
                                </div>

                                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                    {meta && (
                                        <Dialog>
                                            <DialogTrigger asChild>
                                                <Button variant="ghost" size="icon" className="h-7 w-7 rounded-lg text-slate-400 hover:text-indigo-600 hover:bg-indigo-50">
                                                    <Settings2 className="h-3.5 w-3.5" />
                                                </Button>
                                            </DialogTrigger>
                                            <DialogContent className="sm:max-w-md p-6 rounded-3xl shadow-2xl border-slate-200">
                                                <div className="space-y-6">
                                                    <div className="flex items-center gap-3 border-b border-slate-100 pb-4">
                                                        <div className="h-10 w-10 rounded-2xl bg-indigo-50 flex items-center justify-center">
                                                            <Settings2 className="h-5 w-5 text-indigo-500" />
                                                        </div>
                                                        <div>
                                                            <h4 className="text-sm font-bold text-slate-900 capitalize">{step.type.replace("_", " ")}</h4>
                                                            <p className="text-[10px] text-slate-400 font-medium">Configure transformation parameters</p>
                                                        </div>
                                                    </div>
                                                    <TransformationForm
                                                        metadata={meta}
                                                        params={step.params}
                                                        onChange={(params) => updateStepParams(idx, params)}
                                                    />
                                                </div>
                                            </DialogContent>
                                        </Dialog>
                                    )}
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => removeStep(idx)}
                                        className="h-7 w-7 rounded-lg text-slate-400 hover:text-red-600 hover:bg-red-50"
                                    >
                                        <Trash2 className="h-3.5 w-3.5" />
                                    </Button>
                                </div>
                            </div>
                        </div>
                    );
                })}

                {chain.length === 0 && (
                    <div className="py-8 border-2 border-dashed border-slate-100 rounded-3xl flex flex-col items-center justify-center text-center">
                        <p className="text-xs font-medium text-slate-400">No transformations added yet.</p>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-2 gap-2">
                {transformers.map((t) => (
                    <Button
                        key={t.type}
                        variant="outline"
                        size="sm"
                        onClick={() => addStep(t.type)}
                        className="justify-start gap-2 h-9 border-slate-200 text-slate-600 hover:text-indigo-600 hover:border-indigo-200 hover:bg-indigo-50/50 rounded-xl px-3"
                    >
                        <Plus className="h-3 w-3" />
                        <span className="text-[10px] font-bold uppercase tracking-tight truncate">{t.name}</span>
                    </Button>
                ))}
            </div>
        </div>
    );
}
