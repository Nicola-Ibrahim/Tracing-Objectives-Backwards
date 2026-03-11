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
    DialogTitle,
    DialogHeader,
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

        const defaultParams = meta.parameters.reduce((acc, param) => {
            acc[param.name] = param.default;
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
                    <span className="text-[10px] uppercase font-black text-muted-foreground/60 tracking-widest">{title}</span>
                </div>
            )}
            <div className="space-y-3">
                {chain.map((step, idx) => {
                    const meta = transformers.find((t) => t.type === step.type);

                    return (
                        <div
                            key={idx}
                            className="flex flex-col p-4 bg-background border border-border rounded-2xl shadow-sm hover:border-indigo-500/30 transition-all group relative overflow-hidden"
                        >
                            <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500/20 group-hover:bg-indigo-500 transition-colors" />
                            <div className="flex items-center gap-4">
                                <div className="h-7 w-7 rounded-lg bg-muted flex items-center justify-center text-[10px] font-black text-muted-foreground shrink-0 border border-border/50 group-hover:bg-indigo-500 group-hover:text-white transition-all">
                                    {idx + 1}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs font-black text-foreground truncate uppercase tracking-tight group-hover:text-indigo-500 transition-colors">
                                        {step.type.replace("_", " ")}
                                    </p>
                                </div>

                                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-all translate-x-2 group-hover:translate-x-0">
                                    {meta && (
                                        <Dialog>
                                            <DialogTrigger asChild>
                                                <Button variant="ghost" size="icon" className="h-8 w-8 rounded-xl text-muted-foreground hover:text-indigo-500 hover:bg-indigo-500/10 border border-transparent hover:border-indigo-500/20 transition-all">
                                                    <Settings2 className="h-4 w-4" />
                                                </Button>
                                            </DialogTrigger>
                                            <DialogContent className="sm:max-w-md p-0 rounded-3xl shadow-2xl border-border bg-card overflow-hidden">
                                                <div className="space-y-0">
                                                    <DialogHeader className="flex flex-row items-center gap-4 bg-muted/30 p-6 border-b border-border space-y-0 text-left">
                                                        <div className="h-12 w-12 rounded-2xl bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                                                            <Settings2 className="h-6 w-6 text-indigo-500" />
                                                        </div>
                                                        <div className="min-w-0">
                                                            <DialogTitle className="text-sm font-black text-foreground uppercase tracking-tight truncate">
                                                                {step.type.replace("_", " ")}
                                                            </DialogTitle>
                                                            <p className="text-[10px] text-muted-foreground font-black uppercase tracking-widest opacity-60 mt-1">Configure transformation parameters</p>
                                                        </div>
                                                    </DialogHeader>
                                                    <div className="p-8">
                                                        <TransformationForm
                                                            metadata={meta}
                                                            params={step.params}
                                                            onChange={(params) => updateStepParams(idx, params)}
                                                        />
                                                    </div>
                                                </div>
                                            </DialogContent>
                                        </Dialog>
                                    )}
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => removeStep(idx)}
                                        className="h-8 w-8 rounded-xl text-muted-foreground hover:text-rose-500 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition-all"
                                    >
                                        <Trash2 className="h-4 w-4" />
                                    </Button>
                                </div>
                            </div>
                        </div>
                    );
                })}

                {chain.length === 0 && (
                    <div className="py-12 border-2 border-dashed border-border rounded-3xl flex flex-col items-center justify-center text-center bg-muted/5 group hover:bg-muted/10 transition-colors">
                        <Plus className="h-5 w-5 text-muted-foreground/20 mb-3 group-hover:scale-110 transition-transform" />
                        <p className="text-[10px] font-black text-muted-foreground/40 uppercase tracking-widest">No transformations added yet.</p>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-2 gap-3 pt-2">
                {transformers.map((t) => (
                    <Button
                        key={t.type}
                        variant="outline"
                        size="sm"
                        onClick={() => addStep(t.type)}
                        className="justify-start gap-3 h-10 border-border bg-background text-muted-foreground hover:text-indigo-500 hover:border-indigo-500/50 hover:bg-indigo-500/5 rounded-2xl px-4 transition-all group"
                    >
                        <Plus className="h-3.5 w-3.5 group-hover:rotate-90 transition-transform" />
                        <span className="text-[10px] font-black uppercase tracking-widest truncate">{t.name}</span>
                    </Button>
                ))}
            </div>
        </div>
    );
}
