"use client";

import React from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { TransformerMetadata } from "../api";

interface TransformationFormProps {
    metadata: TransformerMetadata;
    params: Record<string, any>;
    onChange: (params: Record<string, any>) => void;
}

export function TransformationForm({ metadata, params, onChange }: TransformationFormProps) {
    const handleParamChange = (name: string, value: any) => {
        onChange({ ...params, [name]: value });
    };

    return (
        <div className="space-y-6">
            {metadata.parameters.map((param) => {
                const { name, type, description, default: defaultValue } = param;
                const value = params[name] !== undefined ? params[name] : (defaultValue ?? "");

                return (
                    <div key={name} className="space-y-3">
                        <div className="flex justify-between items-center group/label">
                            <Label htmlFor={name} className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted-foreground/60 transition-colors group-hover/label:text-indigo-500 font-heading">
                                {name.replace("_", " ")}
                            </Label>
                            {description && (
                                <span className="text-[10px] text-muted-foreground/40 font-medium italic select-none">{description}</span>
                            )}
                        </div>

                        {type === "number" || type === "float" || type === "int" ? (
                            <Input
                                id={name}
                                type="number"
                                step={type === "int" ? "1" : "0.1"}
                                value={value ?? 0}
                                onChange={(e) => handleParamChange(name, parseFloat(e.target.value))}
                                className="bg-background border-border text-foreground h-11 focus:ring-2 focus:ring-indigo-500/10 rounded-[1rem] transition-all"
                            />
                        ) : type === "list" || name === "feature_range" ? (
                            <div className="grid grid-cols-2 gap-3">
                                <div className="relative">
                                    <Input
                                        type="number"
                                        value={Array.isArray(value) ? (value[0] ?? 0) : 0}
                                        onChange={(e) => {
                                            const newVal = [...(Array.isArray(value) ? value : [0, 1])];
                                            newVal[0] = parseFloat(e.target.value);
                                            handleParamChange(name, newVal);
                                        }}
                                        className="bg-background border-border text-foreground h-11 rounded-[1rem] pl-9"
                                    />
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[9px] font-bold text-muted-foreground/40 uppercase font-heading">Min</span>
                                </div>
                                <div className="relative">
                                    <Input
                                        type="number"
                                        value={Array.isArray(value) ? (value[1] ?? 1) : 1}
                                        onChange={(e) => {
                                            const newVal = [...(Array.isArray(value) ? value : [0, 1])];
                                            newVal[1] = parseFloat(e.target.value);
                                            handleParamChange(name, newVal);
                                        }}
                                        className="bg-background border-border text-foreground h-11 rounded-[1rem] pl-9"
                                    />
                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[9px] font-bold text-muted-foreground/40 uppercase font-heading">Max</span>
                                </div>
                            </div>
                        ) : (
                            <Input
                                id={name}
                                value={value ?? ""}
                                onChange={(e) => handleParamChange(name, e.target.value)}
                                className="bg-background border-border text-foreground h-11 rounded-[1rem] focus:ring-2 focus:ring-indigo-500/10 transition-all"
                            />
                        )}
                    </div>
                );
            })}
        </div>
    );
}
