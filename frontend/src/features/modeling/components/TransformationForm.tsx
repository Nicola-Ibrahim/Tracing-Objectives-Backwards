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
        <div className="space-y-4">
            {metadata.parameters.map((param) => {
                const { name, type, description, default: defaultValue } = param;
                const value = params[name] !== undefined ? params[name] : (defaultValue ?? "");

                return (
                    <div key={name} className="space-y-2">
                        <div className="flex justify-between items-center">
                            <Label htmlFor={name} className="text-xs font-bold uppercase tracking-wider text-slate-500">
                                {name.replace("_", " ")}
                            </Label>
                            {description && (
                                <span className="text-[10px] text-slate-400 italic">{description}</span>
                            )}
                        </div>

                        {type === "number" || type === "float" || type === "int" ? (
                            <Input
                                id={name}
                                type="number"
                                step={type === "int" ? "1" : "0.1"}
                                value={value ?? 0}
                                onChange={(e) => handleParamChange(name, parseFloat(e.target.value))}
                                className="bg-slate-50 border-slate-200"
                            />
                        ) : type === "list" || name === "feature_range" ? (
                            <div className="grid grid-cols-2 gap-2">
                                <Input
                                    type="number"
                                    value={Array.isArray(value) ? (value[0] ?? 0) : 0}
                                    onChange={(e) => {
                                        const newVal = [...(Array.isArray(value) ? value : [0, 1])];
                                        newVal[0] = parseFloat(e.target.value);
                                        handleParamChange(name, newVal);
                                    }}
                                    className="bg-slate-50 border-slate-200"
                                    placeholder="Min"
                                />
                                <Input
                                    type="number"
                                    value={Array.isArray(value) ? (value[1] ?? 1) : 1}
                                    onChange={(e) => {
                                        const newVal = [...(Array.isArray(value) ? value : [0, 1])];
                                        newVal[1] = parseFloat(e.target.value);
                                        handleParamChange(name, newVal);
                                    }}
                                    className="bg-slate-50 border-slate-200"
                                    placeholder="Max"
                                />
                            </div>
                        ) : (
                            <Input
                                id={name}
                                value={value ?? ""}
                                onChange={(e) => handleParamChange(name, e.target.value)}
                                className="bg-slate-50 border-slate-200"
                            />
                        )}
                    </div>
                );
            })}
        </div>
    );
}
