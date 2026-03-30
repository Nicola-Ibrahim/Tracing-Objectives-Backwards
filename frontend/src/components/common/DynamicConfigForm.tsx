"use client";

import React, { useEffect, useState } from "react";
import { ParameterDefinition } from "@/features/dataset/types";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { InfoIcon, Brackets } from "lucide-react";

interface DynamicConfigFormProps {
    parameters: ParameterDefinition[];
    onChange: (values: Record<string, unknown>, isValid: boolean) => void;
    className?: string;
}

export function DynamicConfigForm({
    parameters,
    onChange,
    className = "",
}: DynamicConfigFormProps) {
    const [values, setValues] = useState<Record<string, unknown>>(() => {
        const initialValues: Record<string, unknown> = {};
        parameters.forEach((param) => {
            const defaultValue = param.default ?? "";
            if (param.type === "bool" || param.type === "boolean") {
                initialValues[param.name] = param.default === true;
            } else {
                initialValues[param.name] = defaultValue;
            }
        });
        return initialValues;
    });

    useEffect(() => {
        const isValid = parameters.every(p => 
            !p.required || (values[p.name] !== "" && values[p.name] !== null && values[p.name] !== undefined)
        );
        onChange(values, isValid);
    }, [parameters, onChange, values]);

    const checkValidity = (currentValues: Record<string, unknown>) => {
        return parameters.every(p => {
            if (!p.required) return true;
            const val = currentValues[p.name];
            return val !== "" && val !== null && val !== undefined;
        });
    };

    const handleValueChange = (name: string, value: unknown) => {
        const newValues = { ...values, [name]: value };
        setValues(newValues);
        const isValid = checkValidity(newValues);
        onChange(newValues, isValid);
    };

    const renderField = (param: ParameterDefinition) => {
        const { name, type, options, required } = param;
        const value = values[name];

        // 1. Enum / Options (Select)
        if (options && options.length > 0) {
            return (
                <Select
                    value={value !== undefined && value !== null ? String(value) : ""}
                    onValueChange={(val) => handleValueChange(name, val)}
                >
                    <SelectTrigger className="w-full bg-background border-border text-foreground font-medium rounded-[1rem] focus:ring-2 focus:ring-indigo-500/10 transition-all">
                        <SelectValue placeholder={`Select ${name}`} />
                    </SelectTrigger>
                    <SelectContent className="bg-popover border-border text-popover-foreground rounded-[1rem]">
                        {options.map((opt) => (
                            <SelectItem key={String(opt)} value={String(opt)} className="hover:bg-muted focus:bg-muted cursor-pointer transition-colors">
                                {String(opt)}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            );
        }

        // 2. Boolean (Checkbox)
        if (type === "bool" || type === "boolean") {
            return (
                <div className="flex items-center space-x-3 py-2 bg-muted/10 px-3 rounded-[1rem] border border-border/50 hover:bg-muted/20 transition-all group">
                    <Checkbox
                        id={`field-${name}`}
                        checked={!!value}
                        onCheckedChange={(checked) => handleValueChange(name, !!checked)}
                        className="border-border data-[state=checked]:bg-indigo-500 data-[state=checked]:border-indigo-500 transition-all"
                    />
                    <Label htmlFor={`field-${name}`} className="text-[10px] font-bold uppercase text-muted-foreground tracking-widest cursor-pointer opacity-70 group-hover:opacity-100 transition-opacity font-heading">
                        {required ? `${name}*` : name}
                    </Label>
                </div>
            );
        }

        // 3. Number (Input type="number")
        if (["int", "float", "number", "integer"].includes(type.toLowerCase())) {
            return (
                <Input
                    type="number"
                    step={type.toLowerCase() === "float" ? "any" : "1"}
                    placeholder={`Enter ${name}`}
                    value={value as string | number | undefined}
                    className="bg-background border-border text-foreground transition-all focus:ring-2 focus:ring-indigo-500/10 focus:border-indigo-500/50 rounded-[1rem]"
                    onChange={(e) => {
                        const val = type.toLowerCase() === "float" ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
                        handleValueChange(name, isNaN(val) ? "" : val);
                    }}
                    required={required}
                />
            );
        }

        // 4. Complex (Dict/List) Fallback to Textarea
        if (["dict", "list", "object", "array"].includes(type.toLowerCase())) {
            return (
                <div className="relative group">
                    <Textarea
                        placeholder={`Enter JSON for ${name}`}
                        value={typeof value === 'object' ? JSON.stringify(value) : ((value as string) || "")}
                        className="font-mono text-xs bg-muted/20 border-border text-foreground min-h-[100px] focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/10 rounded-[1rem] transition-all"
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
                            const raw = e.target.value;
                            try {
                                if (raw.trim() === "") {
                                    handleValueChange(name, null);
                                } else {
                                    const parsed = JSON.parse(raw);
                                    handleValueChange(name, parsed);
                                }
                            } catch {
                                handleValueChange(name, raw);
                            }
                        }}
                        required={required}
                    />
                    <Brackets className="absolute right-3 bottom-3 size-4 text-muted-foreground/30 pointer-events-none group-hover:text-indigo-500/30 transition-colors" />
                </div>
            );
        }

        // 5. Default (Text Input)
        return (
            <Input
                type="text"
                placeholder={`Enter ${name}`}
                value={(value as string) || ""}
                className="bg-background border-border text-foreground transition-all focus:ring-2 focus:ring-indigo-500/10 focus:border-indigo-500/50 rounded-[1rem]"
                onChange={(e) => handleValueChange(name, e.target.value)}
                required={required}
            />
        );
    };

    return (
        <div className={`grid grid-cols-1 md:grid-cols-2 gap-4 ${className}`}>
            {parameters.map((param) => (
                <div key={param.name} className={`space-y-2 ${["dict", "list", "object", "array"].includes(param.type.toLowerCase()) ? "md:col-span-2" : ""}`}>
                    {param.type !== "bool" && param.type !== "boolean" && (
                        <div className="flex items-center gap-1.5 ml-0.5">
                            <Label htmlFor={`field-${param.name}`} className="text-[10px] font-bold uppercase text-muted-foreground/80 tracking-widest font-heading">
                                {param.name} {param.required && <span className="text-destructive">*</span>}
                            </Label>
                            {param.description && (
                                <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <div className="hover:bg-muted p-0.5 rounded-full transition-colors cursor-help">
                                                <InfoIcon className="size-3 text-muted-foreground/40 hover:text-indigo-500 transition-colors" />
                                            </div>
                                        </TooltipTrigger>
                                        <TooltipContent className="max-w-[200px] text-[10px] bg-popover text-popover-foreground border border-border rounded-[1rem] font-medium p-3">
                                            <p>{param.description}</p>
                                        </TooltipContent>
                                    </Tooltip>
                                </TooltipProvider>
                            )}
                        </div>
                    )}
                    {renderField(param)}
                </div>
            ))}
        </div>
    );
}
