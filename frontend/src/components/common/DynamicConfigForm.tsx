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
    onChange: (values: Record<string, any>, isValid: boolean) => void;
    className?: string;
}

export function DynamicConfigForm({
    parameters,
    onChange,
    className = "",
}: DynamicConfigFormProps) {
    const [values, setValues] = useState<Record<string, any>>({});

    // Initialize values from defaults
    useEffect(() => {
        const initialValues: Record<string, any> = {};
        parameters.forEach((param) => {
            initialValues[param.name] = param.default !== null ? param.default : "";

            // Special handling for booleans
            if (param.type === "bool" || param.type === "boolean") {
                initialValues[param.name] = !!param.default;
            }
        });
        setValues(initialValues);

        // Initial validity check
        const isValid = parameters.every(p => !p.required || (initialValues[p.name] !== "" && initialValues[p.name] !== null));
        onChange(initialValues, isValid);
    }, [parameters]);

    const checkValidity = (currentValues: Record<string, any>) => {
        return parameters.every(p => {
            if (!p.required) return true;
            const val = currentValues[p.name];
            return val !== "" && val !== null && val !== undefined;
        });
    };

    const handleValueChange = (name: string, value: any) => {
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
                    value={String(value)}
                    onValueChange={(val) => handleValueChange(name, val)}
                >
                    <SelectTrigger className="w-full bg-white">
                        <SelectValue placeholder={`Select ${name}`} />
                    </SelectTrigger>
                    <SelectContent>
                        {options.map((opt) => (
                            <SelectItem key={String(opt)} value={String(opt)}>
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
                <div className="flex items-center space-x-2 py-2">
                    <Checkbox
                        id={`field-${name}`}
                        checked={!!value}
                        onCheckedChange={(checked) => handleValueChange(name, !!checked)}
                    />
                    <Label htmlFor={`field-${name}`} className="text-xs font-bold uppercase text-slate-500 cursor-pointer">
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
                    value={value}
                    className="bg-white"
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
                <div className="relative">
                    <Textarea
                        placeholder={`Enter JSON for ${name}`}
                        value={typeof value === 'object' ? JSON.stringify(value) : (value || "")}
                        className="font-mono text-xs bg-slate-50/50 min-h-[80px]"
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
                            const raw = e.target.value;
                            try {
                                if (raw.trim() === "") {
                                    handleValueChange(name, null);
                                } else {
                                    // We try to parse it as JSON, if it fails we keep it as string
                                    const parsed = JSON.parse(raw);
                                    handleValueChange(name, parsed);
                                }
                            } catch (err) {
                                handleValueChange(name, raw);
                            }
                        }}
                        required={required}
                    />
                    <Brackets className="absolute right-2 bottom-2 size-3 text-slate-300 pointer-events-none" />
                </div>
            );
        }

        // 5. Default (Text Input)
        return (
            <Input
                type="text"
                placeholder={`Enter ${name}`}
                value={value || ""}
                className="bg-white"
                onChange={(e) => handleValueChange(name, e.target.value)}
                required={required}
            />
        );
    };

    return (
        <div className={`grid grid-cols-1 md:grid-cols-2 gap-4 ${className}`}>
            {parameters.map((param) => (
                <div key={param.name} className={`space-y-1.5 ${["dict", "list", "object", "array"].includes(param.type.toLowerCase()) ? "md:col-span-2" : ""}`}>
                    {param.type !== "bool" && param.type !== "boolean" && (
                        <div className="flex items-center gap-1.5 ml-0.5">
                            <Label htmlFor={`field-${param.name}`} className="text-[10px] font-black uppercase text-slate-400 tracking-tighter">
                                {param.name} {param.required && <span className="text-destructive">*</span>}
                            </Label>
                            {param.description && (
                                <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <InfoIcon className="size-3 text-slate-300 hover:text-indigo-400 cursor-help transition-colors" />
                                        </TooltipTrigger>
                                        <TooltipContent className="max-w-[200px] text-[10px] bg-slate-900 text-white border-0">
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
