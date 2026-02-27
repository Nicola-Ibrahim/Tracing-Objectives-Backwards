"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { fetchDatasets, fetchDatasetData } from "@/lib/apiClient";
import { DatasetDetailResponse, Vector } from "@/types/api";
import { useToast } from "@/components/ui/ToastContext";
import { useSearchParams, useRouter, usePathname } from "next/navigation";

interface DatasetContextType {
    datasets: string[];
    selectedDataset: string;
    baselineData: DatasetDetailResponse | null;
    isLoading: boolean;
    selectDataset: (name: string) => void;
    refreshDatasets: (selectName?: string) => Promise<void>;
    ranges: {
        objX: [number, number] | null;
        objY: [number, number] | null;
        decX: [number, number] | null;
        decY: [number, number] | null;
    };
    refreshDatasetData: () => Promise<void>;
}

const DatasetContext = createContext<DatasetContextType | undefined>(undefined);

export const DatasetProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [datasets, setDatasets] = useState<string[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<string>("");
    const [baselineData, setBaselineData] = useState<DatasetDetailResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [ranges, setRanges] = useState<DatasetContextType["ranges"]>({
        objX: null,
        objY: null,
        decX: null,
        decY: null,
    });

    const { showToast } = useToast();
    const searchParams = useSearchParams();
    const router = useRouter();
    const pathname = usePathname();

    const refreshDatasets = useCallback(async (selectName?: string) => {
        try {
            const data = await fetchDatasets();
            setDatasets(data);

            const urlDataset = searchParams.get("dataset");

            if (selectName) {
                setSelectedDataset(selectName);
            } else if (urlDataset && data.includes(urlDataset)) {
                setSelectedDataset(urlDataset);
            } else if (data.length > 0 && !selectedDataset) {
                setSelectedDataset(data[0]);
            }
        } catch (err: any) {
            showToast(err.message, "error");
        }
    }, [searchParams, selectedDataset, showToast]);

    const selectDataset = (name: string) => {
        setSelectedDataset(name);
        // Sync to URL
        const params = new URLSearchParams(searchParams);
        params.set("dataset", name);
        router.push(`${pathname}?${params.toString()}`);
    };

    useEffect(() => {
        refreshDatasets();
    }, []); // Initial load

    const refreshDatasetData = useCallback(async () => {
        if (!selectedDataset) return;
        setIsLoading(true);
        try {
            const data = await fetchDatasetData(selectedDataset);
            setBaselineData(data);
            // ... (rest of normalization logic)
            const padding = 0.05;
            const newRanges: DatasetContextType["ranges"] = {
                objX: null,
                objY: null,
                decX: null,
                decY: null,
            };

            if (data.bounds.obj_0 && data.bounds.obj_1) {
                const o0 = data.bounds.obj_0;
                const o1 = data.bounds.obj_1;
                const span0 = o0[1] - o0[0];
                const span1 = o1[1] - o1[0];
                newRanges.objX = [o0[0] - (span0 || 1) * padding, o0[1] + (span0 || 1) * padding];
                newRanges.objY = [o1[0] - (span1 || 1) * padding, o1[1] + (span1 || 1) * padding];
            }

            if (data.X.length > 0) {
                const x0 = data.X.map((v: Vector) => v[0]).filter(v => typeof v === 'number' && !isNaN(v));
                const x1 = data.X.map((v: Vector) => v[1]).filter(v => typeof v === 'number' && !isNaN(v));

                if (x0.length > 0 && x1.length > 0) {
                    const minX0 = Math.min(...x0);
                    const maxX0 = Math.max(...x0);
                    const minX1 = Math.min(...x1);
                    const maxX1 = Math.max(...x1);
                    const spanX0 = maxX0 - minX0;
                    const spanX1 = maxX1 - minX1;
                    newRanges.decX = [minX0 - (spanX0 || 1) * padding, maxX0 + (spanX0 || 1) * padding];
                    newRanges.decY = [minX1 - (spanX1 || 1) * padding, maxX1 + (spanX1 || 1) * padding];
                }
            }
            setRanges(newRanges);
        } catch (err: any) {
            showToast(err.message, "error");
        } finally {
            setIsLoading(false);
        }
    }, [selectedDataset, showToast]);

    useEffect(() => {
        refreshDatasetData();
    }, [selectedDataset, refreshDatasetData]);

    return (
        <DatasetContext.Provider
            value={{
                datasets,
                selectedDataset,
                baselineData,
                isLoading,
                selectDataset,
                refreshDatasets,
                refreshDatasetData,
                ranges,
            }}
        >
            {children}
        </DatasetContext.Provider>
    );
};

export const useDataset = () => {
    const context = useContext(DatasetContext);
    if (context === undefined) {
        throw new Error("useDataset must be used within a DatasetProvider");
    }
    return context;
};
