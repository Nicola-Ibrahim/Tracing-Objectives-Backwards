"use client";

import { useEffect, useState } from "react";
import { fetchDatasets, fetchDatasetData } from "@/lib/apiClient";
import { DatasetDetailResponse, Vector } from "@/types/api";
import ChartJSWrapper, { ChartDataset } from "@/components/ChartJSWrapper";
import { Card, Button } from "@/components/ui";
import { useToast } from "@/components/ui/ToastContext";

export default function ExplorationPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [data, setData] = useState<DatasetDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const { showToast } = useToast();

  useEffect(() => {
    setLoading(true);
    fetchDatasets()
      .then(setDatasets)
      .catch((err) => showToast(err.message, "error"))
      .finally(() => setLoading(false));
  }, [showToast]);

  const handleDatasetSelect = async (name: string) => {
    setSelectedDataset(name);
    setLoading(true);
    setData(null);
    try {
      const response = await fetchDatasetData(name);
      setData(response);
      showToast(`Loaded data for ${name}`, "success");
    } catch (err: any) {
      showToast(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  // Prepare Objective Space Plot Data
  const objectiveDatasets: ChartDataset[] = [];
  const decisionDatasets: ChartDataset[] = [];

  if (data) {
    const objBaseline: any[] = [];
    const objPareto: any[] = [];
    const decBaseline: any[] = [];
    const decPareto: any[] = [];

    data.y.forEach((o: Vector, i: number) => {
      const isPareto = data.is_pareto?.[i];
      if (isPareto) {
        objPareto.push({ x: o[0], y: o[1] });
        decPareto.push({ x: data.X[i][0], y: data.X[i][1] });
      } else {
        objBaseline.push({ x: o[0], y: o[1] });
        decBaseline.push({ x: data.X[i][0], y: data.X[i][1] });
      }
    });

    objectiveDatasets.push({
      label: "Baseline",
      data: objBaseline,
      backgroundColor: "rgba(203, 213, 225, 0.4)",
    });
    if (objPareto.length > 0) {
      objectiveDatasets.push({
        label: "Pareto Front",
        data: objPareto,
        backgroundColor: "#10b981",
        pointRadius: 6,
      });
    }

    decisionDatasets.push({
      label: "Baseline Designs",
      data: decBaseline,
      backgroundColor: "rgba(148, 163, 184, 0.4)",
    });
    if (decPareto.length > 0) {
      decisionDatasets.push({
        label: "Pareto Optimal",
        data: decPareto,
        backgroundColor: "#059669",
        pointRadius: 6,
      });
    }
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      <Card title="Available Datasets">
        <div className="flex flex-wrap gap-2">
          {datasets.map((name) => (
            <Button
              key={name}
              variant={selectedDataset === name ? "primary" : "secondary"}
              onClick={() => handleDatasetSelect(name)}
              className="text-sm"
              isLoading={loading && selectedDataset === name}
            >
              {name}
            </Button>
          ))}
          {loading && datasets.length === 0 && <p className="text-slate-400 text-sm">Fetching datasets...</p>}
        </div>
      </Card>

      {data && (
        <Card className="animate-in fade-in slide-in-from-bottom-4 duration-500 overflow-hidden">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-2xl font-bold text-slate-800">{data.name}</h2>
              <p className="text-sm text-slate-500">Multiobjective Exploration</p>
            </div>
            <div className="px-4 py-2 bg-slate-50 rounded-xl border border-slate-100 flex items-center gap-6">
              <div className="text-center">
                <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Samples</p>
                <p className="text-lg font-bold text-slate-700">{data.y.length}</p>
              </div>
              <div className="w-px h-8 bg-slate-200" />
              <div className="text-center">
                <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Objectives</p>
                <p className="text-lg font-bold text-slate-700">2D</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="h-[400px]">
              <ChartJSWrapper
                title="Objective Space"
                datasets={objectiveDatasets}
                xAxisTitle="Objective 1"
                yAxisTitle="Objective 2"
              />
            </div>
            <div className="h-[400px]">
              <ChartJSWrapper
                title="Decision Space"
                datasets={decisionDatasets}
                xAxisTitle="Dimension 1"
                yAxisTitle="Dimension 2"
              />
            </div>
          </div>
        </Card>
      )}

      {loading && !selectedDataset && (
        <div className="flex justify-center py-20">
          <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        </div>
      )}
    </div>
  );
}
