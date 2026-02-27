"use client";

import { useEffect, useState } from "react";
import { fetchDatasets, fetchDatasetContext } from "@/lib/apiClient";
import { Dataset } from "@/types/api";
import PlotlyWrapper from "@/components/PlotlyWrapper";
import { Card, Button } from "@/components/ui";
import { useToast } from "@/components/ui/ToastContext";

export default function ExplorationPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [context, setContext] = useState<Dataset | null>(null);
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
    setContext(null);
    try {
      const data = await fetchDatasetContext(name);
      setContext(data);
      showToast(`Loaded context for ${name}`, "success");
    } catch (err: any) {
      showToast(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const plotData: any[] = context
    ? [
      {
        x: context.original_objectives.map((o) => o[0]),
        y: context.original_objectives.map((o) => o[1]),
        mode: "markers",
        type: "scatter",
        name: "Original Objectives",
        marker: {
          color: "#6366f1",
          size: 8,
          opacity: 0.6,
        },
      },
    ]
    : [];

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

      {context && (
        <Card className="animate-in fade-in slide-in-from-bottom-4 duration-500 overflow-hidden">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-2xl font-bold text-slate-800">{context.name}</h2>
              <p className="text-sm text-slate-500">Objective Space Visualization</p>
            </div>
            <div className="px-4 py-2 bg-slate-50 rounded-xl border border-slate-100 flex items-center gap-6">
              <div className="text-center">
                <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Samples</p>
                <p className="text-lg font-bold text-slate-700">{context.original_objectives.length}</p>
              </div>
              <div className="w-px h-8 bg-slate-200" />
              <div className="text-center">
                <p className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Objectives</p>
                <p className="text-lg font-bold text-slate-700">2D</p>
              </div>
            </div>
          </div>

          <div className="w-full h-[500px] bg-slate-50/50 rounded-2xl border border-slate-100 p-4">
            <PlotlyWrapper
              data={plotData}
              layout={{
                xaxis: { title: { text: "Objective 1" }, gridcolor: "#f1f5f9", zeroline: false },
                yaxis: { title: { text: "Objective 2" }, gridcolor: "#f1f5f9", zeroline: false },
              }}
            />
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
