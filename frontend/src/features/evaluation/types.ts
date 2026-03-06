export interface EngineCandidate {
  solver_type: string;
  version?: number;
}

export interface DiagnoseRequest {
  dataset_name: string;
  candidates: EngineCandidate[];
  num_samples?: number;
  scale_method: "sd" | "mad" | "iqr";
}

export interface MetricSeries {
  x: number[];
  y: number[];
}

export interface MetricPlotData {
  [engineLabel: string]: MetricSeries;
}

export interface DiagnoseResponse {
  dataset_name: string;
  engines: string[];
  ecdf: MetricPlotData;
  pit: MetricPlotData;
  mace: Record<string, number>;
  warnings: string[];
}

export interface PerformanceRequest {
  dataset_name: string;
  engine: EngineCandidate;
  n_samples?: number;
}

export interface PerformanceResponse {
  dataset_name: string;
  solver_type: string;
  version: number;
  insights: Record<string, any>;
}
