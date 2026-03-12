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

export interface DomainAssessmentData {
  ecdf: MetricPlotData;
  calibration_curves: MetricPlotData;
  metrics: {
    [engineLabel: string]: {
      [metricName: string]: number;
    };
  };
}

export interface DiagnoseResponse {
  dataset_name: string;
  engines: string[];
  capabilities: { [engineName: string]: string };
  objective_space: DomainAssessmentData;
  decision_space: DomainAssessmentData;
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
