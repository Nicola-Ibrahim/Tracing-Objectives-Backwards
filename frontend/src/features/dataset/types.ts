export interface DatasetInfo {
  name: string;
  n_samples: number;
  n_features: number;
  n_objectives: number;
  trained_engines_count: number;
}

export interface DatasetDetails extends DatasetInfo {
  X: number[][];
  y: number[][];
  is_pareto: boolean[];
  bounds: Record<string, [number, number]>;
}
