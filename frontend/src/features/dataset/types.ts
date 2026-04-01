export interface DatasetMetadata {
  n_samples: number;
  n_train: number;
  n_test: number;
  split_ratio: number;
  random_state: number;
  created_at: string;
}

export interface DatasetInfo {
  name: string;
  n_features: number;
  n_objectives: number;
  metadata: DatasetMetadata;
  trained_engines_count: number;
}

export interface DatasetDetails {
  name: string;
  metadata: DatasetMetadata;
  objectives_dim: number;
  decisions_dim: number;
  X: number[][];
  y: number[][];
  is_pareto: boolean[];
  bounds: Record<string, [number, number]>;
  trained_engines?: {
    solver_type: string;
    version: number;
    created_at: string;
  }[];
}

export interface ParameterDefinition {
  name: string;
  type: string;
  required: boolean;
  default: unknown;
  options?: unknown[] | null;
  description?: string | null;
}

export interface GeneratorSchema {
  type: string;
  name: string;
  parameters: ParameterDefinition[];
}

export interface GeneratorsDiscoveryResponse {
  generators: GeneratorSchema[];
}

export interface DatasetGenerationRequest {
  dataset_name: string;
  generator_type: string;
  params: Record<string, unknown>;
  split_ratio?: number;
  random_state?: number;
}
