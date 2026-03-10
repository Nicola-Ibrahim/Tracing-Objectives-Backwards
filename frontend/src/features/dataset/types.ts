export interface DatasetInfo {
  name: string;
  n_samples: number;
  n_features: number;
  n_objectives: number;
  trained_engines_count: number;
}

export interface DatasetDetails extends DatasetInfo {
  samples: number;
  objectives_count: number;
  decisions_count: number;
  X: number[][];
  y: number[][];
  is_pareto: boolean[];
  bounds: Record<string, [number, number]>;
}

export interface ParameterDefinition {
  name: string;
  type: string;
  required: boolean;
  default: any | null;
  options?: any[] | null;
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
  params: Record<string, any>;
  split_ratio?: number;
  random_state?: number;
}
