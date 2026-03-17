export interface TransformationStep {
  type: string;
  params: Record<string, any>;
  columns?: number[];
}

export interface PreviewRequest {
  dataset_name: string;
  split: string;
  sampling_limit: number;
  chain?: TransformationStep[];
  x_chain?: TransformationStep[];
  y_chain?: TransformationStep[];
}

export interface PreviewResponse {
  original: {
    X: number[][];
    y: number[][];
  };
  transformed: {
    X: number[][];
    y: number[][];
  };
  metrics: Record<string, any>;
}

export interface TransformerParameter {
  name: string;
  type: string;
  required: boolean;
  default: any;
  description?: string;
  options?: any[];
}

export interface TransformerMetadata {
  type: string;
  name: string;
  parameters: TransformerParameter[];
}

export interface TransformerRegistryResponse {
  transformers: TransformerMetadata[];
}
