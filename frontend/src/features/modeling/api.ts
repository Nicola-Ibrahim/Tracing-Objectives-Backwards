import { apiClient } from "@/lib/api-client";

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

export interface TransformerMetadata {
  type: string;
  name: string;
  params: Record<string, {
    type: string;
    default: any;
    description?: string;
  }>;
}

export interface TransformerRegistryResponse {
  transformers: TransformerMetadata[];
}

export const getTransformers = async (): Promise<TransformerRegistryResponse> => {
  return apiClient.get("/api/v1/modeling/transformers");
};

export const getPreview = async (data: PreviewRequest): Promise<PreviewResponse> => {
  return apiClient.post("/api/v1/modeling/preview", data);
};
