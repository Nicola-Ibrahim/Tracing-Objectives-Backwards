import { apiClient } from "@/lib/api-client";
import { 
  TransformerRegistryResponse, 
  PreviewRequest, 
  PreviewResponse 
} from "./types";

export const getTransformers = async (): Promise<TransformerRegistryResponse> => {
  return apiClient.get("/modeling/transformers");
};

export const getPreview = async (data: PreviewRequest): Promise<PreviewResponse> => {
  return apiClient.post("/modeling/preview", data);
};

export * from "./types";
