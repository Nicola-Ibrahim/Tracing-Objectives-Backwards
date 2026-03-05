import { apiClient } from "@/lib/api-client";
import { DatasetInfo, DatasetDetails } from "./types";

/**
 * Fetch detailed dataset information including coordinates for plotting.
 */
export const getDatasetDetails = async (
  datasetName: string,
  split: "train" | "test" | "all" = "train"
): Promise<DatasetDetails> => {
  return apiClient.get(`/api/v1/datasets/${datasetName}`, { params: { split } });
};

/**
 * Trigger generation of a new synthetic dataset.
 */
export const generateDataset = async (params: any): Promise<any> => {
  return apiClient.post("/api/v1/datasets/generate", params);
};
