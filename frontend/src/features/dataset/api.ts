import { apiClient } from "@/lib/api-client";
import { DatasetInfo, DatasetDetails, GeneratorsDiscoveryResponse, DatasetGenerationRequest } from "./types";

/**
 * Fetch available dataset generators and their parameters.
 */
export const getGenerators = async (): Promise<GeneratorsDiscoveryResponse> => {
  return apiClient.get("/api/v1/datasets/generators");
};

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
 * Fetch all available datasets.
 */
export const getDatasets = async (): Promise<DatasetInfo[]> => {
  return apiClient.get("/api/v1/datasets");
};

/**
 * Trigger generation of a new synthetic dataset.
 */
export const generateDataset = async (params: DatasetGenerationRequest): Promise<any> => {
  return apiClient.post("/api/v1/datasets", params);
};

/**
 * Delete one or multiple datasets.
 */
export const deleteDatasets = async (datasetNames: string[]): Promise<any> => {
  return apiClient.delete("/api/v1/datasets", { data: { dataset_names: datasetNames } });
};
