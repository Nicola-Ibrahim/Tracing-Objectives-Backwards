import { apiClient } from "@/lib/api-client";
import { DatasetInfo, DatasetDetails, GeneratorsDiscoveryResponse, DatasetGenerationRequest } from "./types";

/**
 * Fetch available dataset generators and their parameters.
 */
export const getGenerators = async (): Promise<GeneratorsDiscoveryResponse> => {
  return apiClient.get("/datasets/generators");
};

/**
 * Fetch detailed dataset information including coordinates for plotting.
 */
export const getDatasetDetails = async (
  datasetName: string,
  split: "train" | "test" | "all" = "train"
): Promise<DatasetDetails> => {
  return apiClient.get(`/datasets/${datasetName}`, { params: { split } });
};

/**
 * Fetch all available datasets.
 */
export const getDatasets = async (): Promise<DatasetInfo[]> => {
  return apiClient.get("/datasets");
};

/**
 * Trigger generation of a new synthetic dataset.
 */
export const generateDataset = async (params: DatasetGenerationRequest): Promise<unknown> => {
  return apiClient.post("/datasets", params);
};

/**
 * Delete one or multiple datasets.
 */
export const deleteDatasets = async (datasetNames: string[]): Promise<unknown> => {
  return apiClient.delete("/datasets", { data: { dataset_names: datasetNames } });
};
