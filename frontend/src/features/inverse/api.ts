import { apiClient } from "@/lib/api-client";
import { DatasetInfo } from "../dataset/types";
import {
  TrainEngineRequest,
  TrainEngineResponse,
  CandidateGenerationRequest,
  CandidateGenerationResponse,
  EngineListItem,
  SolversDiscoveryResponse,
} from "./types";

/**
 * Fetch all available solvers and their parameters.
 */
export const getSolvers = async (): Promise<SolversDiscoveryResponse> => {
  return apiClient.get("/api/v1/inverse/solvers");
};

/**
 * Fetch all available datasets.
 */
export const getDatasets = async (): Promise<DatasetInfo[]> => {
  return apiClient.get("/api/v1/datasets"); // Removed trailing slash
};

/**
 * Trigger training for a new inverse mapping engine.
 */
export const trainEngine = async (params: TrainEngineRequest): Promise<TrainEngineResponse> => {
  return apiClient.post("/api/v1/inverse/train", params);
};

/**
 * List existing trained engines for a dataset.
 */
export const listEnginesForDataset = async (datasetName: string): Promise<EngineListItem[]> => {
  return apiClient.get(`/api/v1/inverse/engines/${datasetName}`);
};

/**
 * Generate candidate solutions for a target objective.
 */
export const generateCandidates = async (params: CandidateGenerationRequest): Promise<CandidateGenerationResponse> => {
  return apiClient.post("/api/v1/inverse/generate", params);
};

/**
 * List all trained engines across all datasets.
 */
export const listAllEngines = async (datasetName?: string): Promise<EngineListItem[]> => {
  return apiClient.get("/api/v1/inverse/engines", { params: { dataset_name: datasetName } });
};

/**
 * Delete one or multiple engines.
 */
export const deleteEngines = async (engines: { dataset_name: string; solver_type: string; version: number }[]): Promise<any> => {
  return apiClient.post("/api/v1/inverse/engines/delete", { engines });
};
