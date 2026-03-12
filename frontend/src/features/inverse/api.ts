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
  return apiClient.get("/api/v1/datasets"); 
};

/**
 * Trigger training for a new inverse mapping engine.
 */
export const trainEngine = async (params: TrainEngineRequest): Promise<TrainEngineResponse> => {
  return apiClient.post("/api/v1/inverse/train", params);
};

/**
 * List existing trained engines for a dataset using the RESTful dataset-relative path.
 */
export const listEnginesForDataset = async (datasetName: string): Promise<EngineListItem[]> => {
  return apiClient.get(`/api/v1/datasets/${datasetName}/engines`);
};

/**
 * Generate candidate solutions for a target objective.
 */
export const generateCandidates = async (params: CandidateGenerationRequest): Promise<CandidateGenerationResponse> => {
  return apiClient.post("/api/v1/inverse/generate", params);
};

/**
 * List all trained engines across all datasets (Inference Hub).
 */
export const listAllEngines = async (): Promise<EngineListItem[]> => {
  return apiClient.get("/api/v1/inverse/engines");
};

/**
 * Delete one or multiple engines, grouped by dataset for RESTful compliance.
 */
export const deleteEngines = async (engines: { dataset_name: string; solver_type: string; version: number }[]): Promise<any> => {
  // Group engines by dataset_name to use the new RESTful delete endpoint
  const grouped = engines.reduce((acc, engine) => {
    if (!acc[engine.dataset_name]) acc[engine.dataset_name] = [];
    acc[engine.dataset_name].push(engine);
    return acc;
  }, {} as Record<string, typeof engines>);

  const results = await Promise.all(
    Object.entries(grouped).map(([datasetName, datasetEngines]) =>
      apiClient.delete(`/api/v1/datasets/${datasetName}/engines`, { data: { engines: datasetEngines } })
    )
  );
  
  return results.flat();
};
