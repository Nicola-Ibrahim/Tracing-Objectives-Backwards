import { apiClient } from "@/lib/api-client";
import { DiagnoseRequest, DiagnoseResponse, PerformanceRequest, PerformanceResponse } from "./types";

/**
 * Trigger diagnostic comparison across multiple engines.
 */
export const diagnoseEngines = async (params: DiagnoseRequest): Promise<DiagnoseResponse> => {
  return apiClient.post("/api/v1/evaluation/diagnose", params);
};

/**
 * Check performance for a single engine.
 */
export const checkPerformance = async (params: PerformanceRequest): Promise<PerformanceResponse> => {
  return apiClient.post("/api/v1/evaluation/performance", params);
};
