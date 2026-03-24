import { apiClient } from "@/lib/api-client";
import { DiagnoseRequest, DiagnoseResponse, DiagnoseAsyncResponse, PerformanceRequest, PerformanceResponse } from "./types";

/**
 * Trigger diagnostic comparison across multiple engines.
 */
export const diagnoseEngines = async (params: DiagnoseRequest): Promise<DiagnoseAsyncResponse> => {
  return apiClient.post("/evaluation/diagnose", params);
};


export const checkPerformance = async (params: PerformanceRequest): Promise<PerformanceResponse> => {
  return apiClient.post("/evaluation/performance", params);
};


/**
 * Helper to map individual diagnostic reports into the monolithic DiagnoseResponse
 * structure expected by the UI.
 */
export const mapReportsToDiagnoseResponse = (reports: any[]): DiagnoseResponse => {
    if (!reports.length) throw new Error("No reports to map");
    
    const first = reports[0];
    const engines = reports.map(r => r.engine.type);
    const capabilities = Object.fromEntries(
        reports.map(r => [r.engine.type, r.engine.capability])
    );

    const objective_space: any = { ecdf: {}, calibration_curves: {}, metrics: {} };
    const decision_space: any = { ecdf: {}, calibration_curves: {}, metrics: {} };

    reports.forEach(r => {
        const engineLabel = r.engine.type;
        
        // Objective Space
        objective_space.ecdf[engineLabel] = r.objective_space.ecdf;
        objective_space.calibration_curves[engineLabel] = r.objective_space.calibration_curves;
        objective_space.metrics[engineLabel] = r.objective_space.metrics;

        // Decision Space
        decision_space.ecdf[engineLabel] = r.decision_space.ecdf;
        decision_space.calibration_curves[engineLabel] = r.decision_space.calibration_curves;
        decision_space.metrics[engineLabel] = r.decision_space.metrics;
    });

    return {
        dataset_name: first.dataset_name,
        engines,
        capabilities,
        objective_space,
        decision_space,
        warnings: []
    };
};
