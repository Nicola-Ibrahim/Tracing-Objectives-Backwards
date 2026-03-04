export interface SolverConfig {
  type: string;
  params: Record<string, any>;
}

export interface TransformConfig {
  target: string;
  type: string;
}

export interface TrainEngineRequest {
  dataset_name: string;
  solver: SolverConfig;
  transforms: TransformConfig[];
  split_ratio?: number;
}

export interface TrainEngineResponse {
  version: number;
  n_train_samples: number;
  n_test_samples: number;
  duration_seconds: number;
  transform_summary: string[];
  training_history: Record<string, any>;
}

export interface CandidateGenerationRequest {
  dataset_name: string;
  target_objective: number[];
  num_candidates: number;
  solver_type?: string;
  version?: number;
  trust_radius?: number;
  concentration_factor?: number;
}

export interface CandidateGenerationResponse {
  solver_type: string;
  target_objective: number[];
  candidate_decisions: number[][];
  candidate_objectives: number[][];
  best_index: number;
  best_decision: number[];
  best_objective: number[];
  all_residuals: number[];
  metadata: Record<string, any>;
}

export interface EngineListItem {
  solver_type: string;
  version: number;
  created_at: string;
}
