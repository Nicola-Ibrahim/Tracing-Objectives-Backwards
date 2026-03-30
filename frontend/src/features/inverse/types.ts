import { ParameterDefinition } from "../dataset/types";

export interface SolverSchema {
  type: string;
  name: string;
  parameters: ParameterDefinition[];
}

export interface SolversDiscoveryResponse {
  solvers: SolverSchema[];
}

export interface SolverConfig {
  type: string;
  params: Record<string, unknown>;
}

export interface TransformConfig {
  target: string;
  type: string;
}

export interface TrainEngineRequest {
  dataset_name: string;
  solver: SolverConfig;
  transforms: TransformConfig[];
}

export interface TrainingHistoryData {
    epochs: number[];
    train_loss: number[];
    val_loss: number[];
    [key: string]: unknown;
}

export interface TrainEngineResponse {
  dataset_name: string;
  solver_type: string;
  engine_version: number;
  status: string;
  duration_seconds: number;
  n_train_samples: number;
  n_test_samples: number;
  split_ratio: number;
  training_history: TrainingHistoryData;
  transform_summary: string[];
}

export interface CandidateGenerationRequest {
  dataset_name: string;
  target_objective: number[];
  n_samples: number;
  solver_type?: string;
  version?: number;
  params?: Record<string, unknown>;
}

export interface CandidateGenerationResponse {
  solver_type: string;
  target_objective: number[];
  candidate_decisions: number[][];
  candidate_objectives: number[][];
  best_index: number;
  best_candidate_decision: number[];
  best_candidate_objective: number[];
  best_candidate_residual: number;
  metadata: {
    vertices_indices?: number[];
    pathway?: string;
    is_simplex_found?: boolean;
    log_likelihood?: number[];
    [key: string]: unknown;
  };
}

export interface EngineListItem {
  dataset_name: string;
  solver_type: string;
  version: number;
  created_at: string;
}

export interface EngineDetailResponse {
  dataset_name: string;
  solver_type: string;
  version: number;
  created_at: string;
  n_train_samples: number;
  n_test_samples: number;
  split_ratio: number;
  training_history: TrainingHistoryData;
  transform_summary: string[];
  hyperparameters: Record<string, unknown>;
}
