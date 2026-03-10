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
  training_history: Record<string, any>;
  transform_summary: string[];
}

export interface CandidateGenerationRequest {
  dataset_name: string;
  target_objective: number[];
  n_samples: number;
  solver_type?: string;
  version?: number;
  params?: Record<string, any>;
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
  metadata: Record<string, any>;
}

export interface EngineListItem {
  dataset_name?: string;
  solver_type: string;
  version: number;
  created_at: string;
}
