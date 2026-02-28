// Corresponds to backend DatasetResponse
export interface Dataset {
    name: string;
    original_objectives: [number, number][];
    original_decisions: number[][]; // D-dimensional bounds/anchors
    bounds: Record<string, [number, number]>;
    is_trained: boolean;
}

// Corresponds to backend GenerationRequest
export interface GenerationRequest {
    dataset_name: string;
    target_objective: [number, number];
    n_samples: number;
    trust_radius: number;
    concentration_factor?: number; // Optional, defaults to 10.0
    error_threshold?: number;
}

// Corresponds to backend GenerationResponse
export interface GenerationResponse {
    pathway: "coherent" | "incoherent";
    target_objective: [number, number];
    candidate_decisions: number[][];
    candidate_objectives: [number, number][];
    objective_space_residual_sorted: number[];
    vertices_indices: number[];
    is_simplex_found: boolean;
    is_coherent: boolean;
    best_index: number;
    best_objective: [number, number];
    best_decision: number[];
}
export type Vector = number[];

// Corresponds to backend DatasetDetailResponse
export interface DatasetDetailResponse {
    name: string;
    X: Vector[];
    y: Vector[];
    is_pareto: boolean[];
    bounds: Record<string, [number, number]>;
    is_trained: boolean;
}

export interface DatasetGenerationRequest {
    function_id: number;
    population_size: number;
    n_var: number;
    generations: number;
    dataset_name: string;
}
