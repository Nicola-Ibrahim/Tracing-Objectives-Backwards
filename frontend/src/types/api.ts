// Corresponds to backend DatasetResponse
export interface Dataset {
    name: string;
    original_objectives: [number, number][];
    original_decisions: number[][]; // D-dimensional bounds/anchors
    bounds: Record<string, [number, number]>;
}

// Corresponds to backend GenerationRequest
export interface GenerationRequest {
    dataset_name: string;
    target_objective: [number, number];
    n_samples: number;
    trust_radius: number;
    concentration_factor?: number; // Optional, defaults to 10.0
}

// Corresponds to backend GenerationResponse
export interface GenerationResponse {
    pathway: "coherent" | "incoherent";
    target_objective: [number, number];
    candidate_decisions: number[][];
    candidate_objectives: [number, number][];
    residual_errors: number[];
    anchor_indices: number[];
    is_inside_mesh: boolean;
}
export type Vector = number[];

// Corresponds to backend DatasetDetailResponse
export interface DatasetDetailResponse {
    name: string;
    X: Vector[];
    y: Vector[];
    is_pareto: boolean[];
    bounds: Record<string, [number, number]>;
}

export interface DatasetGenerationRequest {
    function_id: number;
    population_size: number;
    n_var: number;
    generations: number;
    dataset_name: string;
}
