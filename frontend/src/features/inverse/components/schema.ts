import { z } from "zod";

/**
 * Common schema for solver training parameters.
 */
export const trainEngineSchema = z.object({
  dataset_name: z.string().min(1, "Dataset is required"),
  solver_type: z.enum(["GBPI", "MDN", "probabilistic"]),
  
  // Solver-specific Params (Nested)
  gbpi_params: z.object({
    n_neighbors: z.coerce.number().min(1).default(10),
    trust_radius: z.coerce.number().min(0).default(0.1),
    concentration_factor: z.coerce.number().min(0).default(1.0),
  }).optional(),
  
  mdn_params: z.object({
    n_hidden: z.coerce.number().min(1).default(64),
    n_mixtures: z.coerce.number().min(1).default(5),
    epochs: z.coerce.number().min(1).default(100),
    lr: z.coerce.number().min(0).default(0.001),
  }).optional(),
});

export type TrainEngineFormValues = z.infer<typeof trainEngineSchema>;
