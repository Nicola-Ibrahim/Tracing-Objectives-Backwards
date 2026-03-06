import { z } from "zod";

/**
 * Common schema for solver training parameters.
 * Detailed parameters are managed via discovery/dynamic form.
 */
export const trainEngineSchema = z.object({
  dataset_name: z.string().min(1, "Dataset is required"),
  solver_type: z.string().min(1, "Solver type is required"),
});

export type TrainEngineFormValues = z.infer<typeof trainEngineSchema>;
