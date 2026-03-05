import { z } from "zod";

export const generateCandidatesSchema = z.object({
  dataset_name: z.string().min(1, "Dataset is required"),
  engine_id: z.string().min(1, "Engine is required"),
  objective1: z.coerce.number(),
  objective2: z.coerce.number(),
  n_samples: z.coerce.number().min(1).max(1000),
});

export type GenerateCandidatesFormValues = z.infer<typeof generateCandidatesSchema>;
