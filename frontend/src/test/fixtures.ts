import type { AnalysisResponse, ModelInfoResponse } from "@/lib/types";

export const analysisFixture: AnalysisResponse = {
  request_id: "11111111-2222-3333-4444-555555555555",
  verdict: "potential_steganography",
  risk_level: "elevated",
  stego_score: 52.57,
  score_kind: "uncalibrated_model_score",
  cover_reference_percentile: 91.2,
  risk_context: "At or above the elevated level (top 10% of reference clean images).",
  model: "RandomForest",
  model_version: "v1",
  feature_schema_version: "v1",
  feature_count: 22,
  indicators: [
    {
      name: "lsb_transition_b",
      label: "LSB transition rate (blue channel)",
      value: 0.5011,
      direction: "increases_stego_signal",
      deviation_from_reference_cover: 0.719,
      model_importance: 0.1085,
    },
    {
      name: "entropy_r",
      label: "Intensity histogram entropy (red channel)",
      value: 7.2,
      direction: "informational",
      deviation_from_reference_cover: -0.1,
      model_importance: 0.02,
    },
  ],
  explanation_method: "Features ranked by global model importance times deviation from the reference clean-image distribution.",
  image: { format: "PNG", width: 256, height: 256, original_mode: "L", size_bytes: 65917, alpha_discarded: false, analyzed_as: "RGB (grey replicated)" },
  warnings: ["The image is not 512x512, the size used in the research evaluations."],
  limitations: [
    "The score is an uncalibrated model output, not a probability that the image contains hidden data.",
    "It was not found to detect the JPEG-domain steganography algorithms evaluated in Phase 1B (JMiPOD, J-UNIWARD, UERD).",
  ],
};

export const modelInfoFixture: ModelInfoResponse = {
  name: "RandomForest",
  version: "v1",
  feature_schema: "v1",
  scope: "LSB-focused image steganalysis",
  feature_count: 22,
  artifact_sha256: "34707f83dc385e0846be3df7c56da930aad951a23fd3daa032a050449d02523d",
  training_dataset_id: "features_payload_0.10",
  random_seed: 42,
  reference_performance: { accuracy: 0.54, precision: 0.53, recall: 0.8, f1: 0.64, roc_auc: 0.55, n_samples: 3000 },
  score_interpretation: "stego_score is the model output x 100. It is uncalibrated and is not a probability.",
  risk_levels: {
    definition: "Percentiles of reference clean-image scores.",
    reference_stego: "Stego reference = controlled LSB at 0.10 bits per pixel.",
    levels: {
      elevated: { cover_percentile_at_or_above: 0.9, flag_rate_held_out_test: { clean: 0.11, lsb_stego: 0.14 } },
      high: { cover_percentile_at_or_above: 0.99, flag_rate_held_out_test: { clean: 0.01, lsb_stego: 0.02 } },
    },
  },
  limitations: ["Limitation."],
};

export function pngFile(name = "cover.png", size = 2048): File {
  return new File([new Uint8Array(size)], name, { type: "image/png" });
}
