import summaryJson from "@/data/research-summary.json";

/**
 * Aggregate Phase 1A/1B results, generated from the experiment reports by
 * scripts/export_research_summary.py. Nothing here is typed in by hand.
 */
export const research = summaryJson;

export type Research = typeof summaryJson;

export const CANDIDATE_LABEL: Record<string, string> = {
  LogisticRegression: "Logistic Regression",
  SVM: "SVM",
  RandomForest: "Random Forest",
};

export const EXTERNAL_LABEL: Record<string, string> = {
  "B_lsb_0.10": "Our controlled LSB, 0.10 bits per pixel per channel",
  "B_lsb_0.40": "Our controlled LSB, 0.40 bits per pixel per channel",
  C_jmipod: "JMiPOD (native DCT-domain)",
  C_juniward: "J-UNIWARD (native DCT-domain)",
  C_uerd: "UERD (native DCT-domain)",
};

export const ROBUSTNESS_LABEL: Record<string, string> = {
  jpeg_qf90: "JPEG re-compression, quality 90",
  jpeg_qf75: "JPEG re-compression, quality 75",
  gaussian_noise_sigma2: "Gaussian noise, σ = 2",
  crop_480_offset13: "Crop to 480 × 480 px",
  "resize_0.9x": "Resize to 0.9×",
};

export function fixed(n: number, digits = 3): string {
  return n.toFixed(digits);
}

export function signed(n: number, digits = 3): string {
  return `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;
}

export function interval(ci: readonly number[], digits = 3): string {
  return `[${ci[0].toFixed(digits)}, ${ci[1].toFixed(digits)}]`;
}
