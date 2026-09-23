/**
 * TypeScript mirror of the FastAPI contract (backend/app/schemas/*.py and
 * backend/app/api/routes.py). These are the only API types in the frontend;
 * components import from here instead of redeclaring shapes.
 */

export type Verdict = "likely_cover" | "potential_steganography" | "inconclusive";
export type RiskLevel = "low" | "elevated" | "high" | "not_assessed";
export type Direction = "increases_stego_signal" | "decreases_stego_signal" | "informational";

export interface FeatureIndicator {
  name: string;
  label: string;
  value: number;
  direction: Direction;
  /** Deviation from the reference clean-image mean, in reference standard deviations. */
  deviation_from_reference_cover: number | null;
  /** Global Random Forest feature importance. */
  model_importance: number | null;
}

export interface ImageInfo {
  format: "PNG" | "JPEG";
  width: number;
  height: number;
  original_mode: string;
  size_bytes: number;
  alpha_discarded: boolean;
  analyzed_as: string;
}

export interface AnalysisResponse {
  request_id: string;
  verdict: Verdict;
  risk_level: RiskLevel;
  /** Uncalibrated model score (model output x 100). NOT a probability. */
  stego_score: number;
  score_kind: "uncalibrated_model_score";
  cover_reference_percentile: number | null;
  risk_context: string;
  model: string;
  model_version: string;
  feature_schema_version: string;
  feature_count: number;
  indicators: FeatureIndicator[];
  explanation_method: string;
  image: ImageInfo;
  warnings: string[];
  limitations: string[];
}

export interface RiskLevelDefinition {
  cover_percentile_at_or_above: number;
  flag_rate_held_out_test: { clean: number; lsb_stego: number };
}

export interface ModelInfoResponse {
  name: string;
  version: string;
  feature_schema: string;
  scope: string;
  feature_count: number;
  artifact_sha256: string;
  training_dataset_id: string | null;
  random_seed: number | null;
  /** Phase 1 held-out in-domain test metrics stored in the artifact. */
  reference_performance: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    roc_auc?: number;
    n_samples?: number;
  } | null;
  score_interpretation: string;
  risk_levels: {
    definition: string;
    reference_stego: string;
    levels: Partial<Record<"elevated" | "high", RiskLevelDefinition>>;
  } | null;
  limitations: string[];
}

export interface HealthResponse {
  status: string;
  service: string;
}

export interface ReadyResponse {
  status: "ready" | "not_ready" | string;
  model_loaded: boolean;
  reference_available: boolean;
  reason: string | null;
}

/** Error body returned by every failing backend route. */
export interface ApiErrorBody {
  detail: string;
  code: string;
  request_id: string;
}

export interface EncodeResult {
  /** The generated PNG. */
  blob: Blob;
  embeddedBytes: number | null;
  capacityBytes: number | null;
  requestId: string | null;
}
