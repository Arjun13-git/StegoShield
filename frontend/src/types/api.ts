export interface FeatureIndicator {
  name: string;
  value: number;
  direction: string;
}

export interface AnalysisResponse {
  request_id: string;
  verdict: 'likely_cover' | 'potential_steganography' | 'inconclusive';
  stego_score: number;
  model: string;
  model_version: string;
  feature_count: number;
  indicators: FeatureIndicator[];
  limitations: string[];
}
