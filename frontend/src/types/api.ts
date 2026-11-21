export type NavTab = "diagnose" | "batch" | "monitor";

export interface HealthApiResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
}

export interface ModelInfoApiResponse {
  model_name: string;
  model_version: string;
  backend_type: string;
  architecture: string;
  num_classes: number;
  val_macro_f1: number;
  val_accuracy: number;
  test_accuracy: number;
  model_size_mb: number;
  device: string;
  providers: string[];
  xai_method: string;
  confidence_threshold: number;
}

export interface SampleItem {
  filename: string;
  plant: string;
  condition: string;
  is_healthy: boolean;
  url: string;
}

export interface SamplesApiResponse {
  count: number;
  samples: SampleItem[];
}

export interface TopPrediction {
  class_name: string;
  plant: string;
  condition: string;
  confidence: number;
}

export interface Advisory {
  severity: "none" | "medium" | "high";
  summary: string;
  treatment: string[];
  prevention: string[];
}

export interface PredictApiResponse {
  predicted_class: string;
  plant: string;
  condition: string;
  is_healthy: boolean;
  confidence: number;
  top_3: TopPrediction[];
  is_uncertain: boolean;
  warning: string | null;
  cache_hit: boolean;
  inference_time_ms: number;
  explain_time_ms: number | null;
  xai_method: string | null;
  original_image_base64: string | null;
  heatmap_only_base64: string | null;
  heatmap_overlay_base64: string | null;
  advisory: Advisory;
  model_version: string;
}

export interface BatchResultItem {
  filename: string;
  predicted_class: string;
  plant: string;
  condition: string;
  is_healthy: boolean;
  confidence: number;
  top_3: TopPrediction[];
  is_uncertain: boolean;
  warning: string | null;
  advisory: Advisory;
}

export interface BatchApiResponse {
  count: number;
  total_inference_time_ms: number;
  avg_time_per_image_ms: number;
  model_version: string;
  results: BatchResultItem[];
}

export interface MetricsHistoryItem {
  timestamp: string;
  latency_ms: number;
  confidence: number;
  predicted_class: string;
  plant: string;
  condition: string;
  cache_hit: boolean;
  is_uncertain: boolean;
}

export interface MetricsApiResponse {
  uptime_seconds: number;
  model_version: string;
  backend_type: string;
  total_requests: number;
  total_images_predicted: number;
  cache_hits: number;
  cache_hit_rate: number;
  uncertain_count: number;
  uncertain_rate: number;
  latency_ms: {
    avg: number;
    p50: number;
    p95: number;
    min: number;
    max: number;
  };
  top_predicted_classes: Record<string, number>;
  history: MetricsHistoryItem[];
}
