import type {
  BatchApiResponse,
  HealthApiResponse,
  MetricsApiResponse,
  ModelInfoApiResponse,
  PredictApiResponse,
  SamplesApiResponse,
} from "../types/api";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function resolveApiUrl(path: string): string {
  if (!path) return API_BASE;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}

async function parseErrorMessage(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: unknown; message?: unknown };
    if (typeof data.detail === "string") return data.detail;
    if (typeof data.message === "string") return data.message;
  } catch {
    // ignore JSON parse failure
  }
  return `Lỗi máy chủ (${res.status} ${res.statusText})`;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(resolveApiUrl(path), init);
  if (!res.ok) {
    throw new ApiError(res.status, await parseErrorMessage(res));
  }
  return (await res.json()) as T;
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthApiResponse> {
  return fetchJson<HealthApiResponse>("/api/health", { signal });
}

export function fetchModelInfo(signal?: AbortSignal): Promise<ModelInfoApiResponse> {
  return fetchJson<ModelInfoApiResponse>("/api/model-info", { signal });
}

export function fetchSamples(signal?: AbortSignal): Promise<SamplesApiResponse> {
  return fetchJson<SamplesApiResponse>("/api/samples", { signal });
}

export interface PredictOptions {
  file?: File;
  sampleName?: string;
  includeXai?: boolean;
  signal?: AbortSignal;
}

export function predictSingle({
  file,
  sampleName,
  includeXai = true,
  signal,
}: PredictOptions): Promise<PredictApiResponse> {
  const formData = new FormData();
  if (file) {
    formData.append("file", file);
  } else if (sampleName) {
    formData.append("sample_name", sampleName);
  } else {
    return Promise.reject(new ApiError(400, "Vui lòng chọn ảnh hoặc ảnh mẫu."));
  }
  formData.append("include_xai", includeXai ? "true" : "false");

  return fetchJson<PredictApiResponse>("/api/predict", {
    method: "POST",
    body: formData,
    signal,
  });
}

export interface BatchProgressState {
  phase: "uploading" | "processing";
  percent: number;
  loadedBytes: number;
  totalBytes: number;
}

export function predictBatchWithProgress(
  files: File[],
  onProgress: (state: BatchProgressState) => void,
  signal?: AbortSignal,
): Promise<BatchApiResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", resolveApiUrl("/api/predict-batch"));

    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    const abortHandler = () => {
      xhr.abort();
      reject(new DOMException("Aborted", "AbortError"));
    };

    if (signal) {
      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      signal.addEventListener("abort", abortHandler, { once: true });
    }

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        const pct = Math.min(100, Math.round((event.loaded / event.total) * 100));
        onProgress({
          phase: pct >= 100 ? "processing" : "uploading",
          percent: pct,
          loadedBytes: event.loaded,
          totalBytes: event.total,
        });
      }
    };

    xhr.upload.onloadend = () => {
      onProgress({
        phase: "processing",
        percent: 100,
        loadedBytes: 1,
        totalBytes: 1,
      });
    };

    xhr.onload = () => {
      signal?.removeEventListener("abort", abortHandler);
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const parsed = JSON.parse(xhr.responseText) as BatchApiResponse;
          resolve(parsed);
        } catch {
          reject(new ApiError(xhr.status, "Phản hồi JSON từ máy chủ không hợp lệ."));
        }
      } else {
        let detail = `Lỗi máy chủ (${xhr.status})`;
        try {
          const errJson = JSON.parse(xhr.responseText) as { detail?: string };
          if (typeof errJson.detail === "string") detail = errJson.detail;
        } catch {
          // ignore
        }
        reject(new ApiError(xhr.status, detail));
      }
    };

    xhr.onerror = () => {
      signal?.removeEventListener("abort", abortHandler);
      reject(new ApiError(0, "Không thể kết nối tới backend API. Vui lòng kiểm tra server."));
    };

    xhr.send(formData);
  });
}

export function fetchMetrics(signal?: AbortSignal): Promise<MetricsApiResponse> {
  return fetchJson<MetricsApiResponse>("/api/metrics", { signal });
}
