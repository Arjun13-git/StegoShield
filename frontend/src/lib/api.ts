import { abortedError, ApiClientError, errorFromResponse, networkError, timeoutError } from "./errors";
import type { AnalysisResponse, EncodeResult, HealthResponse, ModelInfoResponse, ReadyResponse } from "./types";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

/** Backend base URL (browser-visible; never put secrets in NEXT_PUBLIC_* variables). */
export function getApiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL;
  return raw.replace(/\/+$/, "");
}

const TIMEOUTS_MS = { status: 8_000, analyze: 60_000, encode: 60_000 } as const;

interface RequestOptions {
  method?: "GET" | "POST";
  body?: FormData;
  timeoutMs: number;
  signal?: AbortSignal;
}

/** fetch with a timeout and caller cancellation; network-level failures become ApiClientError. */
async function send(path: string, opts: RequestOptions): Promise<Response> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, opts.timeoutMs);
  const onCallerAbort = () => controller.abort();
  opts.signal?.addEventListener("abort", onCallerAbort, { once: true });
  try {
    return await fetch(`${getApiBaseUrl()}${path}`, {
      method: opts.method ?? "GET",
      body: opts.body,
      headers: { Accept: "application/json, image/png" },
      signal: controller.signal,
      cache: "no-store",
      credentials: "omit",
    });
  } catch {
    if (timedOut) throw timeoutError();
    if (opts.signal?.aborted) throw abortedError();
    throw networkError();
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", onCallerAbort);
  }
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (response.ok) {
    const body = await readJson(response);
    if (body === null) {
      throw new ApiClientError({ kind: "http", message: "The server returned an unreadable response.", status: response.status });
    }
    return body as T;
  }
  throw errorFromResponse(response.status, await readJson(response), response.headers.get("X-Request-ID"));
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return jsonOrThrow<HealthResponse>(await send("/api/v1/health", { timeoutMs: TIMEOUTS_MS.status, signal }));
}

/**
 * Readiness. The backend answers 503 with a ReadyResponse body while the model
 * is unavailable; that is a normal "not ready" result, not an exception.
 */
export async function getReady(signal?: AbortSignal): Promise<ReadyResponse> {
  const response = await send("/ready", { timeoutMs: TIMEOUTS_MS.status, signal });
  if (response.ok || response.status === 503) {
    const body = (await readJson(response)) as Partial<ReadyResponse> | null;
    if (body && typeof body.model_loaded === "boolean") {
      return {
        status: body.status ?? (response.ok ? "ready" : "not_ready"),
        model_loaded: body.model_loaded,
        reference_available: Boolean(body.reference_available),
        reason: body.reason ?? null,
      };
    }
  }
  throw errorFromResponse(response.status, null, response.headers.get("X-Request-ID"));
}

export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfoResponse> {
  return jsonOrThrow<ModelInfoResponse>(await send("/api/v1/model-info", { timeoutMs: TIMEOUTS_MS.status, signal }));
}

export async function analyzeImage(file: File, signal?: AbortSignal): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("file", file);
  return jsonOrThrow<AnalysisResponse>(await send("/api/v1/analyze", { method: "POST", body: form, timeoutMs: TIMEOUTS_MS.analyze, signal }));
}

function headerInt(response: Response, name: string): number | null {
  const raw = response.headers.get(name);
  if (raw === null) return null;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

export async function encodeImage(file: File, message: string, signal?: AbortSignal): Promise<EncodeResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("message", message);
  const response = await send("/api/v1/encode", { method: "POST", body: form, timeoutMs: TIMEOUTS_MS.encode, signal });
  if (!response.ok) {
    throw errorFromResponse(response.status, await readJson(response), response.headers.get("X-Request-ID"));
  }
  const blob = await response.blob();
  if (blob.size === 0) {
    throw new ApiClientError({ kind: "http", message: "The server returned an empty image.", status: response.status });
  }
  return {
    blob,
    embeddedBytes: headerInt(response, "X-Embedded-Bytes"),
    capacityBytes: headerInt(response, "X-Capacity-Bytes"),
    requestId: response.headers.get("X-Request-ID"),
  };
}
