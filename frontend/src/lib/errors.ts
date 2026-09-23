import type { ApiErrorBody } from "./types";

export type ApiFailureKind = "http" | "network" | "timeout" | "aborted";

/**
 * The single error type thrown by the API client. `message` is always safe to
 * show to a user: it is either the backend's fixed message, a message we map
 * from its error code, or a generic sentence. Raw response bodies, stack traces
 * and exception names are never surfaced.
 */
export class ApiClientError extends Error {
  readonly kind: ApiFailureKind;
  readonly status: number | null;
  readonly code: string | null;
  readonly requestId: string | null;

  constructor(init: { kind: ApiFailureKind; message: string; status?: number | null; code?: string | null; requestId?: string | null }) {
    super(init.message);
    this.name = "ApiClientError";
    this.kind = init.kind;
    this.status = init.status ?? null;
    this.code = init.code ?? null;
    this.requestId = init.requestId ?? null;
  }
}

/** Friendly copy for the backend's stable error codes (backend/app/core/errors.py). */
const CODE_MESSAGES: Record<string, string> = {
  empty_file: "The selected file is empty.",
  invalid_image: "The file could not be read as a PNG or JPEG image. It may be corrupt or a different file type.",
  unsupported_format: "This image format is not supported. Use a PNG or JPEG image.",
  unsupported_color_mode: "This image's colour mode is not supported.",
  invalid_dimensions: "The image dimensions are outside the supported range.",
  file_too_large: "The file is larger than the server accepts.",
  missing_message: "Enter a message to embed.",
  message_too_large: "The message is longer than the server allows.",
  payload_exceeds_capacity: "The message does not fit in this image. Use a larger image or a shorter message.",
  model_unavailable: "The detection model is not available on the server right now.",
  analysis_failed: "The analysis failed on the server. Please try again.",
  encoding_failed: "Encoding failed on the server. Please try again.",
  invalid_request: "The request was malformed. Check the file and fields and try again.",
  internal_error: "The server hit an unexpected error. Please try again.",
};

const STATUS_FALLBACK: Record<number, string> = {
  400: "The server rejected the request.",
  413: "The upload is larger than the server accepts.",
  422: "The request was malformed.",
  429: "Too many requests. Please wait and try again.",
  500: "The server hit an unexpected error.",
  502: "The server is temporarily unreachable.",
  503: "The service is unavailable right now.",
  504: "The server took too long to respond.",
};

const MAX_DETAIL = 200;

export function isApiErrorBody(value: unknown): value is ApiErrorBody {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.detail === "string" && typeof v.code === "string";
}

/** Build a user-safe error from a failed HTTP response body (already parsed, may be anything). */
export function errorFromResponse(status: number, body: unknown, headerRequestId: string | null): ApiClientError {
  if (isApiErrorBody(body)) {
    const requestId = typeof body.request_id === "string" ? body.request_id : headerRequestId;
    const mapped = CODE_MESSAGES[body.code];
    // Backend messages are fixed and non-leaky, but bound the length defensively.
    const message = mapped ?? (body.detail.length <= MAX_DETAIL ? body.detail : STATUS_FALLBACK[status] ?? "The request failed.");
    return new ApiClientError({ kind: "http", message, status, code: body.code, requestId });
  }
  return new ApiClientError({
    kind: "http",
    message: STATUS_FALLBACK[status] ?? `The server returned an unexpected response (HTTP ${status}).`,
    status,
    requestId: headerRequestId,
  });
}

export function networkError(): ApiClientError {
  return new ApiClientError({
    kind: "network",
    message: "Could not reach the StegoShield API. Check that the backend is running and that this site's origin is allowed by its CORS settings.",
  });
}

export function timeoutError(): ApiClientError {
  return new ApiClientError({ kind: "timeout", message: "The request took too long and was cancelled. Please try again." });
}

export function abortedError(): ApiClientError {
  return new ApiClientError({ kind: "aborted", message: "The request was cancelled." });
}

/** Normalise anything thrown into an ApiClientError for display. */
export function toApiError(error: unknown): ApiClientError {
  if (error instanceof ApiClientError) return error;
  return new ApiClientError({ kind: "network", message: "Something went wrong. Please try again." });
}
