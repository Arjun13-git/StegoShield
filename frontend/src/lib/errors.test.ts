import { describe, expect, it } from "vitest";
import { ApiClientError, errorFromResponse, isApiErrorBody, networkError, timeoutError, toApiError } from "./errors";

describe("errorFromResponse", () => {
  it("uses friendly copy for known codes and keeps the request id from the body", () => {
    const err = errorFromResponse(400, { detail: "raw", code: "invalid_image", request_id: "abc" }, "header-id");
    expect(err.message).toMatch(/PNG or JPEG/);
    expect(err).toMatchObject({ kind: "http", status: 400, code: "invalid_image", requestId: "abc" });
  });

  it.each([
    [413, "file_too_large", /larger than the server accepts/],
    [400, "payload_exceeds_capacity", /does not fit/],
    [400, "missing_message", /Enter a message/],
    [503, "model_unavailable", /not available/],
    [422, "invalid_request", /malformed/],
  ])("maps %s %s", (status, code, pattern) => {
    expect(errorFromResponse(status, { detail: "x", code, request_id: "r" }, null).message).toMatch(pattern);
  });

  it("falls back to the backend's own message for an unknown code, and to the header request id", () => {
    const err = errorFromResponse(400, { detail: "Something specific.", code: "brand_new_code" } as never, "hdr");
    expect(err.message).toBe("Something specific.");
    expect(err.requestId).toBe("hdr");
  });

  it("never shows an over-long or non-JSON body; uses a status-based sentence instead", () => {
    const long = errorFromResponse(500, { detail: "x".repeat(500), code: "weird", request_id: "r" }, null);
    expect(long.message).toBe("The server hit an unexpected error.");
    const html = errorFromResponse(502, null, null);
    expect(html.message).toBe("The server is temporarily unreachable.");
    expect(errorFromResponse(418, "<html>Traceback (most recent call last)</html>", null).message).toBe(
      "The server returned an unexpected response (HTTP 418).",
    );
  });
});

describe("helpers", () => {
  it("recognises the backend error body shape", () => {
    expect(isApiErrorBody({ detail: "d", code: "c", request_id: "r" })).toBe(true);
    expect(isApiErrorBody({ detail: "d" })).toBe(false);
    expect(isApiErrorBody(null)).toBe(false);
  });

  it("builds network and timeout errors without status codes", () => {
    expect(networkError()).toMatchObject({ kind: "network", status: null });
    expect(timeoutError()).toMatchObject({ kind: "timeout" });
  });

  it("wraps unknown thrown values in a generic, safe ApiClientError", () => {
    const err = toApiError(new Error("Traceback: /home/user/secret.py"));
    expect(err).toBeInstanceOf(ApiClientError);
    expect(err.message).not.toMatch(/Traceback|secret/);
  });
});
