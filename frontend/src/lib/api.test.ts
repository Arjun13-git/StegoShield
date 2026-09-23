import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { analyzeImage, encodeImage, getApiBaseUrl, getReady } from "./api";
import { ApiClientError } from "./errors";
import { analysisFixture } from "@/test/fixtures";

const fetchMock = vi.fn();

function json(body: unknown, status = 200, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
}

async function thrown(promise: Promise<unknown>): Promise<ApiClientError> {
  try {
    await promise;
  } catch (e) {
    return e as ApiClientError;
  }
  throw new Error("expected the call to reject");
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://api.test:9000/");
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
});

describe("base URL", () => {
  it("comes from NEXT_PUBLIC_API_BASE_URL without a trailing slash, with a local default", () => {
    expect(getApiBaseUrl()).toBe("http://api.test:9000");
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "");
    expect(getApiBaseUrl()).toBe("http://127.0.0.1:8000");
  });
});

describe("analyzeImage", () => {
  it("POSTs the file as multipart field 'file' and returns the typed response", async () => {
    fetchMock.mockResolvedValue(json(analysisFixture));
    const file = new File(["png-bytes"], "a.png", { type: "image/png" });
    const result = await analyzeImage(file);

    expect(result.stego_score).toBe(52.57);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://api.test:9000/api/v1/analyze");
    expect(init.method).toBe("POST");
    expect(init.credentials).toBe("omit");
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
    expect(((init.body as FormData).get("file") as File).name).toBe("a.png");
  });

  it.each([
    [400, "invalid_image", /PNG or JPEG/],
    [413, "file_too_large", /larger than the server accepts/],
    [503, "model_unavailable", /not available/],
    [500, "analysis_failed", /failed on the server/],
    [422, "invalid_request", /malformed/],
  ])("turns a %s %s response into a safe error carrying the request id", async (status, code, message) => {
    fetchMock.mockResolvedValue(json({ detail: "fixed", code, request_id: "req-9" }, status));
    const err = await thrown(analyzeImage(new File(["x"], "a.png")));
    expect(err).toBeInstanceOf(ApiClientError);
    expect(err).toMatchObject({ kind: "http", status, code, requestId: "req-9" });
    expect(err.message).toMatch(message);
  });

  it("does not leak an HTML error page", async () => {
    fetchMock.mockResolvedValue(new Response("<html><body>Traceback /srv/app.py</body></html>", { status: 502 }));
    const err = await thrown(analyzeImage(new File(["x"], "a.png")));
    expect(err.message).toBe("The server is temporarily unreachable.");
    expect(err.message).not.toMatch(/Traceback|srv/);
  });

  it("reports a network failure", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await thrown(analyzeImage(new File(["x"], "a.png")));
    expect(err).toMatchObject({ kind: "network", status: null });
    expect(err.message).toMatch(/Could not reach the StegoShield API/);
  });

  it("times out a request that never answers", async () => {
    vi.useFakeTimers();
    fetchMock.mockImplementation((_url: string, init: RequestInit) => new Promise((_, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    }));
    const pending = thrown(analyzeImage(new File(["x"], "a.png")));
    await vi.advanceTimersByTimeAsync(61_000);
    expect(await pending).toMatchObject({ kind: "timeout" });
  });

  it("reports caller cancellation as 'aborted', not as a network error", async () => {
    const ctl = new AbortController();
    fetchMock.mockImplementation((_url: string, init: RequestInit) => new Promise((_, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    }));
    const pending = thrown(analyzeImage(new File(["x"], "a.png"), ctl.signal));
    ctl.abort();
    expect(await pending).toMatchObject({ kind: "aborted" });
  });
});

describe("encodeImage", () => {
  it("sends file and message, and reads the PNG plus capacity headers", async () => {
    fetchMock.mockResolvedValue(
      new Response(new Uint8Array([137, 80, 78, 71]), {
        status: 200,
        headers: { "Content-Type": "image/png", "X-Embedded-Bytes": "14", "X-Capacity-Bytes": "6136", "X-Request-ID": "enc-1" },
      }),
    );
    const result = await encodeImage(new File(["x"], "c.png", { type: "image/png" }), "hello");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://api.test:9000/api/v1/encode");
    expect((init.body as FormData).get("message")).toBe("hello");
    expect(result).toMatchObject({ embeddedBytes: 14, capacityBytes: 6136, requestId: "enc-1" });
    expect(result.blob.size).toBe(4);
  });

  it("ignores malformed capacity headers instead of showing nonsense", async () => {
    fetchMock.mockResolvedValue(new Response(new Uint8Array([1]), { headers: { "X-Embedded-Bytes": "abc", "X-Capacity-Bytes": "-3" } }));
    const result = await encodeImage(new File(["x"], "c.png"), "m");
    expect(result.embeddedBytes).toBeNull();
    expect(result.capacityBytes).toBeNull();
  });

  it("maps the capacity error", async () => {
    fetchMock.mockResolvedValue(json({ detail: "d", code: "payload_exceeds_capacity", request_id: "r" }, 400));
    const err = await thrown(encodeImage(new File(["x"], "c.png"), "too long"));
    expect(err.message).toMatch(/does not fit/);
  });
});

describe("getReady", () => {
  it("treats a 503 ReadyResponse as a normal 'not ready' answer", async () => {
    fetchMock.mockResolvedValue(json({ status: "not_ready", model_loaded: false, reference_available: false, reason: "artifact_missing" }, 503));
    expect(await getReady()).toMatchObject({ model_loaded: false, reason: "artifact_missing" });
  });

  it("reports ready", async () => {
    fetchMock.mockResolvedValue(json({ status: "ready", model_loaded: true, reference_available: true, reason: null }));
    expect(await getReady()).toMatchObject({ model_loaded: true, reference_available: true });
  });

  it("errors on an unrelated 503 body", async () => {
    fetchMock.mockResolvedValue(new Response("upstream down", { status: 503 }));
    await expect(getReady()).rejects.toBeInstanceOf(ApiClientError);
  });
});
