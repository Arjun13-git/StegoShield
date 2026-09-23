"use client";

import { useEffect, useState } from "react";
import { getReady } from "@/lib/api";

export type ApiStatus = "checking" | "ready" | "model_unavailable" | "unreachable";

const POLL_MS = 30_000;

/** Polls GET /ready. "model_unavailable" = API up but the frozen model is not loaded. */
export function useApiStatus(): ApiStatus {
  const [status, setStatus] = useState<ApiStatus>("checking");

  useEffect(() => {
    const controller = new AbortController();
    async function check() {
      try {
        const ready = await getReady(controller.signal);
        setStatus(ready.model_loaded ? "ready" : "model_unavailable");
      } catch {
        if (!controller.signal.aborted) setStatus("unreachable");
      }
    }
    void check();
    const id = setInterval(check, POLL_MS);
    return () => {
      controller.abort();
      clearInterval(id);
    };
  }, []);

  return status;
}
