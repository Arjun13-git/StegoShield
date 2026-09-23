"use client";

import { useEffect, useState } from "react";
import { getModelInfo } from "@/lib/api";
import { toApiError } from "@/lib/errors";
import type { ModelInfoResponse } from "@/lib/types";

export type ModelInfoState =
  | { status: "loading" }
  | { status: "ready"; info: ModelInfoResponse }
  | { status: "error"; message: string; requestId: string | null };

/** Fetches GET /api/v1/model-info once. */
export function useModelInfo(): ModelInfoState {
  const [state, setState] = useState<ModelInfoState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    getModelInfo(controller.signal)
      .then((info) => setState({ status: "ready", info }))
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        const err = toApiError(e);
        setState({ status: "error", message: err.message, requestId: err.requestId });
      });
    return () => controller.abort();
  }, []);

  return state;
}
