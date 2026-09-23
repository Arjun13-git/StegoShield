"use client";

import { ModelInfoCard } from "@/components/analysis/ModelInfoCard";
import { useModelInfo } from "@/hooks/useModelInfo";

/** Live GET /api/v1/model-info, in its detailed form. */
export function ResearchModelPanel() {
  return <ModelInfoCard state={useModelInfo()} detailed />;
}
