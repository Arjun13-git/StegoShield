"use client";

import { useApiStatus, type ApiStatus } from "@/hooks/useApiStatus";
import { cn } from "@/lib/utils";

const COPY: Record<ApiStatus, { label: string; dot: string }> = {
  checking: { label: "Checking API…", dot: "bg-muted animate-pulse" },
  ready: { label: "API ready · model loaded", dot: "bg-ok" },
  model_unavailable: { label: "API up · model unavailable", dot: "bg-warn" },
  unreachable: { label: "API unreachable", dot: "bg-danger" },
};

export function StatusIndicator({ className }: { className?: string }) {
  const status = useApiStatus();
  const { label, dot } = COPY[status];
  return (
    <p className={cn("inline-flex items-center gap-2 text-xs text-muted", className)} role="status">
      <span className={cn("h-2 w-2 rounded-full", dot)} aria-hidden="true" />
      <span>{label}</span>
    </p>
  );
}
