import type { ApiClientError } from "@/lib/errors";
import { Alert } from "./Alert";
import { CopyButton } from "./CopyButton";

/** User-safe error display: friendly message, request ID when the backend gave one, optional retry. */
export function ErrorNotice({ error, title = "Request failed", action }: { error: ApiClientError; title?: string; action?: React.ReactNode }) {
  return (
    <Alert variant="error" title={title} role="alert">
      <p className="text-fg/90">{error.message}</p>
      {(error.requestId || error.code) && (
        <p className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs">
          {error.status !== null && <span>HTTP {error.status}</span>}
          {error.code && <span>{error.code}</span>}
          {error.requestId && (
            <span className="inline-flex items-center gap-2">
              request id: <span className="break-all text-fg">{error.requestId}</span>
              <CopyButton value={error.requestId} label="Copy request ID" />
            </span>
          )}
        </p>
      )}
      {action && <div className="mt-3">{action}</div>}
    </Alert>
  );
}
