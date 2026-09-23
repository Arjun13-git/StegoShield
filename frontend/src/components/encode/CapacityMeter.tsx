import { formatBytes } from "@/lib/utils";

/** Message bytes embedded vs. the image's capacity, both as reported by the API. */
export function CapacityMeter({ embedded, capacity }: { embedded: number; capacity: number }) {
  const ratio = capacity > 0 ? Math.min(1, embedded / capacity) : 0;
  return (
    <div>
      <p className="text-sm font-medium text-fg">Payload used</p>
      <div
        role="meter"
        aria-label="Embedded message bytes compared with image capacity"
        aria-valuemin={0}
        aria-valuemax={capacity}
        aria-valuenow={embedded}
        aria-valuetext={`${embedded} of ${capacity} bytes`}
        className="mt-2 h-2.5 overflow-hidden rounded-full bg-line"
      >
        <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(ratio * 100, embedded > 0 ? 1.5 : 0)}%` }} />
      </div>
      <p className="mt-1.5 text-xs text-muted">
        <span className="font-mono text-fg">{embedded.toLocaleString("en-US")}</span> bytes embedded of{" "}
        <span className="font-mono text-fg">{capacity.toLocaleString("en-US")}</span> bytes capacity ({(ratio * 100).toFixed(2)}%,{" "}
        {formatBytes(capacity)} available).
      </p>
    </div>
  );
}
