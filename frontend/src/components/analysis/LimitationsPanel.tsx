import { Card } from "@/components/ui/Card";

/** Native <details>: keyboard/screen-reader accessible, open by default so limits are never hidden. */
export function LimitationsPanel({ limitations }: { limitations: string[] }) {
  return (
    <Card className="p-0 sm:p-0">
      <details open className="group">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 sm:px-5">
          <span>
            <span className="block font-mono text-[11px] uppercase tracking-[0.14em] text-muted">Read before relying on this result</span>
            <span className="block text-base font-semibold sm:text-lg">What this detector cannot tell you</span>
          </span>
          <span className="text-xs text-muted group-open:hidden">Show</span>
          <span className="hidden text-xs text-muted group-open:inline">Hide</span>
        </summary>
        <ul className="list-disc space-y-2 border-t border-line px-4 py-4 pl-9 text-sm leading-relaxed text-fg/90 sm:px-5 sm:pl-10">
          {limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </details>
    </Card>
  );
}
