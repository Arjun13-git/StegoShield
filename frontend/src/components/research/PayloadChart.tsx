import { research, fixed } from "@/lib/research";

/** Horizontal bars of ROC-AUC per payload on a fixed 0-1 axis, with the 0.5 chance level marked. */
export function PayloadChart() {
  const rows = research.payload_sensitivity.results;
  return (
    <figure>
      <figcaption className="mb-3 text-xs text-muted">ROC-AUC of the frozen model vs. embedding payload (bits per pixel). The vertical line is chance (0.5).</figcaption>
      <ul className="flex flex-col gap-2.5">
        {rows.map((r) => (
          <li key={r.payload_bpp} className="grid grid-cols-[4.5rem_1fr_3.5rem] items-center gap-3 text-sm">
            <span className="font-mono text-muted">{r.payload_bpp.toFixed(2)} bpp</span>
            <div
              role="img"
              aria-label={`ROC-AUC ${fixed(r.roc_auc)} at ${r.payload_bpp.toFixed(2)} bits per pixel`}
              className="relative h-3 rounded-sm bg-line"
            >
              <div className="h-full rounded-sm bg-accent/70" style={{ width: `${r.roc_auc * 100}%` }} />
              <span className="absolute inset-y-[-3px] left-1/2 w-px bg-fg/60" aria-hidden="true" />
            </div>
            <span className="text-right font-mono tabular-nums text-fg">{fixed(r.roc_auc)}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}
