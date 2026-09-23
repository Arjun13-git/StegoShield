import { Card, CardHeading } from "@/components/ui/Card";
import type { Direction, FeatureIndicator } from "@/lib/types";

const DIRECTION_TEXT: Record<Direction, string> = {
  increases_stego_signal: "Deviates like the LSB-stego reference",
  decreases_stego_signal: "Deviates against the LSB-stego reference",
  informational: "Informational",
};

const DIRECTION_MARK: Record<Direction, string> = {
  increases_stego_signal: "▲",
  decreases_stego_signal: "▼",
  informational: "●",
};

function formatDeviation(z: number | null): string {
  if (z === null) return "n/a";
  return `${z >= 0 ? "+" : "−"}${Math.abs(z).toFixed(2)} σ`;
}

export function IndicatorList({ indicators, method }: { indicators: FeatureIndicator[]; method: string }) {
  return (
    <Card>
      <CardHeading eyebrow="Explanation" title="Heuristic model evidence" />
      <p className="mb-4 text-sm leading-relaxed text-muted">
        These are the features that mattered most to the model for this image, ranked by a heuristic. They are not proof, not a
        causal explanation, and do not show where any hidden data would be.
      </p>
      {indicators.length === 0 ? (
        <p className="text-sm text-muted">No indicators were returned for this image.</p>
      ) : (
        <ol className="divide-y divide-line">
          {indicators.map((ind) => (
            <li key={ind.name} className="grid gap-x-4 gap-y-1 py-3 sm:grid-cols-[1fr_auto]">
              <div className="min-w-0">
                <p className="text-sm font-medium text-fg">{ind.label}</p>
                <p className="break-all font-mono text-[11px] text-muted">{ind.name}</p>
              </div>
              <dl className="grid grid-cols-[auto_auto] justify-start gap-x-4 gap-y-0.5 text-xs sm:justify-end sm:text-right">
                <dt className="text-muted">Value</dt>
                <dd className="font-mono text-fg">{ind.value.toFixed(4)}</dd>
                <dt className="text-muted">vs clean reference</dt>
                <dd className="font-mono text-fg">{formatDeviation(ind.deviation_from_reference_cover)}</dd>
                <dt className="text-muted">Model importance</dt>
                <dd className="font-mono text-fg">{ind.model_importance === null ? "n/a" : ind.model_importance.toFixed(4)}</dd>
              </dl>
              <p className="text-xs text-muted sm:col-span-2">
                <span aria-hidden="true">{DIRECTION_MARK[ind.direction]} </span>
                {DIRECTION_TEXT[ind.direction]}
              </p>
            </li>
          ))}
        </ol>
      )}
      <p className="mt-3 border-t border-line pt-3 text-xs leading-relaxed text-muted">{method}</p>
    </Card>
  );
}
