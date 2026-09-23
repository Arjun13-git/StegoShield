import { Badge, type Tone } from "@/components/ui/Badge";
import { Card, CardHeading } from "@/components/ui/Card";
import type { AnalysisResponse, ModelInfoResponse, RiskLevel } from "@/lib/types";
import { formatScore, riskLabel, verdictLabel } from "@/lib/utils";

const RISK_TONE: Record<RiskLevel, Tone> = { low: "neutral", elevated: "warn", high: "warn", not_assessed: "neutral" };

/** Marker position on a 0-100 track. */
function PercentileMeter({ percentile, levels }: { percentile: number; levels: ModelInfoResponse["risk_levels"] | null }) {
  const pos = Math.min(100, Math.max(0, percentile));
  const ticks = levels
    ? Object.entries(levels.levels).map(([name, def]) => ({ name, at: (def?.cover_percentile_at_or_above ?? 0) * 100 }))
    : [];
  return (
    <div>
      <div
        role="meter"
        aria-label="Position among reference clean-image scores"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pos * 10) / 10}
        aria-valuetext={`${pos.toFixed(1)}th percentile of reference clean images`}
        className="relative mt-2 h-2.5 rounded-full bg-line"
      >
        <div className="absolute inset-y-0 left-0 rounded-full bg-accent/50" style={{ width: `${pos}%` }} />
        {ticks.map((t) => (
          <span key={t.name} className="absolute -top-1 h-4.5 w-px bg-fg/50" style={{ left: `${t.at}%` }} aria-hidden="true" />
        ))}
        <span
          className="absolute top-1/2 h-4 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-sm bg-fg ring-2 ring-canvas"
          style={{ left: `${pos}%` }}
          aria-hidden="true"
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] text-muted">
        <span>0 · lowest of reference</span>
        <span>100 · highest of reference</span>
      </div>
      {ticks.length > 0 && (
        <p className="mt-1 text-[11px] text-muted">
          Vertical marks: {ticks.map((t) => `${t.name} from the ${t.at.toFixed(0)}th percentile`).join(", ")}.
        </p>
      )}
    </div>
  );
}

export function ScoreCard({ analysis, riskLevels }: { analysis: AnalysisResponse; riskLevels: ModelInfoResponse["risk_levels"] | null }) {
  return (
    <Card>
      <CardHeading eyebrow="Analysis result" title="Detection signal" />

      <div className="grid gap-5 sm:grid-cols-[auto_1fr] sm:items-start">
        <div>
          <p className="font-mono text-5xl font-semibold tabular-nums tracking-tight text-fg">{formatScore(analysis.stego_score)}</p>
          <p className="mt-1 text-sm font-medium text-fg">Stego score</p>
          <p className="text-xs text-muted">Uncalibrated model score, 0–100.</p>
          <p className="text-xs text-muted">Not a probability of hidden data.</p>
        </div>

        <div className="min-w-0">
          <dl className="grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted">Risk level</dt>
            <dd>
              <Badge tone={RISK_TONE[analysis.risk_level]}>{riskLabel(analysis.risk_level)}</Badge>
            </dd>
            <dt className="text-muted">Verdict</dt>
            <dd className="font-medium text-fg">{verdictLabel(analysis.verdict)}</dd>
            <dt className="text-muted">Reference percentile</dt>
            <dd className="font-mono text-fg">
              {analysis.cover_reference_percentile === null ? "Not available" : analysis.cover_reference_percentile.toFixed(1)}
            </dd>
          </dl>

          {analysis.cover_reference_percentile !== null ? (
            <div className="mt-3">
              <p className="text-xs text-muted">
                The score is at or above {analysis.cover_reference_percentile.toFixed(1)}% of reference clean images (Phase 1 validation split).
              </p>
              <PercentileMeter percentile={analysis.cover_reference_percentile} levels={riskLevels} />
            </div>
          ) : (
            <p className="mt-3 text-xs text-muted">
              No reference distribution matches this model, so the score cannot be placed on a scale and the verdict is inconclusive.
            </p>
          )}
        </div>
      </div>

      <p className="mt-4 border-t border-line pt-3 text-sm leading-relaxed text-muted">{analysis.risk_context}</p>
    </Card>
  );
}
