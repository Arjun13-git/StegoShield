"use client";

import { Card, CardHeading } from "@/components/ui/Card";
import { CopyButton } from "@/components/ui/CopyButton";
import type { ModelInfoState } from "@/hooks/useModelInfo";
import { percent, shortHash } from "@/lib/utils";

/**
 * Deployed-model facts from GET /api/v1/model-info. `detailed` adds the stored
 * in-domain test metrics and the risk-level definitions (Research page).
 */
export function ModelInfoCard({ state, detailed = false }: { state: ModelInfoState; detailed?: boolean }) {
  return (
    <Card>
      <CardHeading eyebrow="Model" title="Deployed detector" />
      {state.status === "loading" && <p className="text-sm text-muted">Loading model information…</p>}
      {state.status === "error" && (
        <p className="text-sm text-muted" role="status">
          Model information is unavailable: {state.message}
          {state.requestId && <span className="font-mono text-xs"> (request id {state.requestId})</span>}
        </p>
      )}
      {state.status === "ready" && (
        <>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
            <dt className="text-muted">Model</dt>
            <dd className="text-fg">
              {state.info.name} {state.info.version}
            </dd>
            <dt className="text-muted">Scope</dt>
            <dd className="text-fg">{state.info.scope}</dd>
            <dt className="text-muted">Feature schema</dt>
            <dd className="text-fg">
              {state.info.feature_schema} · {state.info.feature_count} features
            </dd>
            <dt className="text-muted">Training data</dt>
            <dd className="break-all font-mono text-xs text-fg">{state.info.training_dataset_id ?? "unknown"}</dd>
            <dt className="text-muted">Artifact SHA-256</dt>
            <dd className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs text-fg" title={state.info.artifact_sha256}>
                {shortHash(state.info.artifact_sha256)}
              </span>
              <CopyButton value={state.info.artifact_sha256} label="Copy full model SHA-256" />
            </dd>
          </dl>

          <p className="mt-3 text-xs leading-relaxed text-muted">{state.info.score_interpretation}</p>

          {detailed && state.info.reference_performance && (
            <div className="mt-4 border-t border-line pt-3">
              <h3 className="text-sm font-medium text-fg">Stored in-domain test metrics</h3>
              <p className="mb-2 text-xs text-muted">
                BOSSBase held-out test split, controlled LSB at 0.10 bits per pixel ({state.info.reference_performance.n_samples ?? "?"} images).
              </p>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-5">
                {(["accuracy", "precision", "recall", "f1", "roc_auc"] as const).map((k) => {
                  const v = state.info.reference_performance?.[k];
                  return (
                    <div key={k}>
                      <dt className="text-xs text-muted">{k === "roc_auc" ? "ROC-AUC" : k === "f1" ? "F1" : k[0].toUpperCase() + k.slice(1)}</dt>
                      <dd className="font-mono text-fg">{typeof v === "number" ? v.toFixed(3) : "n/a"}</dd>
                    </div>
                  );
                })}
              </dl>
            </div>
          )}

          {detailed && state.info.risk_levels && (
            <div className="mt-4 border-t border-line pt-3">
              <h3 className="text-sm font-medium text-fg">Risk levels</h3>
              <p className="mb-2 text-xs leading-relaxed text-muted">
                {state.info.risk_levels.definition} {state.info.risk_levels.reference_stego}
              </p>
              <div className="overflow-x-auto">
                <table className="w-full min-w-80 text-left text-sm">
                  <caption className="sr-only">Risk level flag rates on the held-out test split</caption>
                  <thead className="text-xs text-muted">
                    <tr>
                      <th scope="col" className="py-1 pr-3 font-normal">Level</th>
                      <th scope="col" className="py-1 pr-3 font-normal">Reference percentile</th>
                      <th scope="col" className="py-1 pr-3 font-normal">Clean images flagged</th>
                      <th scope="col" className="py-1 font-normal">0.10 bpp stego flagged</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(state.info.risk_levels.levels).map(([name, def]) =>
                      def ? (
                        <tr key={name} className="border-t border-line">
                          <th scope="row" className="py-1.5 pr-3 font-medium capitalize text-fg">{name}</th>
                          <td className="py-1.5 pr-3 font-mono text-fg">≥ {(def.cover_percentile_at_or_above * 100).toFixed(0)}th</td>
                          <td className="py-1.5 pr-3 font-mono text-fg">{percent(def.flag_rate_held_out_test.clean)}</td>
                          <td className="py-1.5 font-mono text-fg">{percent(def.flag_rate_held_out_test.lsb_stego)}</td>
                        </tr>
                      ) : null,
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
