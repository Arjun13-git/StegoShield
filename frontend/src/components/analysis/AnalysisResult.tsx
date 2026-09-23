import { CopyButton } from "@/components/ui/CopyButton";
import type { ModelInfoState } from "@/hooks/useModelInfo";
import type { AnalysisResponse } from "@/lib/types";
import { ImageInfoCard } from "./ImageInfoCard";
import { IndicatorList } from "./IndicatorList";
import { LimitationsPanel } from "./LimitationsPanel";
import { ModelInfoCard } from "./ModelInfoCard";
import { ScoreCard } from "./ScoreCard";
import { WarningsList } from "./WarningsList";

export function AnalysisResult({ analysis, modelInfo }: { analysis: AnalysisResponse; modelInfo: ModelInfoState }) {
  return (
    <div className="flex flex-col gap-4">
      <ScoreCard analysis={analysis} riskLevels={modelInfo.status === "ready" ? modelInfo.info.risk_levels : null} />
      <WarningsList warnings={analysis.warnings} />
      <LimitationsPanel limitations={analysis.limitations} />
      <IndicatorList indicators={analysis.indicators} method={analysis.explanation_method} />
      <div className="grid gap-4 md:grid-cols-2">
        <ImageInfoCard image={analysis.image} />
        <ModelInfoCard state={modelInfo} />
      </div>
      <p className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
        <span>
          {analysis.model} {analysis.model_version} · feature schema {analysis.feature_schema_version} · {analysis.feature_count} features
        </span>
        <span>· request id {analysis.request_id}</span>
        <CopyButton value={analysis.request_id} label="Copy request ID" />
      </p>
    </div>
  );
}
