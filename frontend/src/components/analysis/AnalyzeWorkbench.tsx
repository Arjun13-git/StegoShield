"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Card, CardHeading } from "@/components/ui/Card";
import { ErrorNotice } from "@/components/ui/ErrorNotice";
import { ProgressNote } from "@/components/ui/ProgressNote";
import { ImageDropzone } from "@/components/upload/ImageDropzone";
import { ReplaceImageButton } from "@/components/upload/ReplaceImageButton";
import { SelectedImageCard } from "@/components/upload/SelectedImageCard";
import { useModelInfo } from "@/hooks/useModelInfo";
import { analyzeImage } from "@/lib/api";
import { ApiClientError, toApiError } from "@/lib/errors";
import type { AnalysisResponse } from "@/lib/types";
import { AnalysisResult } from "./AnalysisResult";

type Phase =
  | { kind: "idle" }
  | { kind: "analyzing" }
  | { kind: "success"; analysis: AnalysisResponse }
  | { kind: "error"; error: ApiClientError };

const PROGRESS = ["Validating image…", "Extracting image features…", "Running frozen detector…", "Preparing analysis…"] as const;

export function AnalyzeWorkbench() {
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const modelInfo = useModelInfo();
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  function select(next: File) {
    controller.current?.abort();
    setFile(next);
    setPhase({ kind: "idle" });
  }

  function reset() {
    controller.current?.abort();
    setFile(null);
    setPhase({ kind: "idle" });
  }

  async function analyze() {
    if (!file || phase.kind === "analyzing") return;
    controller.current?.abort();
    const ctl = new AbortController();
    controller.current = ctl;
    setPhase({ kind: "analyzing" });
    try {
      const analysis = await analyzeImage(file, ctl.signal);
      if (!ctl.signal.aborted) setPhase({ kind: "success", analysis });
    } catch (e) {
      const error = toApiError(e);
      if (error.kind !== "aborted") setPhase({ kind: "error", error });
    }
  }

  const busy = phase.kind === "analyzing";

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:items-start">
      <Card className="lg:sticky lg:top-20">
        <CardHeading eyebrow="Input" title="Image to analyze" />
        {file ? (
          <SelectedImageCard file={file} onRemove={reset} disabled={busy} />
        ) : (
          <ImageDropzone onSelect={select} />
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <Button variant="primary" onClick={analyze} disabled={!file || busy} aria-busy={busy}>
            {busy ? "Analyzing…" : phase.kind === "success" ? "Analyze again" : "Analyze image"}
          </Button>
          {file && <ReplaceImageButton onSelect={select} disabled={busy} />}
        </div>
        {busy && (
          <div className="mt-4">
            <ProgressNote messages={PROGRESS} label="Analysis in progress" />
          </div>
        )}
        <p className="mt-4 text-xs leading-relaxed text-muted">
          The image is sent to the StegoShield API for analysis. It is not stored by this interface. Browser-side checks here are
          only a convenience; the server validates every upload.
        </p>
      </Card>

      <div aria-live="polite" className="min-w-0">
        {phase.kind === "success" && <AnalysisResult analysis={phase.analysis} modelInfo={modelInfo} />}
        {phase.kind === "error" && (
          <ErrorNotice
            error={phase.error}
            title="Analysis failed"
            action={
              <Button onClick={analyze} disabled={!file}>
                Try again
              </Button>
            }
          />
        )}
        {phase.kind === "analyzing" && (
          <Card>
            <p className="text-sm text-muted">Analysis in progress. Results appear here.</p>
          </Card>
        )}
        {phase.kind === "idle" && (
          <Card className="border-dashed">
            <p className="text-sm font-medium text-fg">{file ? "Ready to analyze" : "No image selected"}</p>
            <p className="mt-1 text-sm leading-relaxed text-muted">
              {file
                ? "Press “Analyze image” to score it. The result, the evidence behind it and the detector’s limitations will appear here."
                : "Choose a PNG or JPEG on the left. You will get an uncalibrated model score, where it falls among clean reference images, the features behind it, and the detector’s limitations."}
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}
