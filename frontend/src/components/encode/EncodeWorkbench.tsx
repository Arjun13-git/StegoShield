"use client";

import { useEffect, useId, useRef, useState } from "react";
import { AnalysisResult } from "@/components/analysis/AnalysisResult";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { Card, CardHeading } from "@/components/ui/Card";
import { ErrorNotice } from "@/components/ui/ErrorNotice";
import { ProgressNote } from "@/components/ui/ProgressNote";
import { ImageDropzone } from "@/components/upload/ImageDropzone";
import { ReplaceImageButton } from "@/components/upload/ReplaceImageButton";
import { SelectedImageCard } from "@/components/upload/SelectedImageCard";
import { useModelInfo } from "@/hooks/useModelInfo";
import { analyzeImage, encodeImage } from "@/lib/api";
import { ApiClientError, toApiError } from "@/lib/errors";
import type { AnalysisResponse, EncodeResult } from "@/lib/types";
import { CLIENT_LIMITS, checkMessage, formatBytes, utf8ByteLength } from "@/lib/utils";
import { CapacityMeter } from "./CapacityMeter";

const OUTPUT_NAME = "stegoshield-encoded.png";
const PROGRESS = ["Checking the image…", "Embedding the message…", "Preparing the PNG…"] as const;

type EncodePhase =
  | { kind: "idle" }
  | { kind: "encoding" }
  | { kind: "success"; result: EncodeResult; url: string }
  | { kind: "error"; error: ApiClientError };

type AnalyzePhase =
  | { kind: "idle" }
  | { kind: "analyzing" }
  | { kind: "success"; analysis: AnalysisResponse }
  | { kind: "error"; error: ApiClientError };

export function EncodeWorkbench() {
  const messageId = useId();
  const messageHelpId = useId();
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [touched, setTouched] = useState(false);
  const [phase, setPhase] = useState<EncodePhase>({ kind: "idle" });
  const [analysisPhase, setAnalysisPhase] = useState<AnalyzePhase>({ kind: "idle" });
  const modelInfo = useModelInfo();
  const controller = useRef<AbortController | null>(null);

  // Release the generated-image blob URL when it is replaced or the page unmounts.
  const resultUrl = phase.kind === "success" ? phase.url : null;
  useEffect(() => {
    return () => {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [resultUrl]);
  useEffect(() => () => controller.current?.abort(), []);

  const messageProblem = checkMessage(message);
  const messageBytes = utf8ByteLength(message);
  const busy = phase.kind === "encoding";

  function clearOutputs() {
    controller.current?.abort();
    setPhase({ kind: "idle" });
    setAnalysisPhase({ kind: "idle" });
  }

  async function encode(event?: React.FormEvent) {
    event?.preventDefault();
    setTouched(true);
    if (!file || messageProblem || busy) return;
    controller.current?.abort();
    const ctl = new AbortController();
    controller.current = ctl;
    setAnalysisPhase({ kind: "idle" });
    setPhase({ kind: "encoding" });
    try {
      const result = await encodeImage(file, message, ctl.signal);
      if (ctl.signal.aborted) return;
      setPhase({ kind: "success", result, url: URL.createObjectURL(result.blob) });
    } catch (e) {
      const error = toApiError(e);
      if (error.kind !== "aborted") setPhase({ kind: "error", error });
    }
  }

  async function analyzeGenerated() {
    if (phase.kind !== "success" || analysisPhase.kind === "analyzing") return;
    const ctl = new AbortController();
    controller.current = ctl;
    setAnalysisPhase({ kind: "analyzing" });
    try {
      const generated = new File([phase.result.blob], OUTPUT_NAME, { type: "image/png" });
      const analysis = await analyzeImage(generated, ctl.signal);
      if (!ctl.signal.aborted) setAnalysisPhase({ kind: "success", analysis });
    } catch (e) {
      const error = toApiError(e);
      if (error.kind !== "aborted") setAnalysisPhase({ kind: "error", error });
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
        <Card>
          <CardHeading eyebrow="Step 1" title="Cover image" />
          {file ? (
            <>
              <SelectedImageCard
                file={file}
                disabled={busy}
                onRemove={() => {
                  setFile(null);
                  clearOutputs();
                }}
              />
              <div className="mt-3 flex flex-wrap gap-3">
                <ReplaceImageButton
                  disabled={busy}
                  onSelect={(f) => {
                    setFile(f);
                    clearOutputs();
                  }}
                />
              </div>
            </>
          ) : (
            <ImageDropzone
              title="Drop a cover image here, or click to choose"
              onSelect={(f) => {
                setFile(f);
                clearOutputs();
              }}
            />
          )}
        </Card>

        <Card>
          <CardHeading eyebrow="Step 2" title="Message to embed" />
          <form onSubmit={encode} noValidate className="flex flex-col gap-3">
            <label htmlFor={messageId} className="text-sm font-medium text-fg">
              Secret message (plain text)
            </label>
            <textarea
              id={messageId}
              value={message}
              onChange={(e) => {
                setMessage(e.target.value);
                if (phase.kind !== "idle" || analysisPhase.kind !== "idle") clearOutputs();
              }}
              onBlur={() => setTouched(true)}
              rows={5}
              spellCheck={false}
              autoComplete="off"
              disabled={busy}
              aria-invalid={touched && messageProblem !== null}
              aria-describedby={messageHelpId}
              className="w-full resize-y rounded-md border border-line-strong bg-canvas px-3 py-2 font-mono text-sm text-fg placeholder:text-muted/70 focus-visible:border-accent"
              placeholder="Text to hide in the image's least-significant bits"
            />
            <p id={messageHelpId} className="flex flex-wrap justify-between gap-2 text-xs text-muted">
              <span className={touched && messageProblem ? "text-danger" : undefined}>
                {touched && messageProblem ? messageProblem : "Kept in this page only; never saved in the browser."}
              </span>
              <span className="font-mono">
                {messageBytes.toLocaleString("en-US")} / {CLIENT_LIMITS.maxMessageBytes.toLocaleString("en-US")} bytes
              </span>
            </p>

            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" variant="primary" disabled={!file || busy} aria-busy={busy}>
                {busy ? "Encoding…" : "Encode message"}
              </Button>
              {!file && <span className="text-xs text-muted">Choose a cover image first.</span>}
            </div>
            {busy && <ProgressNote messages={PROGRESS} label="Encoding in progress" />}
          </form>

          <Alert variant="info" title="Educational demonstration" className="mt-4">
            The message is written unencrypted into pixel least-significant bits and saved as a PNG. Anyone can read it back, so
            this is not a secure channel. Re-saving as JPEG or resizing destroys it.
          </Alert>
        </Card>
      </div>

      <div aria-live="polite" className="flex flex-col gap-6">
        {phase.kind === "error" && (
          <ErrorNotice
            error={phase.error}
            title="Encoding failed"
            action={
              <Button onClick={() => void encode()} disabled={!file || busy}>
                Try again
              </Button>
            }
          />
        )}

        {phase.kind === "success" && (
          <Card>
            <CardHeading eyebrow="Result" title="Stego image generated" />
            <div className="grid gap-5 md:grid-cols-[minmax(0,16rem)_1fr]">
              <div className="flex h-56 items-center justify-center overflow-hidden rounded-md border border-line bg-canvas">
                {/* eslint-disable-next-line @next/next/no-img-element -- generated blob: URL */}
                <img src={phase.url} alt="Generated stego image" className="max-h-full max-w-full object-contain [image-rendering:pixelated]" />
              </div>
              <div className="flex flex-col gap-4">
                {phase.result.embeddedBytes !== null && phase.result.capacityBytes !== null ? (
                  <CapacityMeter embedded={phase.result.embeddedBytes} capacity={phase.result.capacityBytes} />
                ) : (
                  <p className="text-sm text-muted">Embedded size and capacity were not reported by the server.</p>
                )}
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                  <dt className="text-muted">Output</dt>
                  <dd className="text-fg">PNG · {formatBytes(phase.result.blob.size)}</dd>
                  {phase.result.requestId && (
                    <>
                      <dt className="text-muted">Request id</dt>
                      <dd className="break-all font-mono text-xs text-fg">{phase.result.requestId}</dd>
                    </>
                  )}
                </dl>
                <div className="flex flex-wrap gap-3">
                  <a
                    href={phase.url}
                    download={OUTPUT_NAME}
                    className="inline-flex min-h-10 items-center justify-center rounded-md bg-accent px-4 py-2 text-sm font-medium text-canvas hover:bg-accent-strong"
                  >
                    Download PNG
                  </a>
                  <Button onClick={analyzeGenerated} disabled={analysisPhase.kind === "analyzing"} aria-busy={analysisPhase.kind === "analyzing"}>
                    {analysisPhase.kind === "analyzing" ? "Analyzing…" : "Analyze generated image"}
                  </Button>
                </div>
                {analysisPhase.kind === "analyzing" && <ProgressNote messages={["Analyzing the generated image…"]} label="Analysis in progress" />}
              </div>
            </div>
          </Card>
        )}

        {analysisPhase.kind === "error" && <ErrorNotice error={analysisPhase.error} title="Analysis failed" />}
        {analysisPhase.kind === "success" && (
          <div className="flex flex-col gap-3">
            <Alert variant="info" title="Detector result for the generated image">
              A single result is anecdotal. The detector is weak in testing, so this image may score as clean or as elevated;
              neither outcome shows it is undetectable or detectable in general.
            </Alert>
            <AnalysisResult analysis={analysisPhase.analysis} modelInfo={modelInfo} />
          </div>
        )}
      </div>
    </div>
  );
}
