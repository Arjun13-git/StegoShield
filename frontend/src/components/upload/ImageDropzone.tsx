"use client";

import { useId, useRef, useState } from "react";
import { CLIENT_LIMITS, checkImageFile, formatBytes } from "@/lib/utils";
import { cn } from "@/lib/utils";

/**
 * Drag-and-drop / file-picker for one image. The native file input stays in the
 * accessibility tree (keyboard + screen readers); the surrounding label is the
 * drop target. Checks here are advisory: the backend validates authoritatively.
 */
export function ImageDropzone({
  onSelect,
  disabled = false,
  title = "Drop an image here, or click to choose",
}: {
  onSelect: (file: File) => void;
  disabled?: boolean;
  title?: string;
}) {
  const inputId = useId();
  const errorId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  function accept(file: File | undefined) {
    if (!file) return;
    const issue = checkImageFile(file);
    setProblem(issue);
    if (issue === null) onSelect(file);
    if (inputRef.current) inputRef.current.value = ""; // allow re-selecting the same file
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files[0]);
        }}
        className={cn(
          "flex min-h-44 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-4 py-8 text-center transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent",
          dragging ? "border-accent bg-accent/10" : "border-line-strong bg-surface-2/40 hover:border-accent",
          disabled && "cursor-not-allowed opacity-60 hover:border-line-strong",
        )}
      >
        <svg viewBox="0 0 24 24" className="h-8 w-8 text-muted" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <path d="M12 16V5m0 0-4 4m4-4 4 4M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="text-sm font-medium text-fg">{title}</span>
        <span className="text-xs text-muted">PNG or JPEG · up to about {formatBytes(CLIENT_LIMITS.maxUploadBytes)}</span>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={CLIENT_LIMITS.accept}
          disabled={disabled}
          className="sr-only"
          aria-describedby={problem ? errorId : undefined}
          onChange={(e) => accept(e.target.files?.[0])}
        />
      </label>
      {problem && (
        <p id={errorId} role="alert" className="mt-2 text-sm text-danger">
          {problem}
        </p>
      )}
    </div>
  );
}
