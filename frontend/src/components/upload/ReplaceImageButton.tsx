"use client";

import { useId, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { CLIENT_LIMITS, checkImageFile } from "@/lib/utils";

/** “Choose a different image” with the same advisory checks as the dropzone. */
export function ReplaceImageButton({ onSelect, disabled = false }: { onSelect: (file: File) => void; disabled?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const errorId = useId();
  const [problem, setProblem] = useState<string | null>(null);

  return (
    <>
      <Button onClick={() => inputRef.current?.click()} disabled={disabled} aria-describedby={problem ? errorId : undefined}>
        Choose a different image
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept={CLIENT_LIMITS.accept}
        className="sr-only"
        tabIndex={-1}
        aria-label="Choose a different image"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          const issue = checkImageFile(f);
          setProblem(issue);
          if (issue === null) onSelect(f);
        }}
      />
      {problem && (
        <p id={errorId} role="alert" className="basis-full text-sm text-danger">
          {problem}
        </p>
      )}
    </>
  );
}
