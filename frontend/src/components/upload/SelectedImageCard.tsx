"use client";

import { useImagePreview } from "@/hooks/useImagePreview";
import { formatBytes } from "@/lib/utils";
import { Button } from "@/components/ui/Button";

/** Preview + file facts for the selected image, with a remove action. */
export function SelectedImageCard({ file, onRemove, disabled = false }: { file: File; onRemove: () => void; disabled?: boolean }) {
  const { preview, loading, decodeFailed } = useImagePreview(file);

  return (
    <div className="flex flex-col gap-4 sm:flex-row">
      <div className="flex h-44 w-full shrink-0 items-center justify-center overflow-hidden rounded-md border border-line bg-canvas sm:w-56">
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element -- local blob: preview, next/image cannot optimise it
          <img src={preview.url} alt={`Preview of ${file.name}`} className="max-h-full max-w-full object-contain [image-rendering:pixelated]" />
        ) : (
          <p className="px-3 text-center text-xs text-muted">
            {loading ? "Loading preview…" : decodeFailed ? "Preview unavailable. The server will still check the file." : ""}
          </p>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
          <dt className="text-muted">File</dt>
          <dd className="truncate text-fg" title={file.name}>
            {file.name}
          </dd>
          <dt className="text-muted">Size</dt>
          <dd className="text-fg">{formatBytes(file.size)}</dd>
          <dt className="text-muted">Type</dt>
          <dd className="text-fg">{file.type || "unknown"}</dd>
          <dt className="text-muted">Dimensions</dt>
          <dd className="text-fg">{preview ? `${preview.width} × ${preview.height} px` : "—"}</dd>
        </dl>
        <Button variant="ghost" className="mt-3 -ml-3" onClick={onRemove} disabled={disabled} aria-label={`Remove ${file.name}`}>
          Remove image
        </Button>
      </div>
    </div>
  );
}
