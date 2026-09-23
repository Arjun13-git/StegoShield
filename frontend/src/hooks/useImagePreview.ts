"use client";

import { useEffect, useState } from "react";

export interface ImagePreview {
  url: string;
  width: number;
  height: number;
}

interface Loaded {
  file: File;
  preview: ImagePreview | null; // null: the browser could not decode it (the backend still decides validity)
}

/**
 * Local preview of a selected image via a blob: URL (never sent anywhere, revoked
 * on change/unmount). Returns `loading` until the browser has decoded the image.
 */
export function useImagePreview(file: File | null): { preview: ImagePreview | null; loading: boolean; decodeFailed: boolean } {
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => setLoaded({ file, preview: { url, width: img.naturalWidth, height: img.naturalHeight } });
    img.onerror = () => setLoaded({ file, preview: null });
    img.src = url;
    return () => {
      img.onload = null;
      img.onerror = null;
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const current = loaded && loaded.file === file ? loaded : null;
  return {
    preview: current?.preview ?? null,
    loading: file !== null && current === null,
    decodeFailed: current !== null && current.preview === null,
  };
}
