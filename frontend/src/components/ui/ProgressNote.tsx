"use client";

import { useEffect, useState } from "react";
import { Spinner } from "./Spinner";

/**
 * Generic, time-based progress wording. The API exposes no progress events, so
 * these messages do NOT correspond to specific backend steps.
 */
export function ProgressNote({ messages, label }: { messages: readonly string[]; label: string }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setIndex((i) => Math.min(i + 1, messages.length - 1)), 1600);
    return () => clearInterval(id);
  }, [messages.length]);

  return (
    <div className="flex items-center gap-3 text-sm text-muted">
      <Spinner />
      <span aria-hidden="true">{messages[index]}</span>
      <span className="sr-only" role="status">
        {label}
      </span>
    </div>
  );
}
