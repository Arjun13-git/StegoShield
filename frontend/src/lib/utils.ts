import type { RiskLevel, Verdict } from "./types";

/** Client-side limits for USER EXPERIENCE only; the backend is authoritative. */
export const CLIENT_LIMITS = {
  maxUploadBytes: 10 * 1024 * 1024,
  maxMessageBytes: 16 * 1024,
  accept: "image/png,image/jpeg",
} as const;

const ALLOWED_TYPES = new Set(["image/png", "image/jpeg"]);

/** Advisory pre-check. Returns a user-facing problem, or null if nothing obviously wrong. */
export function checkImageFile(file: { type: string; size: number; name: string }): string | null {
  if (file.size === 0) return "This file is empty.";
  if (!ALLOWED_TYPES.has(file.type)) return "Only PNG and JPEG images are supported.";
  if (file.size > CLIENT_LIMITS.maxUploadBytes) {
    return `This file is ${formatBytes(file.size)}; the usual limit is ${formatBytes(CLIENT_LIMITS.maxUploadBytes)}.`;
  }
  return null;
}

/** UTF-8 byte length, which is what the backend counts against capacity. */
export function utf8ByteLength(text: string): number {
  return new TextEncoder().encode(text).length;
}

export function checkMessage(message: string): string | null {
  if (message.length === 0) return "Enter a message to embed.";
  if (utf8ByteLength(message) > CLIENT_LIMITS.maxMessageBytes) {
    return `The message is longer than ${formatBytes(CLIENT_LIMITS.maxMessageBytes)}.`;
  }
  return null;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  likely_cover: "Likely cover",
  potential_steganography: "Potential steganography",
  inconclusive: "Inconclusive",
};

export const RISK_LABEL: Record<RiskLevel, string> = {
  low: "Low",
  elevated: "Elevated",
  high: "High",
  not_assessed: "Not assessed",
};

export function verdictLabel(v: string): string {
  return VERDICT_LABEL[v as Verdict] ?? "Unknown";
}

export function riskLabel(r: string): string {
  return RISK_LABEL[r as RiskLevel] ?? "Unknown";
}

export function formatScore(score: number): string {
  return score.toFixed(2);
}

export function shortHash(hash: string, head = 8, tail = 6): string {
  return hash.length <= head + tail + 1 ? hash : `${hash.slice(0, head)}…${hash.slice(-tail)}`;
}

export function percent(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
