import { cn } from "@/lib/utils";

export type Tone = "neutral" | "accent" | "warn" | "ok";

const TONES: Record<Tone, string> = {
  neutral: "border-line-strong text-muted",
  accent: "border-accent/60 bg-accent/10 text-accent",
  warn: "border-warn/60 bg-warn/10 text-warn",
  ok: "border-ok/50 bg-ok/10 text-ok",
};

export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: React.ReactNode; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium", TONES[tone], className)}>
      {children}
    </span>
  );
}
