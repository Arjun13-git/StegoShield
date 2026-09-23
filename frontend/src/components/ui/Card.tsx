import { cn } from "@/lib/utils";

export function Card({ className, children, ...rest }: React.ComponentProps<"section">) {
  return (
    <section className={cn("rounded-lg border border-line bg-surface p-4 sm:p-5", className)} {...rest}>
      {children}
    </section>
  );
}

export function CardHeading({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-3">
      <div className="min-w-0">
        {eyebrow && <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">{eyebrow}</p>}
        <h2 className="text-base font-semibold text-fg sm:text-lg">{title}</h2>
      </div>
      {children}
    </div>
  );
}
