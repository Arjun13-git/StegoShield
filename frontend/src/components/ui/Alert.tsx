import { cn } from "@/lib/utils";

type Variant = "info" | "warning" | "error";

const STYLES: Record<Variant, { box: string; label: string }> = {
  info: { box: "border-accent/40 bg-accent/10", label: "Note" },
  warning: { box: "border-warn/40 bg-warn/10", label: "Warning" },
  error: { box: "border-danger/50 bg-danger/10", label: "Error" },
};

/** Text label + icon-free border colour: state is never conveyed by colour alone. */
export function Alert({
  variant = "info",
  title,
  children,
  className,
  role,
}: {
  variant?: Variant;
  title?: string;
  children: React.ReactNode;
  className?: string;
  role?: "alert" | "status";
}) {
  const s = STYLES[variant];
  return (
    <div role={role} className={cn("rounded-md border px-3 py-2.5 text-sm", s.box, className)}>
      <p className="font-medium text-fg">
        <span className="mr-1.5 font-mono text-[11px] uppercase tracking-wider text-muted">{s.label}</span>
        {title}
      </p>
      <div className="mt-1 text-muted">{children}</div>
    </div>
  );
}
