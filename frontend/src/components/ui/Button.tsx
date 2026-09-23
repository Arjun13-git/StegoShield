import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost";

const STYLES: Record<Variant, string> = {
  primary: "bg-accent text-canvas hover:bg-accent-strong disabled:bg-line-strong disabled:text-muted",
  secondary: "border border-line-strong bg-surface-2 text-fg hover:border-accent disabled:text-muted disabled:hover:border-line-strong",
  ghost: "text-muted hover:text-fg disabled:opacity-50",
};

export function Button({ variant = "secondary", className, type = "button", ...rest }: React.ComponentProps<"button"> & { variant?: Variant }) {
  return (
    <button
      type={type}
      className={cn(
        "inline-flex min-h-10 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed",
        STYLES[variant],
        className,
      )}
      {...rest}
    />
  );
}
