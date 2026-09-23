import { cn } from "@/lib/utils";

/** Horizontally scrollable, captioned table so wide research tables stay usable on phones. */
export function DataTable({ caption, head, children, className }: { caption: string; head: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full min-w-[32rem] border-collapse text-left text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead className="text-xs text-muted">
          <tr className="border-b border-line-strong">{head}</tr>
        </thead>
        <tbody className="[&>tr]:border-b [&>tr]:border-line">{children}</tbody>
      </table>
    </div>
  );
}

export function Th({ children, className, scope = "col" }: { children: React.ReactNode; className?: string; scope?: "col" | "row" }) {
  return (
    <th scope={scope} className={cn("px-2 py-2 font-normal", className)}>
      {children}
    </th>
  );
}

export function Td({ children, className, mono = true }: { children: React.ReactNode; className?: string; mono?: boolean }) {
  return <td className={cn("px-2 py-2 text-fg", mono && "font-mono tabular-nums", className)}>{children}</td>;
}
