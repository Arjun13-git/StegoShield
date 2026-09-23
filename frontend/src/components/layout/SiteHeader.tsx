"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { StatusIndicator } from "./StatusIndicator";

const NAV = [
  { href: "/analyze", label: "Analyze" },
  { href: "/encode", label: "Encode" },
  { href: "/research", label: "Research" },
] as const;

function ShieldMark() {
  return (
    <svg viewBox="0 0 24 24" className="h-6 w-6 text-accent" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M12 3 4.5 6v5.5c0 4.5 3.1 8 7.5 9.5 4.4-1.5 7.5-5 7.5-9.5V6L12 3Z" strokeLinejoin="round" />
      <path d="M9 12h6M12 9v6" strokeLinecap="round" opacity="0.6" />
    </svg>
  );
}

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const links = NAV.map(({ href, label }) => {
    const active = pathname === href || pathname.startsWith(`${href}/`);
    return (
      <Link
        key={href}
        href={href}
        aria-current={active ? "page" : undefined}
        onClick={() => setOpen(false)}
        className={cn(
          "rounded-md px-3 py-2 text-sm font-medium transition-colors",
          active ? "bg-surface-2 text-fg" : "text-muted hover:text-fg",
        )}
      >
        {label}
      </Link>
    );
  });

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-canvas/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <Link href="/analyze" className="flex items-center gap-2.5" aria-label="StegoShield home">
          <ShieldMark />
          <span className="leading-tight">
            <span className="block text-base font-semibold tracking-tight">StegoShield</span>
            <span className="hidden text-[11px] text-muted sm:block">Explainable Image Steganalysis</span>
          </span>
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-1 md:flex">
          {links}
        </nav>

        <div className="flex items-center gap-3">
          <StatusIndicator className="hidden lg:inline-flex" />
          <button
            type="button"
            className="rounded-md border border-line-strong px-3 py-2 text-sm md:hidden"
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? "Close" : "Menu"}
          </button>
        </div>
      </div>

      {open && (
        <div id="mobile-nav" className="border-t border-line px-4 py-3 md:hidden">
          <nav aria-label="Primary mobile" className="flex flex-col gap-1">
            {links}
          </nav>
          <StatusIndicator className="mt-3" />
        </div>
      )}
    </header>
  );
}
