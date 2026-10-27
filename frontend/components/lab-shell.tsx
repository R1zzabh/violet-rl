"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck } from "lucide-react";

export default function LabShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="lab-shell">
      <header className="shell-header">
        <Link href="/" className="brand-lockup" aria-label="VIOLET training monitor">
          <span className="brand-mark" aria-hidden="true">V</span>
          <span className="brand-copy">
            <span className="brand-name">VIOLET</span>
            <span className="brand-subtitle">Red / Blue classroom lab</span>
          </span>
        </Link>
        <nav className="primary-nav" aria-label="Main navigation">
          <Link href="/" aria-current={pathname === "/" ? "page" : undefined}>Training</Link>
          <Link href="/server" aria-current={pathname.startsWith("/server") ? "page" : undefined}>Server Live</Link>
          <Link href="/breaches" aria-current={pathname.startsWith("/breaches") ? "page" : undefined}>Breaches</Link>
          <Link href="/episodes" aria-current={pathname.startsWith("/episodes") ? "page" : undefined}>Episodes</Link>
          <Link href="/learning" aria-current={pathname.startsWith("/learning") ? "page" : undefined}>Learning</Link>
          <Link href="/experiments" aria-current={pathname.startsWith("/experiments") ? "page" : undefined}>Experiments</Link>
        </nav>
        <div className="shell-boundary"><ShieldCheck size={14} aria-hidden="true" /><span>localhost · isolated target</span></div>
      </header>
      <main className="shell-main">{children}</main>
      <footer className="shell-footer">
        <span>VIOLET / synthetic records / fixed HTTP routes</span>
        <span>Live decisions are recorded with correlated target evidence.</span>
      </footer>
    </div>
  );
}
