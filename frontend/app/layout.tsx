import type { Metadata } from "next";
import { DM_Sans, Space_Mono } from "next/font/google";
import Link from "next/link";
import {
  Activity,
  Bug,
  BookOpenCheck,
  ChartNoAxesCombined,
  LayoutDashboard,
  Network,
  Shield,
  Swords,
  UsersRound,
} from "lucide-react";
import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-dm-sans",
  display: "swap",
});

const spaceMono = Space_Mono({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-space-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VIOLET · Violet Ledger Cyber Range",
  description:
    "Evidence-backed Red and Blue policy research for the isolated Violet Ledger training target.",
};

const simulationLinks = [
  { href: "/simulation", label: "Graph simulation", icon: Activity },
  { href: "/training", label: "Graph training", icon: Swords },
  { href: "/evaluation", label: "Graph evaluation", icon: ChartNoAxesCombined },
  { href: "/network", label: "Topology", icon: Network },
  { href: "/agents", label: "Agent registry", icon: UsersRound },
  { href: "/behavior", label: "Behavior analysis", icon: BookOpenCheck },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${dmSans.variable} ${spaceMono.variable}`}>
      <body>
        <div className="app-frame">
          <aside className="app-sidebar" aria-label="Main navigation">
            <Link className="brand-lockup" href="/" aria-label="VIOLET command center">
              <span className="brand-mark" aria-hidden="true">V</span>
              <span>
                <span className="brand-name">VIOLET</span>
                <span className="brand-subtitle" style={{ display: "block" }}>Research console</span>
              </span>
            </Link>
            <span className="nav-kicker">Workspace</span>
            <nav className="app-nav">
              <Link href="/"><LayoutDashboard size={15} aria-hidden="true" />Command center</Link>
              <Link className="nav-primary" href="/sqli"><Bug size={15} aria-hidden="true" />SQLi learning lab</Link>
              <Link href="/range"><Shield size={15} aria-hidden="true" />Violet Ledger archive</Link>
            </nav>
            <span className="nav-kicker">Simulation research</span>
            <nav className="app-nav" aria-label="Simulation research">
              {simulationLinks.map(({ href, label, icon: Icon }) => (
                <Link href={href} key={href}><Icon size={15} aria-hidden="true" />{label}</Link>
              ))}
              <Link href="/experiments"><ChartNoAxesCombined size={15} aria-hidden="true" />Experiments</Link>
            </nav>
            <div className="sidebar-status" aria-label="Range target boundary">
              <div className="sidebar-status-title">Target boundary</div>
              <div className="sidebar-status-value"><span className="status-lamp" />Isolated HTTP lab</div>
              <div className="sidebar-status-note">Synthetic records · no public target<br />Executor and database networks separated</div>
            </div>
          </aside>
          <div className="app-workspace">
            <header className="app-topbar">
              <span className="topbar-context">Cyber range / live target research</span>
              <div className="topbar-right">
                <span>SQLI LAB · V1</span>
                <span className="environment-pill"><span className="status-lamp" />private range</span>
              </div>
            </header>
            <main className="app-content">{children}</main>
            <footer className="app-footer">
              VIOLET · Real HTTP evidence and graph simulation are stored and evaluated separately.
            </footer>
          </div>
        </div>
      </body>
    </html>
  );
}
