import type { Metadata } from "next";
import { DM_Sans, Space_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const dmSans = DM_Sans({
  subsets: [],
  weight: ["400", "500", "600", "700"],
  variable: "--font-dm-sans",
  display: "swap",
});

const spaceMono = Space_Mono({
  subsets: [],
  weight: ["400", "700"],
  variable: "--font-space-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VIOLET — Multi-Agent RL Cyber Attack & Defense Simulator",
  description:
    "A simulated cyber network modeled as a Partially Observable Markov Game, where autonomous attacker and defender agents learn sequential strategies through competitive multi-agent reinforcement learning.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="antialiased">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="bg-[#0a0a0a] text-[#f5f5f5] min-h-screen flex flex-col">
        <nav className="brutal-container max-w-[1280px] mx-auto w-full px-6 py-3 flex flex-wrap items-center justify-between gap-y-2 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-2xl font-bold tracking-tight" style={{ fontFamily: dmSans.variable }}>
              VIOLET
            </span>
            <span className="badge-brutal badge-violet mono hidden sm:inline">MARL · Cyber</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm mono" style={{ fontFamily: spaceMono.variable }}>
            <span className="badge-brutal badge-green">Phase 1</span>
            <span className="text-muted-foreground hidden md:inline">|</span>
            <Link href="/" className="ml-2 hover:text-violet-400 transition-colors">Overview</Link>
            <Link href="/simulation" className="ml-2 hover:text-violet-400 transition-colors">Simulation</Link>
            <Link href="/training" className="ml-2 hover:text-violet-400 transition-colors">Training</Link>
            <Link href="/range-training" className="ml-2 hover:text-violet-400 transition-colors">Real Range</Link>
            <Link href="/behavior" className="ml-2 hover:text-violet-400 transition-colors">Behavior</Link>
            <Link href="/evaluation" className="ml-2 hover:text-violet-400 transition-colors">Evaluation</Link>
            <Link href="/experiments" className="ml-2 hover:text-violet-400 transition-colors">Experiments</Link>
            <Link href="/agents" className="ml-2 hover:text-violet-400 transition-colors">Agents</Link>
            <Link href="/network" className="ml-2 hover:text-violet-400 transition-colors">Network</Link>
          </div>
        </nav>
        <main className="max-w-[1280px] mx-auto px-6 py-6 w-full flex-1">{children}</main>
        <footer className="border-t-2 border-[#222] mt-10 pb-6 pt-4 text-center mono text-xs text-muted-foreground w-full shrink-0" style={{ fontFamily: spaceMono.variable }}>
          <span className="badge-brutal badge-violet mr-2">v0.1.0</span>
          VIOLET · Multi-Agent RL Cyber Attack &amp; Defense Simulator
          <br />
          <span className="text-muted-foreground opacity-60">All activity is simulated. No real exploitation. No real targets.</span>
        </footer>
      </body>
    </html>
  );
}
