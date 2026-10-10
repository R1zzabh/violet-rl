import Link from "next/link";
import {
  Network,
  Shield,
  Activity,
  BarChart3,
  GitBranch,
  FlaskConical,
  Zap,
} from "lucide-react";

const PHASES = [
  {
    phase: "1",
    red: "Scripted (fixed strategies)",
    blue: "PPO (learning)",
    goal: "Verify environment, training loop, and baselines",
  },
  {
    phase: "2",
    red: "Multiple scripted strategies",
    blue: "PPO trained across strategies",
    goal: "Generalization, cross-strategy evaluation",
  },
  {
    phase: "3",
    red: "PPO (learning)",
    blue: "PPO (learning)",
    goal: "Self-play, non-stationarity, cross-play matrix",
  },
];

const CARD_ROW = [
  {
    icon: Network,
    label: "Cyber Network Simulator",
    desc: "6–10 node enterprise topology, partial observability, configurable vulnerabilities and connectivity.",
    tag: "BLOCK B",
  },
  {
    icon: Shield,
    label: "Attacker & Defender Agents",
    desc: "Red exploits, escalates, moves laterally. Blue monitors, isolates, patches, deploys honeypots.",
    tag: "BLOCK C/D",
  },
  {
    icon: Activity,
    label: "RL Training Loop",
    desc: "PettingZoo ParallelEnv, Ray RLlib PPO, configurable rewards, checkpointed runs, independent evaluation.",
    tag: "BLOCK F",
  },
  {
    icon: BarChart3,
    label: "Evidence-Based Dashboard",
    desc: "No fabricated claims. Every behavior is backed by measured trajectories and cross-play evaluation.",
    tag: "BLOCK G",
  },
];

export default function HomePage() {
  return (
    <div className="space-y-14">
      {/* ---- HERO ---- */}
      <section className="card-brutal-lg bg-[#111] relative">
        <div className="flex flex-col sm:flex-row items-start justify-between gap-6 mb-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-sm mono uppercase tracking-[0.2em] text-violet-400 font-bold">
                Versatile Intelligent Opponent Learning
              </span>
              <span className="badge-brutal badge-violet mono">v0.1.0</span>
            </div>
            <h1 className="text-[clamp(2rem,5vw,3.6rem)] font-bold tracking-tight leading-[1.02] mb-3" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              Multi-Agent RL{" "}
              <span className="text-violet-400">Cyber</span> Attack{" "}
              <span className="text-violet-400">&amp;</span> Defense
            </h1>
            <p className="max-w-2xl text-[#b0b0b0] leading-relaxed text-[15px]">
              A simulated cyber conflict where autonomous attacker and defender agents
              learn sequential strategies through competitive multi-agent reinforcement
              learning — measured, reproducible, and never fabricating results.
            </p>
          </div>
          <div className="shrink-0 flex flex-row sm:flex-col gap-3 flex-wrap">
            <Link href="/simulation" className="btn-brutal btn-brutal-red">
              <Activity size={14} />
              Live Simulation
            </Link>
            <Link href="/training" className="btn-brutal btn-brutal-ghost mono">
              <GitBranch size={14} />
              Training Runs
            </Link>
          </div>
        </div>

        {/* Status strip */}
        <div className="flex flex-wrap gap-4 border-t border-[#222] pt-4 mt-2">
          <div className="flex items-center gap-2 mono text-xs">
            <span className="status-dot status-dot-violet" />
            <span className="text-muted-foreground uppercase tracking-wider">Phase</span>
            <span className="font-semibold text-white">1</span>
          </div>
          <div className="flex items-center gap-2 mono text-xs">
            <span className="status-dot status-dot-blue" />
            <span className="text-muted-foreground uppercase tracking-wider">Blue</span>
            <span className="font-semibold text-white">RL (PPO)</span>
          </div>
          <div className="flex items-center gap-2 mono text-xs">
            <span className="status-dot status-dot-red" />
            <span className="text-muted-foreground uppercase tracking-wider">Red</span>
            <span className="font-semibold text-white">Scripted</span>
          </div>
          <div className="flex items-center gap-2 mono text-xs">
            <span className="status-dot status-dot-yellow" />
            <span className="text-muted-foreground uppercase tracking-wider">Environment</span>
            <span className="font-semibold text-white">PettingZoo</span>
          </div>
        </div>
      </section>

      {/* ---- WHAT IT IS / ISN'T ---- */}
      <section>
        <div className="flex items-center gap-3 mb-4">
          <FlaskConical size={18} className="text-violet-400" />
          <h2 className="text-lg font-bold tracking-tight" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            What VIOLET is — and isn&apos;t
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="card-brutal border-l-4 border-violet-400">
            <h3 className="text-xs mono uppercase tracking-[0.18em] text-violet-400 font-bold mb-3">
              It IS
            </h3>
            <ul className="space-y-2 text-sm leading-relaxed">
              <li className="flex gap-3">
                <span className="text-violet-400 mt-0.5 shrink-0">·</span>
                <span>A Markov game simulator with scripted and learnable Red/Blue agents</span>
              </li>
              <li className="flex gap-3">
                <span className="text-violet-400 mt-0.5 shrink-0">·</span>
                <span>A research platform for measuring whether RL produces better cyber strategies than baselines</span>
              </li>
              <li className="flex gap-3">
                <span className="text-violet-400 mt-0.5 shrink-0">·</span>
                <span>An evidence-based system: every &quot;learned behavior&quot; claim is backed by measured trajectories</span>
              </li>
            </ul>
          </div>
          <div className="card-brutal border-l-4 border-red-500">
            <h3 className="text-xs mono uppercase tracking-[0.18em] text-red-400 font-bold mb-3">
              It is NOT
            </h3>
            <ul className="space-y-2 text-sm leading-relaxed">
              <li className="flex gap-3">
                <span className="text-red-400 mt-0.5 shrink-0">·</span>
                <span>A real hacker AI</span>
              </li>
              <li className="flex gap-3">
                <span className="text-red-400 mt-0.5 shrink-0">·</span>
                <span>A system that attacks external networks</span>
              </li>
              <li className="flex gap-3">
                <span className="text-red-400 mt-0.5 shrink-0">·</span>
                <span>A tool for real exploitation</span>
              </li>
            </ul>
            <p className="mt-3 text-xs mono text-muted-foreground border-t border-[#222] pt-3">
              All activity happens inside the simulated environment. No real malware.
              No real targets.
            </p>
          </div>
        </div>
      </section>

      {/* ---- THREE PHASES ---- */}
      <section>
        <div className="flex items-center gap-3 mb-5">
          <GitBranch size={18} className="text-yellow-400" />
          <h2 className="text-lg font-bold tracking-tight" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            Three-phase roadmap
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {PHASES.map((p) => (
            <div key={p.phase} className="card-brutal p-5 border-l-[6px]" style={{ borderLeftColor: p.phase === "1" ? "#ff2a2a" : p.phase === "2" ? "#f5c518" : "#2a7aff" }}>
              <div className="flex items-center justify-between mb-4">
                <span className="text-4xl font-black mono text-white" style={{ fontFamily: "'Space Mono', monospace" }}>
                  P{p.phase}
                </span>
                <span className="badge-brutal mono">
                  {p.phase === "1" ? "Scripted Red" : p.phase === "2" ? "Multi-strategy Red" : "RL Red vs RL Blue"}
                </span>
              </div>
              <div className="space-y-3 text-sm">
                <div>
                  <div className="section-label mb-1">🔴 Red</div>
                  <div className="text-white font-semibold text-base">{p.red}</div>
                </div>
                <div>
                  <div className="section-label mb-1">🔵 Blue</div>
                  <div className="text-white font-semibold text-base">{p.blue}</div>
                </div>
                <div className="pt-2 border-t border-[#222]">
                  <div className="section-label mb-1">Goal</div>
                  <div className="text-[#b0b0b0]">{p.goal}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ---- STACK CARDS ---- */}
      <section>
        <div className="flex items-center gap-3 mb-5">
          <Zap size={18} className="text-violet-400" />
          <h2 className="text-lg font-bold tracking-tight" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            Stack
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {CARD_ROW.map((c) => (
            <div key={c.label} className="card-brutal">
              <div className="flex items-start justify-between mb-3">
                <c.icon size={18} className="text-violet-400" />
                <span className="badge-brutal badge-violet mono">{c.tag}</span>
              </div>
              <h3 className="font-bold text-white mb-2 text-base">{c.label}</h3>
              <p className="text-sm text-[#b0b0b0] leading-relaxed">{c.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- EVIDENCE STATEMENT ---- */}
      <section className="card-brutal-lg bg-[#111] border-yellow-500 border-2">
        <div className="flex items-start gap-4">
          <div className="shrink-0">
            <span className="badge-brutal badge-yellow mono">Evidence First</span>
          </div>
          <div style={{ minWidth: 0 }}>
            <h2 className="font-bold text-white text-lg mb-2" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              No fabricated learning curves.
            </h2>
            <p className="text-sm text-[#b0b0b0] leading-relaxed max-w-3xl">
              The dashboard will not say &quot;Blue learned to understand lateral movement&quot;
              unless we have evidence. It will say things like:
            </p>
            <pre className="mt-4 bg-[#000] border border-[#222] p-4 mono text-xs leading-relaxed overflow-x-auto" style={{ fontFamily: "'Space Mono', monospace" }}>
{`Against lateral-movement attacks:

  Episode 1–500   Containment: 21%
  Episode 4500–5000  Containment: 74%

  Observed behavior:
  Isolation selected after suspicious lateral-movement event in 83% of
  evaluated trajectories.

  Baseline: Rule-based policy = 41% · RL policy = 74%`}
            </pre>
            <p className="mt-4 text-sm text-[#b0b0b0]">
              Not: &quot;The neural network understands that APP is strategically important&quot;.
              We observe: &quot;The trained policy selected APP in 71/100 evaluation episodes
              under this condition.&quot;
            </p>
          </div>
        </div>
      </section>

      {/* ---- QUICK START ---- */}
      <section className="border-t border-[#222] pt-8">
        <h2 className="text-lg font-bold tracking-tight mb-4" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          Quick start
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="card-brutal p-5">
            <div className="section-label mb-2">Full stack (backend + frontend + postgres)</div>
            <pre className="bg-[#000] border border-[#222] p-4 mono text-xs leading-relaxed overflow-x-auto" style={{ fontFamily: "'Space Mono', monospace" }}>
{`$ git clone &lt;repo&gt; && cd violet
$ docker compose up`}
            </pre>
          </div>
          <div className="card-brutal p-5">
            <div className="section-label mb-2">Local development</div>
            <pre className="bg-[#000] border border-[#222] p-4 mono text-xs leading-relaxed overflow-x-auto" style={{ fontFamily: "'Space Mono', monospace" }}>
{`$ pip install -e &quot;.[dev]&quot;
$ cd frontend && npm install
$ npm run dev`}
            </pre>
          </div>
        </div>
        <p className="mt-4 text-xs mono text-muted-foreground" style={{ fontFamily: "'Space Mono', monospace" }}>
          Open the dashboard after starting. The `/dashboard` route shows the latest stored experiment.
        </p>
      </section>
    </div>
  );
}
