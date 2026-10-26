import Link from "next/link";
import {
  Activity,
  ArrowUpRight,
  BookOpenCheck,
  Bug,
  ChartNoAxesCombined,
  Shield,
  Swords,
} from "lucide-react";

const researchPages = [
  ["Graph simulation", "/simulation", "Network transitions are simulated in memory."],
  ["Graph training", "/training", "PPO results describe the PettingZoo environment."],
  ["Agent registry", "/agents", "Inspect the simulator’s Red and Blue policies."],
  ["Graph evaluation", "/evaluation", "Review simulated benchmark outcomes."],
] as const;

export default function HomePage() {
  return (
    <div className="space-y-9">
      <section className="card-brutal-lg relative overflow-hidden">
        <div aria-hidden="true" className="pointer-events-none absolute -right-10 -top-28 h-80 w-80 rounded-full border border-[#d4af37]/10">
          <div className="absolute inset-10 rounded-full border border-[#d4af37]/10" />
          <div className="absolute inset-20 rounded-full border border-[#f07832]/15" />
        </div>
        <div className="relative max-w-3xl">
          <div className="eyebrow">Versatile Intelligent Opponent Learning</div>
          <h1 className="page-heading mt-4 max-w-3xl">Two learning agents.<br className="hidden sm:block" /> One real, bounded target.</h1>
          <p className="page-lede">
            Red composes requests and Blue changes target controls through separate PPO policies.
            Each decision, HTTP response, verification, reward, and episode update is saved for
            replay. Graph research remains a separate simulated environment.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/sqli" className="btn-brutal btn-brutal-yellow"><Bug size={15} />Open SQLi learning lab<ArrowUpRight size={14} /></Link>
            <Link href="/simulation" className="btn-brutal btn-brutal-ghost"><Activity size={15} />Explore graph simulation</Link>
          </div>
        </div>
      </section>

      <section aria-labelledby="environment-heading">
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <span className="eyebrow">Environment boundary</span>
            <h2 id="environment-heading" className="mt-2 text-xl font-semibold tracking-tight">Two environments, separate evidence</h2>
          </div>
          <span className="badge-brutal">Research mode</span>
        </div>
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          <Link href="/sqli" className="group panel p-6 transition hover:border-[#d4af37]/40">
            <div className="flex items-center justify-between">
              <span className="badge-brutal badge-yellow">Live server evidence</span>
              <ArrowUpRight size={17} className="text-[#c1aa91] transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </div>
            <div className="mt-5 flex items-center gap-3"><Bug className="text-[#d4af37]" size={21} /><h3 className="text-lg font-semibold">SQLi learning lab</h3></div>
            <p className="mt-2 max-w-lg text-sm leading-6 text-[#c1aa91]">Separate Red and Blue PPO agents interact with an isolated HTTP service and a read-only synthetic-record database.</p>
            <div className="mt-5 flex flex-wrap gap-2"><span className="chip-mono">composed actions</span><span className="chip-mono">verified controls</span><span className="chip-mono">episode replay</span></div>
          </Link>
          <Link href="/range" className="group panel p-6 transition hover:border-[#d4af37]/40">
            <div className="flex items-center justify-between">
              <span className="badge-brutal badge-green">Historical HTTP runs</span>
              <ArrowUpRight size={17} className="text-[#c1aa91] transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </div>
            <div className="mt-5 flex items-center gap-3"><Shield className="text-[#d4af37]" size={21} /><h3 className="text-lg font-semibold">Violet Ledger archive</h3></div>
            <p className="mt-2 max-w-lg text-sm leading-6 text-[#c1aa91]">Replay the prior Alice and Bob document-access lab, including its persisted decisions, target evidence, and control checks.</p>
            <div className="mt-5 flex flex-wrap gap-2"><span className="chip-mono">IDOR scenario</span><span className="chip-mono">legacy runs</span><span className="chip-mono">separate store</span></div>
          </Link>
          <Link href="/simulation" className="group panel p-6 transition hover:border-[#d4af37]/40">
            <div className="flex items-center justify-between">
              <span className="badge-brutal">Simulation research</span>
              <ArrowUpRight size={17} className="text-[#c1aa91] transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </div>
            <div className="mt-5 flex items-center gap-3"><Swords className="text-[#f07832]" size={21} /><h3 className="text-lg font-semibold">Graph environment</h3></div>
            <p className="mt-2 max-w-lg text-sm leading-6 text-[#c1aa91]">A PettingZoo parallel environment for studying partially observable Red and Blue decisions over an in-memory network model.</p>
            <div className="mt-5 flex flex-wrap gap-2"><span className="chip-mono">simulated transitions</span><span className="chip-mono">graph PPO</span></div>
          </Link>
        </div>
      </section>

      <section aria-labelledby="research-nav-heading">
        <div className="mb-4 flex items-center gap-3"><BookOpenCheck size={17} className="text-[#d4af37]" /><h2 id="research-nav-heading" className="text-lg font-semibold">Simulation research</h2></div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {researchPages.map(([title, href, description]) => (
            <Link href={href} key={href} className="panel p-4 transition hover:border-[#d4af37]/35">
              <span className="section-label">Graph only</span>
              <h3 className="mt-2 font-semibold">{title}</h3>
              <p className="mt-2 text-xs leading-5 text-[#a99079]">{description}</p>
            </Link>
          ))}
        </div>
      </section>

      <section className="rounded-xl border border-[#d4af37]/15 bg-[#130d09]/80 p-5">
        <div className="section-label">Evidence standard</div>
        <p className="mt-2 max-w-4xl text-sm leading-6 text-[#c1aa91]">A policy update, reward increase, or changed action probability does not establish improved security. Real-range improvement reports must compare frozen policies against baselines on paired held-out seeds and include uncertainty intervals.</p>
        <Link href="/range" className="mt-4 inline-flex items-center gap-2 text-xs font-semibold text-[#f1d995] hover:text-white">Open range records<ChartNoAxesCombined size={14} /></Link>
      </section>
    </div>
  );
}
