"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ArrowLeft, ArrowRight, Download, ExternalLink, Radio, RefreshCw } from "lucide-react";
import { rangeApi, type RangeEvent, type SqliEpisodeIndexItem, type TrainingRun, type TrainingUpdateRecord } from "../../lib/range-api";
import { actionLabel, buildRoleReport, checkpointEvidence, fixed, isPolicyDecision, markdownReport, nextPolicyEvidence, number, percent, redact, type RoleReport } from "../../lib/episode-learning-report";

const RUN_KEY = "violet.learning.journal.run";
const EMPTY_EVENTS: RangeEvent[] = [];
const short = (id: string) => `${id.slice(0, 8)}…`;
const time = (value: string) => new Date(value).toLocaleString(undefined, {hour12:false});
const statusClass = (value: string) => value === "completed" ? "pill-green" : value === "failed" || value === "incomplete" ? "pill-amber" : "pill-blue";

function Evidence({ title, value }: { title: string; value: unknown }) {
  const [open, setOpen] = useState(false);
  return <details className="learning-evidence-detail" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>{title}</summary>{open && <pre>{JSON.stringify(redact(value), null, 2)}</pre>}
  </details>;
}

function Stat({ label, children, foot }: {label: string; children: ReactNode; foot?: string}) {
  return <div className="learning-stat"><span>{label}</span><strong>{children}</strong>{foot && <small>{foot}</small>}</div>;
}

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function RoleSection({ report, events }: {report: RoleReport; events: RangeEvent[]}) {
  const {role, metrics, settings, diagnostics} = report;
  const name = role === "red" ? "Red" : "Blue";
  const signalPositive = diagnostics.filter((item) => (number(item.normalized_advantage) ?? 0) > 0).length;
  const signalNegative = diagnostics.filter((item) => (number(item.normalized_advantage) ?? 0) < 0).length;
  const metricLabels: [string, string][] = [["policy_loss", "Actor loss"], ["value_loss", "Critic loss"], ["entropy", "Entropy"],
    ["post_update_approx_kl", "Post-update KL"], ["clip_fraction", "Clipped fraction"], ["gradient_norm", "Gradient norm before clipping"]];
  return <section className="learning-role-report" data-role={role} aria-label={`${name} learning report`}>
    <div className="learning-role-heading"><div><span className={`pill pill-${role}`}>{name} policy</span><h3>{role === "red" ? "Request selection" : "Server control selection"}</h3></div>
      <span className="learning-role-version">v{report.beforeVersion ?? "?"} <span aria-label="to">→</span> v{report.afterVersion ?? "?"}</span></div>
    <p className="learning-narrative">{report.narrative}</p>
    <div className="learning-role-metrics">
      <Stat label="Episode reward" foot="Recorded feedback total">{fixed(report.reward, 3)}</Stat>
      <Stat label="Transitions in update" foot={`${report.transitions.filter((event) => event.payload.learning_eligible === true).length} marked eligible`}>{number(metrics.sample_count) ?? "—"}</Stat>
      <Stat label="Optimizer steps" foot={report.applied ? "Recorded for this episode" : "No applied update recorded"}>{number(metrics.optimizer_steps) ?? "—"}</Stat>
      <Stat label="Update evidence" foot={report.diagnosticCoverage === null ? "Optimizer inputs unavailable" : `${diagnostics.length} saved diagnostic rows`}>{report.applied ? report.diagnosticCoverage === 1 ? "Complete" : "Partial" : report.update ? "Skipped" : "Pending"}</Stat>
    </div>
    <h4>What feedback shaped this update?</h4>
    <p className="learning-narrative">{role === "red"
      ? "The recorded reward reflects verified unauthorized reads, accepted syntax progress, and request rejections."
      : "The recorded reward reflects verified prevention and the cost of denying legitimate service."} PPO compares this feedback and future returns with the critic&apos;s estimate.</p>
    {Object.keys(report.rewardComponents).length ? <div className="learning-table-scroll"><table><caption>Recorded {name} reward components, summed across transitions</caption><thead><tr><th>Reward component</th><th>Total</th></tr></thead><tbody>
      {Object.entries(report.rewardComponents).map(([key, value]) => <tr key={key}><td>{key.replaceAll("_", " ")}</td><td>{fixed(value)}</td></tr>)}
    </tbody></table></div> : <p className="learning-empty">Reward components have not been recorded.</p>}
    <p className="learning-narrative">{diagnostics.length
      ? `${signalPositive} sampled actions had positive normalized advantage; ${signalNegative} had negative normalized advantage. Positive values give PPO a signal to favor those choices in those states, subject to clipping, entropy, and shared model parameters.`
      : "Advantage signals will appear when the optimizer records its exact transition inputs."}</p>
    <h4>How did action probabilities change?</h4>
    <p className="learning-narrative">Before and after probabilities use the same saved observations and recurrent states. Values below are means over this optimizer batch; each decision&apos;s exact comparison is available below.</p>
    {report.changes.length ? <><div className="learning-table-scroll learning-policy-changes"><table><caption>Largest measured probability changes · {name}</caption><thead><tr><th>Choice</th><th>Before</th><th>After</th><th>Change</th></tr></thead><tbody>
      {report.changes.slice(0, 6).map((change) => <tr key={`${change.component}/${change.option}`}><td><small>{change.component}</small> {change.option.replaceAll("_", " ")}</td><td>{percent(change.before)}</td><td>{percent(change.after)}</td><td>{change.delta >= 0 ? "+" : ""}{(change.delta * 100).toFixed(2)} pp</td></tr>)}
    </tbody></table></div><Evidence title="All probability changes and observation counts" value={report.changes} /></>
      : <p className="learning-empty">No recorded before/after distributions are available for an applied update.</p>}
    <h4>Optimizer report</h4>
    <div className="learning-table-scroll"><table><caption>{name} PPO diagnostics</caption><thead><tr><th>Measurement</th><th>Recorded value</th></tr></thead><tbody>
      {metricLabels.map(([key,label]) => <tr key={key}><td>{label}</td><td>{fixed(metrics[key] ?? (key === "post_update_approx_kl" ? metrics.approx_kl : undefined), 6)}</td></tr>)}
    </tbody></table></div>
    <p className="learning-narrative">Entropy measures action uncertainty; KL measures movement from the old policy. The gradient norm is recorded before clipping{number(settings.gradient_clip_max_norm) !== null ? ` at ${settings.gradient_clip_max_norm}` : ""}. These diagnostics describe an update, and do not establish better held-out performance.</p>
    <Evidence title="PPO settings and complete optimizer metrics" value={{settings, metrics, update_event_id:report.update?.event_id, policy_id:report.update?.payload.policy_id}} />
    <details className="learning-evidence-detail"><summary>Exact advantages, returns, and action probabilities ({diagnostics.length} inputs)</summary>
      {diagnostics.length ? <div className="learning-table-scroll learning-diagnostic-table"><table><caption>Persisted optimizer inputs · {name}</caption><thead><tr><th>Round</th><th>Reward</th><th>Critic</th><th>Advantage</th><th>Normalized</th><th>Return target</th><th>Old P(action)</th><th>New P(action)</th></tr></thead><tbody>
        {diagnostics.map((item,index) => <tr key={String(item.decision_id ?? index)}><td>{(number(item.step_id) ?? index) + 1}</td><td>{fixed(item.reward)}</td><td>{fixed(item.critic_value)}</td><td>{fixed(item.advantage)}</td><td>{fixed(item.normalized_advantage)}</td><td>{fixed(item.return_target)}</td><td>{fixed(item.old_joint_probability,8)}</td><td>{fixed(item.new_joint_probability,8)}</td></tr>)}
      </tbody></table></div> : <p className="learning-empty">No optimizer inputs recorded for this role.</p>}
    </details>
    <details className="learning-evidence-detail"><summary>Decisions, observations, and outcome feedback ({report.decisions.length})</summary>
      <div className="learning-decision-list">{report.decisions.map((decision,index) => {
        const payload=decision.payload;
        const transition=report.transitions.find((item) => item.payload.decision_id === payload.decision_id);
        const diagnostic=diagnostics.find((item) => item.decision_id === payload.decision_id);
        const execution=events.filter((item) => item.payload.decision_id === payload.decision_id && ["sqli_target_response","sqli_control_acknowledged"].includes(item.event_type));
        return <div className="learning-decision" key={decision.event_id}>
          <strong>Round {(number(payload.step_id) ?? index) + 1} · v{String(payload.policy_version ?? "?")}</strong>
          <p>{actionLabel(payload.selected_action ?? payload.action)}</p>
          <div className="learning-decision-grid"><span>Sample probability: {percent(payload.sampled_action_probability)}</span><span>Critic: {fixed(payload.critic_value ?? payload.value_estimate)}</span><span>Reward: {fixed(transition?.payload.reward)}</span><span>{transition?.payload.learning_eligible === true ? "Eligible transition" : "Eligibility not confirmed"}</span></div>
          <Evidence title={`Round ${(number(payload.step_id) ?? index)+1}: exact observation, mask, probabilities, execution, and transition`} value={{decision_event:decision, execution_events:execution, transition:transition ?? null, optimizer_input:diagnostic ?? null}} />
        </div>;
      })}</div>
    </details>
  </section>;
}

type EpisodePage = {runId: string; cursor: string | null; items: SqliEpisodeIndexItem[]; nextCursor: string | null; hasMore: boolean; updates: TrainingUpdateRecord[]; run: TrainingRun};
type Trace = {episodeId: string; events: RangeEvent[]; fetchedAt: string};

export default function EpisodeJournal() {
  const [runs,setRuns]=useState<TrainingRun[]>([]);
  const [runId,setRunId]=useState("");
  const [cursor,setCursor]=useState<string|null>(null);
  const [history,setHistory]=useState<(string|null)[]>([]);
  const [page,setPage]=useState<EpisodePage|null>(null);
  const [selectedId,setSelectedId]=useState("");
  const [follow,setFollow]=useState(true);
  const [trace,setTrace]=useState<Trace|null>(null);
  const [nextTrace,setNextTrace]=useState<{episodeId:string; nextEpisodeId:string|null; nextIndex:number|null; events:RangeEvent[]} | null>(null);
  const [error,setError]=useState("");
  const [traceError,setTraceError]=useState("");
  const [nextError,setNextError]=useState("");
  const [revision,setRevision]=useState(0);
  const [initialLoading,setInitialLoading]=useState(true);
  const visiblePage=page?.runId===runId&&page.cursor===cursor?page:null;
  const selected=follow
    ? visiblePage?.items.find((episode)=>episode.status==="completed"&&episode.outcome)??visiblePage?.items[0]
    : visiblePage?.items.find((episode)=>episode.episode_id===selectedId);
  const events=useMemo(()=>trace && trace.episodeId===selected?.episode_id?trace.events:EMPTY_EVENTS,[trace,selected?.episode_id]);
  const following=nextTrace?.episodeId===selected?.episode_id?nextTrace:null;
  const red=useMemo(()=>buildRoleReport(events,"red"),[events]);
  const blue=useMemo(()=>buildRoleReport(events,"blue"),[events]);
  const saved=useMemo(()=>checkpointEvidence(events),[events]);
  const activation=useMemo(()=>nextPolicyEvidence(events,following?.events??[]),[events,following]);
  const outcome=events.findLast((event)=>event.event_type==="sqli_episode_outcome");
  const updates=visiblePage?.updates??[];
  const latest=(role:string)=>updates.filter((update)=>update.role===role).at(-1);

  useEffect(()=>{
    let cancelled=false;
    let timer:ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const result=await rangeApi.trainingRuns(100);
        if(cancelled)return;
        const training=result.items.filter((run)=>run.role==="joint"&&run.mode==="train"&&run.config.scenario_id==="sqli_lab_v1");
        setRuns(training);
        setRunId((current)=>{
          if(current&&training.some((run)=>run.training_run_id===current))return current;
          const query=new URLSearchParams(window.location.search).get("run");
          const savedId=window.localStorage.getItem(RUN_KEY);
          const preferred=training.find((run)=>run.training_run_id===(query??savedId));
          return (preferred??training.find((run)=>["running","queued"].includes(run.status))??training[0])?.training_run_id??"";
        });
        setError("");
      } catch(reason) {if(!cancelled)setError(reason instanceof Error?reason.message:"Training runs could not be loaded.");}
      finally {if(!cancelled){setInitialLoading(false);timer=setTimeout(load,15000);}}
    }
    void load();return()=>{cancelled=true;clearTimeout(timer);};
  },[revision]);

  useEffect(()=>{
    if(!runId)return;
    let cancelled=false;
    let timer:ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const [index,run,recorded]=await Promise.all([
          rangeApi.sqliEpisodes({runId,phase:"training",cursor,limit:20}),rangeApi.trainingRun(runId),rangeApi.trainingUpdates(runId),
        ]);
        if(cancelled)return;
        setPage({runId,cursor,items:index.items,nextCursor:index.next_cursor,hasMore:index.has_more,updates:recorded.items,run});
        setError("");
      } catch(reason){if(!cancelled)setError(reason instanceof Error?reason.message:"Episode journal could not be refreshed.");}
      finally{if(!cancelled)timer=setTimeout(load,7000);}
    }
    void load();return()=>{cancelled=true;clearTimeout(timer);};
  },[runId,cursor,revision]);

  useEffect(()=>{
    const episodeId=selected?.episode_id;
    if(!episodeId||!runId)return;
    let cancelled=false;
    let timer:ReturnType<typeof setTimeout>;
    let savedEvents:RangeEvent[]=[];
    let after=0;
    async function load() {
      try {
        for(let pages=0;pages<40;pages++){
          const result=await rangeApi.episodeTimeline(runId,episodeId!,after,500);
          if(cancelled)return;
          const byId=new Map(savedEvents.map((event)=>[event.event_id,event]));
          for(const event of result.items)byId.set(event.event_id,event);
          savedEvents=[...byId.values()].sort((a,b)=>a.seq-b.seq);
          if(result.items.length)after=Math.max(after,result.next_after_seq);
          if(!result.has_more)break;
          if(pages===39)throw new Error("The episode trace is incomplete: more saved events remain.");
        }
        if(cancelled)return;
        setTrace({episodeId:episodeId!,events:savedEvents,fetchedAt:new Date().toISOString()});
        setTraceError("");
      } catch(reason){if(!cancelled)setTraceError(reason instanceof Error?reason.message:"Episode evidence could not be loaded.");}
      finally{if(!cancelled&&!savedEvents.some((event)=>event.event_type==="sqli_episode_outcome"))timer=setTimeout(load,3000);}
    }
    void load();return()=>{cancelled=true;clearTimeout(timer);};
  },[runId,selected?.episode_id,revision]);

  useEffect(()=>{
    const episodeId=selected?.episode_id;
    const episodeIndex=selected?.episode_index;
    if(!episodeId||episodeIndex===undefined||!runId)return;
    let cancelled=false;
    let timer:ReturnType<typeof setTimeout>;
    async function load(){
      try {
        const result=await rangeApi.episodes(runId,{afterIndex:episodeIndex},1);
        const next=result.items[0];
        const timeline=next?await rangeApi.episodeTimeline(runId,next.episode_id,0,50):null;
        if(cancelled)return;
        setNextTrace({episodeId:episodeId!,nextEpisodeId:next?.episode_id??null,nextIndex:next?.episode_index??null,events:timeline?.items??[]});
        setNextError("");
        if(!timeline?.items.some((event)=>event.event_type==="sqli_episode_started"))timer=setTimeout(load,7000);
      }catch(reason){if(!cancelled){setNextError(reason instanceof Error?reason.message:"Next-episode evidence could not be loaded.");timer=setTimeout(load,7000);}}
    }
    void load();return()=>{cancelled=true;clearTimeout(timer);};
  },[runId,selected?.episode_id,selected?.episode_index,revision]);

  function chooseRun(id:string){setRunId(id);setCursor(null);setHistory([]);setFollow(true);setSelectedId("");window.localStorage.setItem(RUN_KEY,id);}
  const pipeline:[string,boolean][]=[
    ["Experience collected",red.decisions.length>0&&blue.decisions.length>0],
    ["Advantages recorded",red.diagnostics.length>0&&blue.diagnostics.length>0],
    ["Red PPO update",red.applied],["Blue PPO update",blue.applied],
    ["Checkpoint saved",saved.verified],["Next policies observed",activation.matched],
  ];

  return <section className="learning-journal" aria-label="Live episode learning journal">
    <header className="learning-journal-header"><div><p className="eyebrow">Live research notebook / saved episode evidence</p><h1 className="page-title">Learning, episode by episode.</h1><p className="page-deck">Read what feedback each agent received, what changed in its policy, and the evidence behind every update. Reports refresh as training records new episodes.</p></div>
      <div className="learning-journal-actions"><span><i className="learning-status-dot" data-status={error?"error":visiblePage?.run.status==="running"?"running":"waiting"}/>{visiblePage?.run.status==="running"?"Training live":"Recorded training"}</span><button className="button button-secondary button-small" onClick={()=>setRevision((value)=>value+1)}><RefreshCw size={13}/>Refresh journal</button></div></header>
    {error&&<div className="error-banner" role="alert">{error} Last successfully loaded evidence remains available.</div>}
    <div className="learning-run-selector"><label>Training run<select value={runId} onChange={(event)=>chooseRun(event.target.value)} disabled={initialLoading}>
      {!runs.length&&<option value="">{initialLoading?"Loading training runs…":"No training runs recorded"}</option>}
      {runs.map((run)=><option key={run.training_run_id} value={run.training_run_id}>{run.status} · {short(run.training_run_id)} · {String(run.config.episodes)} requested updates</option>)}
    </select></label><span className="pill">{visiblePage?.run.config.checkpoint_run_id?`Continued from ${short(String(visiblePage.run.config.checkpoint_run_id))}`:"Fresh policy initialization"}</span></div>
    <div className="learning-overview">
      <Stat label="Paired updates in this run" foot={`${String(visiblePage?.run.config.episodes??"—")} requested · failed attempts excluded`}>{Math.min(updates.filter((item)=>item.role==="red").length,updates.filter((item)=>item.role==="blue").length)}</Stat>
      <Stat label="Latest recorded Red policy" foot={`${latest("red")?.global_step??0} optimizer transitions`}>v{number(latest("red")?.metrics.policy_version_after)??"—"}</Stat>
      <Stat label="Latest recorded Blue policy" foot={`${latest("blue")?.global_step??0} optimizer transitions`}>v{number(latest("blue")?.metrics.policy_version_after)??"—"}</Stat>
      <Stat label="Report evidence" foot="Performance requires held-out evaluation">{selected?`Episode ${selected.episode_index+1}`:"Waiting"}</Stat>
    </div>
    {!visiblePage?.items.length?<div className="learning-empty">{initialLoading||runId&&!visiblePage?"Loading saved episodes…":"No episode evidence has been recorded for this run yet."}</div>:<div className="learning-journal-layout">
      <aside className="learning-episode-sidebar"><div className="learning-episode-topline"><h2>Episode journal</h2><button className={`button button-small ${follow?"":"button-secondary"}`} onClick={()=>{setFollow(true);setCursor(null);setHistory([]);}} aria-pressed={follow}><Radio size={12}/>Follow latest</button></div>
        <p className="learning-episode-meta">{follow?"Following the latest completed report":"Reading a saved episode"}</p>
        <div className="learning-episode-list">{visiblePage.items.map((episode)=><button className="learning-episode-item" key={episode.episode_id} aria-current={episode.episode_id===selected?.episode_id?"true":undefined} data-status={episode.status} onClick={()=>{setFollow(false);setSelectedId(episode.episode_id);}}>
          <span className="learning-episode-topline"><strong>Episode {episode.episode_index+1}</strong><span className={`pill ${statusClass(episode.status)}`}>{episode.status}</span></span>
          <span className="learning-episode-meta">{time(episode.created_at)} · {episode.outcome?.steps??"—"} rounds</span>
          <span className="learning-episode-meta">Red {fixed(episode.outcome?.red_return,2)} / Blue {fixed(episode.outcome?.blue_return,2)}</span>
        </button>)}</div>
        <div className="learning-pagination"><button className="button button-secondary button-small" disabled={!history.length} onClick={()=>{setCursor(history.at(-1)??null);setHistory(history.slice(0,-1));setFollow(true);}}><ArrowLeft size={12}/>Newer</button><button className="button button-secondary button-small" disabled={!visiblePage.hasMore} onClick={()=>{setHistory([...history,cursor]);setCursor(visiblePage.nextCursor);setFollow(true);}}>Older<ArrowRight size={12}/></button></div>
      </aside>
      <article className="learning-report-body" aria-label="Selected episode report">
        {!selected?<p className="learning-empty">Select an episode to read its report.</p>:<>
          <header className="learning-report-header"><div><p className="eyebrow">Episode {selected.episode_index+1} / seed {selected.seed??"unavailable"}</p><h2>What changed after this episode?</h2><span className={`pill ${statusClass(selected.status)}`}>{selected.status}</span> <code>{short(selected.episode_id)}</code></div>
            <div className="learning-journal-actions"><button className="button button-secondary button-small" disabled={!events.length} onClick={()=>download(`violet-episode-${selected.episode_index+1}.md`,markdownReport(selected,events,following?.events??[]),"text/markdown;charset=utf-8")}><Download size={13}/>Download report</button><button className="button button-secondary button-small" disabled={!events.length} onClick={()=>download(`violet-episode-${selected.episode_index+1}-evidence.json`,JSON.stringify(redact({schema_id:"violet.episode_learning_report.v1",episode:selected,events,next_episode_events:following?.events??[]}),null,2),"application/json")} aria-label="Export episode evidence as JSON">Evidence JSON</button></div>
          </header>
          {traceError&&<div className="learning-inline-error" role="alert">{traceError}</div>}
          {!events.length?<div className="learning-empty">Loading this episode&apos;s saved evidence…</div>:<>
            <div className="learning-evidence-note">These are recorded observations, rewards, and optimizer measurements. Probability changes describe this training batch. They do not prove that either policy performs better on unseen fixtures.</div>
            <div className="learning-pipeline" aria-label="Recorded learning sequence">{pipeline.map(([label,recorded],index)=><div key={label} data-state={recorded?"recorded":"waiting"}><span>{String(index+1).padStart(2,"0")}</span><strong>{label}</strong><small>{recorded?"Recorded":"Not yet evidenced"}</small></div>)}</div>
            <div className="learning-outcome-grid"><Stat label="Verified unauthorized access">{outcome?.payload.verified_exposure===true?"Observed":outcome?.payload.verified_exposure===false?"Not observed":"Unavailable"}</Stat><Stat label="Legitimate availability">{percent(outcome?.payload.legitimate_query_availability)}</Stat><Stat label="Verified prevention">{percent(outcome?.payload.verified_prevention_rate)}</Stat></div>
            {!outcome&&<p className="learning-inline-error">The final outcome has not been saved. This is a partial report; no completed update is inferred.</p>}
            {outcome?.payload.joint_update_applied===false&&<p className="learning-inline-error">This episode was excluded from learning. Both policies require eligible transitions from a completed episode before updating.</p>}
            <div className="learning-role-grid"><RoleSection report={red} events={events}/><RoleSection report={blue} events={events}/></div>
            <section className="learning-next-use"><h3>Checkpoint and use in the next episode</h3><p>{saved.verified?"Both policy-update records and the episode outcome agree on the saved checkpoint hash.":saved.mismatch?"Checkpoint records disagree. This report cannot confirm a common saved policy pair.":"A correlated checkpoint for both applied updates has not been recorded."}</p><code>{saved.hash??"Checkpoint digest pending"}</code>
              <p>{activation.matched?`The next episode's start event confirms Red v${red.afterVersion} and Blue v${blue.afterVersion} were used.`:activation.observed?"The next episode is recorded, but its versions do not confirm both expected updated policies.":"No subsequent episode start has been observed yet. Activation is not inferred from a saved checkpoint."}</p>
              {nextError&&<p className="learning-inline-error">{nextError}</p>}
              {following?.nextEpisodeId&&<button className="button button-secondary button-small" onClick={()=>{setFollow(false);setSelectedId(following.nextEpisodeId!);if(!visiblePage.items.some((episode)=>episode.episode_id===following.nextEpisodeId)){setCursor(history.at(-1)??null);setHistory(history.slice(0,-1));setRevision((value)=>value+1);}}}>Read next episode {following.nextIndex!==null?following.nextIndex+1:""}<ArrowRight size={12}/></button>}
              {activation.observed&&<Evidence title="Next episode's actual start event and first recorded decisions" value={{policy_comparison:activation.comparisons,start_event:activation.start,decisions:following?.events.filter(isPolicyDecision).slice(0,4)}}/>}
            </section>
            <Evidence title="Observed action frequencies in this episode" value={{red:red.actionCounts,blue:blue.actionCounts}}/>
            <Evidence title="Episode outcome, evidence identifiers, and reward contract" value={{episode:selected,outcome_event:outcome??null,event_count:events.length,first_event_id:events[0]?.event_id,last_event_id:events.at(-1)?.event_id}}/>
            <footer className="learning-report-footer"><span>{events.length} recorded events · {trace?.episodeId===selected.episode_id?`loaded ${time(trace.fetchedAt)}`:"loading"}</span><a href="/episodes">Open event explorer<ExternalLink size={12}/></a></footer>
          </>}
        </>}
      </article>
    </div>}
  </section>;
}
