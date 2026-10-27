import { expect, test, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

const runId="journal-test-run";
const hash="a".repeat(64);
const run={training_run_id:runId,role:"joint",mode:"train",status:"running",config:{scenario_id:"sqli_lab_v1",episodes:100},result:null,stop_requested:false,created_at:"2026-10-26T19:00:00Z",updated_at:"2026-10-26T19:00:00Z"};
const episode=(index:number,status="completed")=>({episode_id:`episode-${index}`,run_id:runId,episode_index:index,status,seed:index,created_at:`2026-10-26T19:0${index}:00Z`,timestamp:`2026-10-26T19:0${index}:00Z`,phase:"training",run_mode:"train",run_type:"joint",run_status:"running",outcome_key:"blue",outcome:{status,winner:"blue",reason:"recorded outcome",verified_exposure:false,challenge_disclosed:false,steps:1,red_return:0.5,blue_return:1},policy_versions_at_start:{red:4+index,blue:4+index}});
const event=(type:string,payload:Record<string,unknown>,seq:number,id:string)=>({event_id:`${id}-${seq}`,event_type:type,payload,seq,run_id:runId,episode_id:id,occurred_at:"2026-10-26T19:00:00Z",previous_hash:"",event_hash:""});
function timeline(index:number,applied=true){
 const id=`episode-${index}`,before=4+index,after=applied?before+1:before,offset=index*100;
 const events=[event("sqli_episode_started",{policy_versions:{red:before,blue:before}},offset+1,id)];
 for(const [n,role] of ["red","blue"].entries()){
  const decisionId=`${id}-${role}`;
  const action=role==="red"?{schema_id:"sqli.composed_action.v1",field:"handle"}:{operation:"MONITOR"};
  events.push(event(`sqli_${role}_decision`,{role,step_id:0,decision_id:decisionId,action,observation:{features:[0,1],private_token:"SHOULD-NEVER-EXPORT"},policy_version:before,sampled_action_probability:.5,critic_value:.2},offset+2+n,id));
  events.push(event("sqli_transition",{role,step_id:0,decision_id:decisionId,action,reward:role==="red"?.5:1,reward_components:{verified_feedback:role==="red"?.5:1},learning_eligible:applied,accepted:true},offset+4+n,id));
  events.push(event("sqli_ppo_update",{role,updated:applied,update_status:applied?"applied":"skipped_incomplete_joint_episode",policy_version_before:before,policy_version:after,checkpoint_sha256:hash,metrics:{sample_count:applied?1:0,optimizer_steps:applied?4:0,policy_loss:-.01,value_loss:.2,entropy:.6,post_update_approx_kl:.01,clip_fraction:0,gradient_norm:.4},ppo_settings:{gradient_clip_max_norm:.5},transition_diagnostics:applied?[{step_id:0,decision_id:decisionId,critic_value:.2,reward:.5,advantage:.3,normalized_advantage:1,return_target:.5,old_joint_probability:.5,new_joint_probability:.6,old_component_probabilities:[[.5,.5]],new_component_probabilities:[[.6,.4]]}]:[]},offset+6+n,id));
 }
 events.push(event("sqli_episode_outcome",{joint_update_applied:applied,red_return:.5,blue_return:1,checkpoint_sha256:hash,action_schema_id:"sqli.composed_action.v1",verified_exposure:false,verified_prevention_rate:1,legitimate_query_availability:1},offset+8,id));
 return events.sort((a,b)=>a.seq-b.seq);
}

async function mockJournal(page:Page){
 let advanced=false;
 const methods:string[]=[];
 await page.addInitScript(()=>localStorage.clear());
 await page.route("**/range/**",async route=>{
  const request=route.request();methods.push(request.method());
  const url=new URL(request.url());
  let body:unknown;
  if(url.pathname==="/range/training-runs")body={items:[run],has_more:false,next_offset:1};
  else if(url.pathname===`/range/training-runs/${runId}`)body=run;
  else if(url.pathname.endsWith("/updates"))body={items:["red","blue"].map(role=>({role,update_index:advanced?2:1,metrics:{policy_version_after:advanced?7:5},global_step:advanced?2:1,checkpoint_sha256:hash}))};
  else if(url.pathname==="/range/sqli/episodes")body={items:advanced?[episode(2),episode(1,"incomplete"),episode(0)]:[episode(1,"incomplete"),episode(0)],next_cursor:null,has_more:false};
  else if(url.pathname.endsWith("/timeline")){
   const id=url.pathname.split("/").at(-2)!;
   const index=Number(id.split("-").at(-1));
   const items=timeline(index,index!==1).filter(item=>item.seq>Number(url.searchParams.get("after_seq")??0));
   body={items,next_after_seq:items.at(-1)?.seq??0,has_more:false};
  } else if(url.pathname.endsWith("/episodes")){
   const after=Number(url.searchParams.get("after_index"));
   body={items:after===0?[{episode_id:"episode-1",episode_index:1}]:[]};
  }else return route.fulfill({status:404,contentType:"application/json",body:JSON.stringify({detail:"not found"})});
  return route.fulfill({status:200,contentType:"application/json",body:JSON.stringify(body)});
 });
 return {methods,advance:()=>{advanced=true;}};
}

test("live report shows both policies, verifies next use, exports redacted evidence, and refreshes",async({page},testInfo)=>{
 const state=await mockJournal(page);
 const errors:string[]=[];page.on("pageerror",e=>errors.push(e.message));
 await page.goto("/learning");
 await expect(page.getByRole("heading",{name:"What changed after this episode?"})).toBeVisible();
 await expect(page.getByRole("region",{name:"Red learning report"})).toContainText("v4 → v5");
 await expect(page.getByRole("region",{name:"Blue learning report"})).toContainText("v4 → v5");
 await expect(page.getByText("The next episode's start event confirms Red v5 and Blue v5 were used.",{exact:true})).toBeVisible();
 const downloadEvent=page.waitForEvent("download");
 await page.getByRole("button",{name:"Download report",exact:true}).click();
 const download=await downloadEvent;
 const output=testInfo.outputPath("report.md");await download.saveAs(output);
 const report=await readFile(output,"utf8");
 expect(report).toContain("## RED policy");expect(report).toContain("## BLUE policy");expect(report).not.toContain("SHOULD-NEVER-EXPORT");
 await page.locator(".learning-episode-item").filter({hasText:"Episode 2"}).click();
 await expect(page.getByText("This episode was excluded from learning.",{exact:false})).toBeVisible();
 await expect(page.getByRole("region",{name:"Red learning report"})).toContainText("no learning is attributed");
 state.advance();
 await page.getByRole("button",{name:"Follow latest"}).click();
 await page.getByRole("button",{name:"Refresh journal"}).click();
 await expect(page.getByRole("region",{name:"Red learning report"})).toContainText("v6 → v7");
 expect(state.methods.every(method=>method==="GET")).toBe(true);
 expect(errors).toEqual([]);
});

test("mobile report and expanded tables stay inside viewport",async({page})=>{
 await page.setViewportSize({width:375,height:812});
 await mockJournal(page);await page.goto("/learning");
 const red=page.getByRole("region",{name:"Red learning report"});
 await expect(red).toContainText("v4 → v5");
 await red.getByText("Exact advantages, returns, and action probabilities (1 inputs)",{exact:true}).click();
 const dimensions=await page.evaluate(()=>({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));
 expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width+1);
});
