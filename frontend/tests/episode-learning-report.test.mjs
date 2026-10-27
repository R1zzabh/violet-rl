import test from 'node:test';
import assert from 'node:assert/strict';
import { buildRoleReport, checkpointEvidence, nextPolicyEvidence, markdownReport } from '../lib/episode-learning-report.ts';

const hash='a'.repeat(64);
function event(type,payload,seq=1){return {event_id:`${type}-${seq}`,event_type:type,seq,payload,run_id:'run',episode_id:'episode',occurred_at:'2026-10-26T19:00:00Z',previous_hash:'',event_hash:''};}
function update(role,extra={}){
 return event('sqli_ppo_update',{role,updated:true,update_status:'applied',policy_version_before:4,policy_version:5,checkpoint_sha256:hash,
  metrics:{sample_count:2,optimizer_steps:4},transition_diagnostics:[
   {decision_id:`${role}-a`,normalized_advantage:1,old_component_probabilities:[[0.2,0.8]],new_component_probabilities:[[0.4,0.6]]},
   {decision_id:`${role}-b`,normalized_advantage:-1,old_component_probabilities:[[0.4,0.6]],new_component_probabilities:[[0.6,0.4]]},
  ],...extra},role==='red'?10:11);
}
const outcome=()=>event('sqli_episode_outcome',{red_return:3,blue_return:-2,checkpoint_sha256:hash,action_schema_id:'sqli.composed_action.v1'},12);

test('reports exact paired probability means and isolates feedback by role',()=>{
 const events=[event('sqli_transition',{role:'red',reward:3,reward_components:{verified:3}}),event('sqli_transition',{role:'blue',reward:-2,reward_components:{availability:-2}}),update('red'),update('blue'),outcome()];
 const red=buildRoleReport(events,'red'), blue=buildRoleReport(events,'blue');
 assert.equal(red.reward,3);assert.equal(blue.reward,-2);
 assert.deepEqual(red.rewardComponents,{verified:3});assert.deepEqual(blue.rewardComponents,{availability:-2});
 assert.equal(red.diagnosticCoverage,1);assert.equal(red.beforeVersion,4);assert.equal(red.afterVersion,5);
 const change=red.changes.find(c=>c.option==='handle');
 assert.ok(Math.abs(change.before-.3)<1e-12);assert.ok(Math.abs(change.after-.5)<1e-12);assert.ok(Math.abs(change.delta-.2)<1e-12);
});

test('incomplete or skipped episodes do not attribute learning to supplied diagnostics',()=>{
 const report=buildRoleReport([update('red',{updated:false,update_status:'skipped_incomplete_joint_episode',policy_version:4})],'red');
 assert.equal(report.applied,false);assert.equal(report.changes.length,0);assert.equal(report.diagnostics.length,0);
 assert.match(report.narrative,/no learning is attributed/);
 assert.equal(buildRoleReport([],'red').reward,null);
 assert.equal(buildRoleReport([],'red').afterVersion,null);
});

test('Blue control submissions and fixture resets are not sampled model decisions',()=>{
 const events=[
  event('sqli_blue_decision',{phase:'control_submitted',action:'RESET',decision_id:null}),
  event('sqli_blue_decision',{phase:'policy_decision',decision_id:'blue-a',policy_version:4,selected_action:{operation:'MONITOR'}}),
  event('sqli_blue_decision',{phase:'control_submitted',action:'MONITOR',decision_id:'blue-a'}),
 ];
 const report=buildRoleReport(events,'blue');
 assert.equal(report.decisions.length,1);assert.equal(report.beforeVersion,4);
 assert.deepEqual(report.actionCounts,[{label:'MONITOR',count:1}]);
});

test('a missing checkpoint outcome is pending and a conflicting digest is a mismatch',()=>{
 assert.deepEqual(checkpointEvidence([update('red'),update('blue')]),{verified:false,hash,mismatch:false});
 assert.equal(checkpointEvidence([update('red'),update('blue'),outcome()]).verified,true);
 assert.equal(checkpointEvidence([update('red'),update('blue',{checkpoint_sha256:'b'.repeat(64)}),outcome()]).mismatch,true);
});

test('activation requires later start evidence for both recorded new policy versions',()=>{
 const events=[update('red'),update('blue'),outcome()];
 assert.equal(nextPolicyEvidence(events,[]).matched,false);
 assert.equal(nextPolicyEvidence(events,[event('sqli_episode_started',{policy_versions:{red:5,blue:4}},20)]).matched,false);
 assert.equal(nextPolicyEvidence(events,[event('sqli_episode_started',{policy_versions:{red:5,blue:5}},20)]).matched,true);
 assert.equal(nextPolicyEvidence(events,[event('sqli_episode_started',{policy_versions:{red:5,blue:5}},3)]).matched,false);
});

test('reports redact nested secrets from decisions and activation evidence',()=>{
 const episode={run_id:'run',episode_id:'episode',episode_index:0,status:'completed'};
 const events=[event('sqli_red_decision',{observation:{secret_phrase:'DO-NOT-EXPORT'},action:{operation:'MONITOR'}}),update('red'),update('blue'),outcome()];
 const next=[event('sqli_episode_started',{policy_versions:{red:5,blue:5},token:'DO-NOT-EXPORT'},20)];
 const markdown=markdownReport(episode,events,next);
 assert.ok(!markdown.includes('DO-NOT-EXPORT'));assert.match(markdown,/redacted/);assert.match(markdown,/held-out evaluation/);
});
