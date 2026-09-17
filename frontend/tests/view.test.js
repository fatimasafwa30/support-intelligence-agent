import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {analyze, canSubmit, renderResult, metric, demos, traceStages, renderPending} from '../view.js';

// Synthetic contract fixture. Never shipped as an inference result.
const fixture = () => ({version: 1, customer_message: 'Test query', predicted_intent: 'test_intent', intent_confidence: .7123,
  final_action: 'AUTO_HANDLE', target_queue: 'self_service', decision_reasons: ['test_policy_reason'],
  risk: {level: 'low', categories: []}, retrieval: {attempts: 1, top_similarity: .4567, sufficiency: 'sufficient'},
  evidence: [{evidence_id: 'test_evidence_1', similarity: .4567, customer_problem: 'Past problem', brand_resolution: 'Past response', urls: ['https://support.apple.com/test']}],
  reply: {text: 'Test reply', grounded: true, used_evidence_ids: ['test_evidence_1'], used_urls: [], provider: 'mock'},
  verification_passed: true, generation_attempts: 1, latency_ms: 123.4});

test('blank, oversized, busy and offline submissions are blocked', () => {
  for (const value of ['', '  ', '\n', 'x'.repeat(4001)]) assert.equal(canSubmit(value, false, true), false);
  assert.equal(canSubmit('query', true, true), false);
  assert.equal(canSubmit('query', false, false), false);
  assert.equal(canSubmit('query', false, true), true);
});
test('all three actions use distinct text, backed by policy reasons', () => {
  for (const [action, label] of [['AUTO_HANDLE', 'Auto-handle'], ['ASK_CLARIFICATION', 'Ask clarification'], ['ESCALATE', 'Escalate to a human']]) {
    const data = fixture(); data.final_action = action;
    const html = renderResult(data);
    assert.ok(html.includes(label)); assert.ok(html.includes('test policy reason'));
    assert.ok(html.includes('test_policy_reason'));
  }
});
test('decision summary leads into observable trace, evidence and response', () => {
  const html = renderResult(fixture());
  const order = ['id="decision-title"', 'aria-label="Decision trace"', 'id="evidence-title"', 'id="reply-title"'];
  const positions = order.map(id => html.indexOf(id));
  assert.ok(positions.every(position => position >= 0));
  assert.deepEqual(positions, [...positions].sort((a, b) => a - b));
});
test('trace distinguishes actual verification from routing, including clarification', () => {
  const data = fixture(); data.final_action = 'ASK_CLARIFICATION';
  const stages = traceStages(data);
  assert.equal(stages[3].value, 'Passed');
  assert.equal(stages[4].status, 'clarify');
  assert.equal(stages[4].value, 'Ask clarification');
  data.verification_passed = false;
  assert.equal(traceStages(data)[3].value, 'Not passed');
});
test('early escalation marks retrieval and verification as not run, not failed', () => {
  const data = fixture(); data.final_action = 'ESCALATE'; data.reply = null;
  data.retrieval.attempts = 0; data.evidence = [];
  const stages = traceStages(data);
  assert.equal(stages[2].status, 'skipped');
  assert.equal(stages[3].status, 'skipped');
  assert.equal(stages[3].value, 'Not run');
  assert.equal(stages[4].status, 'escalate');
});
test('pending trace reports no completed stages or fabricated percentages', () => {
  const pending = renderPending();
  assert.equal((pending.match(/Awaiting result/g) || []).length, 5);
  assert.ok(!pending.includes('%'));
  assert.ok(!pending.includes('Passed'));
});
test('risk attention reflects returned risk without inventing a failed verification', () => {
  const data = fixture(); data.final_action = 'ESCALATE';
  data.risk.level = 'critical';
  assert.equal(traceStages(data)[1].status, 'attention');
  assert.equal(traceStages(data)[3].value, 'Passed');
  data.risk.level = 'low';
  assert.equal(traceStages(data)[1].status, 'observed');
});
test('trace and evidence selection have non-color state labels', () => {
  const html = renderResult(fixture());
  assert.ok(html.includes('FINAL DECISION / AUTO_HANDLE'));
  assert.ok(html.includes('class="selection-label">Selected'));
  assert.ok(html.includes('aria-label="Decision stages"'));
  assert.ok(html.includes('class="stage-label">Observed'));
  assert.ok(!html.includes('scroll horizontally'));
});
test('policy block and clarification stop are explicit without fabricating stage progress', () => {
  const data = fixture();
  data.final_action = 'ESCALATE'; data.risk.level = 'critical';
  data.retrieval.attempts = 0; data.reply = null; data.generation_attempts = 0;
  data.decision_reasons = ['early_critical_hazard_short_circuit'];
  assert.ok(renderResult(data).includes('Policy block'));
  assert.ok(renderResult(data).includes('Automation stopped'));
  data.final_action = 'ASK_CLARIFICATION';
  assert.ok(renderResult(data).includes('Awaiting clarification'));
  assert.ok(!renderPending().includes('Policy block'));
  assert.ok(!renderPending().includes('scroll horizontally'));
});
test('evidence inspector has labelled selectable cases and preserves original text', () => {
  const html = renderResult(fixture());
  assert.ok(html.includes('aria-pressed="true"'));
  assert.ok(html.includes('aria-controls="source-0"'));
  assert.ok(html.includes('CUSTOMER'));
  assert.ok(html.includes('APPLESUPPORT'));
  assert.ok(html.includes('Source metadata'));
});
test('actual evidence, reply and measured scores render with citation membership', () => {
  const data = fixture();
  const html = renderResult(data);
  for (const text of ['Past problem', 'Past response', 'test_evidence_1', 'Test reply', '0.712', '0.457', 'Cited in reply']) assert.ok(html.includes(text));
  data.reply.used_evidence_ids = [];
  assert.ok(!renderResult(data).includes('Cited in reply'));
});
test('missing evidence and skipped generation are explicit; null scores are not fabricated', () => {
  const data = fixture(); data.evidence = []; data.reply = null;
  data.retrieval = {attempts: 0, top_similarity: null, sufficiency: 'unknown'};
  data.intent_confidence = null;
  const html = renderResult(data);
  for (const text of ['No historical evidence returned', 'Retrieval did not run', 'No reply constructed', 'Not available']) assert.ok(html.includes(text));
  assert.equal(metric(null), 'Not available'); assert.equal(metric(NaN), 'Not available');
  assert.equal(metric(0), '0.000');
});
test('labels, judge scores, trace and rationales are never interpolated', () => {
  const data = fixture();
  const privateFields = {gold_intent: 'SECRET_A', gold_risk: 'SECRET_B', gold_action: 'SECRET_C', annotation_notes: 'SECRET_D', judge_scores: 'SECRET_E', trace: 'SECRET_F', rationale: 'SECRET_G'};
  Object.assign(data, privateFields); Object.assign(data.reply, privateFields); Object.assign(data.evidence[0], privateFields);
  assert.ok(!renderResult(data).includes('SECRET_'));
});
test('untrusted content is escaped and unsafe source URLs are not links', () => {
  const data = fixture(); data.reply.text = '<img src=x onerror=alert(1)>';
  data.evidence[0].urls = ['javascript:alert(1)', 'https://support.apple.com/test'];
  const html = renderResult(data);
  assert.ok(html.includes('&lt;img')); assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('javascript:')); assert.ok(html.includes('rel="noopener noreferrer"'));
});
test('API errors do not expose response bodies; offline errors are actionable', async () => {
  await assert.rejects(analyze('query', async () => ({ok: false, status: 503, text: () => 'SECRET_STACK'})), /Analysis unavailable/);
  await assert.rejects(analyze('query', async () => {throw new TypeError('SECRET_URL');}), /Cannot reach the local API/);
  await assert.rejects(analyze('query', async () => ({ok: false, status: 409})), /Another analysis/);
});
test('API submits only the message and returns server data unchanged', async () => {
  const data = fixture();
  const result = await analyze('query', async (url, request) => {
    assert.equal(url, '/api/analyze'); assert.deepEqual(JSON.parse(request.body), {message: 'query'});
    return {ok: true, json: async () => data};
  });
  assert.equal(result, data);
});
test('semantic controls and reduced motion are provided', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const css = await readFile(new URL('../styles.css', import.meta.url), 'utf8');
  for (const token of ['for="message"', 'id="message"', 'role="status"', 'role="alert"', 'type="submit"', 'SYNTHETIC DEMOS']) assert.ok(html.includes(token));
  assert.ok(css.includes('prefers-reduced-motion:reduce')); assert.ok(css.includes(':focus-visible'));
  assert.equal(demos.length, 4);
});

 test('zero retrieval attempts alone never imply a policy short-circuit', () => {
  const data=fixture(); data.final_action='ESCALATE'; data.risk.level='critical'; data.retrieval.attempts=0; data.reply=null; data.generation_attempts=0;
  assert.ok(!renderResult(data).includes('Policy short-circuit'));
  assert.ok(!renderResult(data).includes('Policy block'));
  data.decision_reasons=['early_critical_hazard_short_circuit'];
  assert.ok(renderResult(data).includes('Policy short-circuit'));
  assert.ok(renderResult(data).includes('Human review was selected before response construction'));
 });
 test('clarification keeps failed verification and false grounding visible', () => {
  const data=fixture(); data.final_action='ASK_CLARIFICATION'; data.reply.grounded=false; data.reply.used_evidence_ids=[]; data.verification_passed=false;
  const html=renderResult(data);
  for(const text of ['Resolution withheld','Clarification request','Not passed','False','Cited sources: 0']) assert.ok(html.includes(text));
  assert.ok(!html.includes('not applicable'));
  data.verification_passed=null;
  assert.equal(traceStages(data)[3].status,'unavailable');
 });
 test('limited evidence and nullable fields do not acquire invented values', () => {
  const data=fixture(); data.retrieval.sufficiency='insufficient'; data.intent_confidence=null; data.risk=null;
  const html=renderResult(data); assert.ok(html.includes('Limited evidence: insufficient')); assert.ok(html.includes('Not assessed')); assert.ok(html.includes('Not available'));
 });
 test('long customer, source and reply text stays escaped and intact', () => {
  const data=fixture(); const long='Long<&>'.repeat(1000); data.reply.text=long; data.evidence[0].customer_problem=long; data.evidence[0].evidence_id='x'.repeat(300);
  const html=renderResult(data); assert.ok(html.includes('Long&lt;&amp;&gt;'.repeat(1000))); assert.ok(!html.includes('Long<&>'));
 });
