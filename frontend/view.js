// Presentation only. Policy, scores, citations and routing originate in Python.
export const demos = [
  {label: 'Battery drain', message: 'My iPhone battery is draining so fast after the update'},
  {label: 'Account security', message: 'My Apple ID was hacked and someone changed my password.'},
  {label: 'Damaged screen', message: 'My iPhone has a cracked screen. How can I arrange a repair?'},
  {label: 'Unclear issue', message: 'Hello'}
];
export const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const humanize = value => String(value ?? '').replaceAll('_', ' ').toLowerCase();
export const metric = (value, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'Not available';
export const canSubmit = (message, busy, ready) => Boolean(message.trim()) && message.length <= 4000 && !busy && ready;
export const actions = {
  AUTO_HANDLE: {title: 'Auto-handle', label: 'AUTOMATED RESPONSE', style: 'auto'},
  ASK_CLARIFICATION: {title: 'Ask clarification', label: 'MORE CONTEXT NEEDED', style: 'clarify'},
  ESCALATE: {title: 'Escalate to a human', label: 'HUMAN REVIEW', style: 'escalate'}
};

function links(urls) {
  return urls.filter(url => {try {return ['http:', 'https:'].includes(new URL(url).protocol);} catch {return false;}})
    .map(url => `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(url)} <span aria-hidden="true">↗</span></a>`).join('');
}

export function criticalShortCircuit(data) {
  return data.final_action === 'ESCALATE' && data.decision_reasons.includes('early_critical_hazard_short_circuit') && data.retrieval.attempts === 0 && data.generation_attempts === 0 && !data.reply;
}
export function decisionExplanation(data) {
  if (criticalShortCircuit(data)) return 'Critical-risk policy stopped automation before retrieval and response construction.';
  if (data.final_action === 'ASK_CLARIFICATION') return 'Resolution withheld. The controller selected a request for more context.';
  if (data.final_action === 'ESCALATE') return 'The controller selected human review. Inspect the policy reasons for this handoff.';
  return 'The controller selected automated handling. Review the historical support evidence and returned response.';
}
const stageNames = ['Understand', 'Assess risk', 'Retrieve', 'Verify', 'Decide'];

export function traceStages(data) {
  const reply = data.reply;
  const stages = [
    {name: stageNames[0], value: humanize(data.predicted_intent) || 'Not available', detail: `Confidence ${metric(data.intent_confidence)}`, status: data.predicted_intent ? 'observed' : 'unavailable'},
    {name: stageNames[1], value: data.risk?.level ?? 'Not assessed', detail: data.risk?.categories.length ? data.risk.categories.map(humanize).join(' · ') : 'No categories returned', status: data.risk ? 'observed' : 'unavailable'},
    {name: stageNames[2], value: data.retrieval.attempts ? data.retrieval.sufficiency : 'Not run', detail: data.retrieval.attempts ? `Top similarity ${metric(data.retrieval.top_similarity)} · ${data.retrieval.attempts} attempt${data.retrieval.attempts === 1 ? '' : 's'}` : 'No retrieval attempts', status: data.retrieval.attempts ? 'observed' : 'skipped'},
    {name: stageNames[3], value: !reply ? 'Not run' : data.verification_passed === true ? 'Passed' : data.verification_passed === false ? 'Not passed' : 'Not available', detail: reply ? `Grounded flag: ${reply.grounded ? 'true' : 'false'}` : 'No reply constructed', status: !reply ? 'skipped' : data.verification_passed === true ? 'observed' : data.verification_passed === false ? 'attention' : 'unavailable'},
    {name: stageNames[4], value: actions[data.final_action].title, detail: humanize(data.target_queue), status: actions[data.final_action].style}
  ];
  if (data.final_action === 'ESCALATE' && data.risk && ['critical', 'high'].includes(data.risk.level)) stages[1].status = 'attention';
  return stages;
}

export function renderPending() {
  return `<div class="pending-heading"><span class="request-dot" aria-hidden="true"></span><strong>Analysis in progress</strong><span>Waiting for the local controller. Stage outputs appear when the request completes.</span></div><ol aria-label="Decision stages" class="trace pending">${stageNames.map((name, i) => `<li><span class="trace-node">0${i + 1}</span><h2>${name}</h2><p>Awaiting result</p></li>`).join('')}</ol>`;
}

function renderTrace(data) {
  return `<section class="trace-section" aria-label="Decision trace"><div class="section-heading"><h2>Decision trace</h2><span class="quiet">Observed outputs · ${metric(data.latency_ms, 0)} ms total</span></div><ol aria-label="Decision stages" class="trace">${traceStages(data).map((stage, i) => `<li class="stage-${stage.status}"><span class="trace-node" aria-label="${stage.status === 'skipped' ? 'Not run' : stage.status === 'unavailable' ? 'Not available' : 'Observed output'}">${String(i + 1).padStart(2, '0')}</span><span class="stage-label">${stage.status === 'skipped' ? 'Not run' : stage.status === 'unavailable' ? 'Unavailable' : stage.status === 'attention' ? (i === 1 && criticalShortCircuit(data) ? 'Policy block' : 'Needs attention') : stage.status === 'escalate' ? 'Automation stopped' : stage.status === 'clarify' ? 'Awaiting clarification' : stage.status === 'auto' ? 'Auto-handle' : 'Observed'}</span><h3>${stage.name}</h3><strong>${escape(stage.value)}</strong><p>${escape(stage.detail)}</p></li>`).join('')}</ol>${criticalShortCircuit(data) ? '<p class="path-note"><strong>Policy short-circuit → human review.</strong> Retrieval and verification did not run.</p>' : ''}</section>`;
}

export function renderEvidence(data) {
  if (!data.evidence.length) return `<div class="no-evidence"><span class="document-mark" aria-hidden="true">—</span><div><strong>No historical evidence returned</strong><p>${criticalShortCircuit(data) ? 'Retrieval did not run: the critical-risk policy bypassed it. This is not a retrieval failure.' : data.retrieval.attempts === 0 ? 'Retrieval did not run for this decision.' : 'The retrieval step returned no matching resolutions.'}</p><span class="quiet">No source material to inspect.</span></div></div>`;
  const selectors = data.evidence.map((e, i) => `<button type="button" class="source-selector" data-evidence-index="${i}" aria-pressed="${i === 0}" aria-controls="source-${i}"><span>CASE 0${i + 1}</span><strong>${metric(e.similarity)}</strong><small>similarity</small><span class="selection-label">${i === 0 ? 'Selected' : 'Inspect'}</span><span class="source-preview">${escape(e.customer_problem)}</span>${data.reply?.used_evidence_ids.includes(e.evidence_id) ? '<span class="cited-dot" title="Cited in reply" aria-label="Cited in reply">Cited</span>' : ''}</button>`).join('');
  const panels = data.evidence.map((e, i) => `<article class="source-document" id="source-${i}" ${i === 0 ? '' : 'hidden'} aria-label="Historical case ${i + 1}"><div class="source-provenance"><span>SOURCE RECORD / ${String(i + 1).padStart(2, '0')}</span>${data.reply?.used_evidence_ids.includes(e.evidence_id) ? '<span class="cited">Cited in reply</span>' : '<span>Retrieved context</span>'}</div><div class="source-exchange"><section><h3>CUSTOMER</h3><p>${escape(e.customer_problem)}</p></section><section class="brand-resolution"><h3>APPLESUPPORT</h3><p>${escape(e.brand_resolution)}</p><div class="source-links">${links(e.urls)}</div></section></div><details class="source-metadata"><summary>Source metadata</summary><dl><dt>Evidence ID</dt><dd><code>${escape(e.evidence_id)}</code></dd><dt>Retrieval rank</dt><dd>${i + 1}</dd><dt>Similarity score</dt><dd>${metric(e.similarity, 4)}</dd></dl></details></article>`).join('');
  return `<div class="source-selectors" role="group" aria-label="Select historical evidence">${selectors}</div>${panels}`;
}

function renderResponse(data) {
  const reply = data.reply;
  const clarify = data.final_action === 'ASK_CLARIFICATION';
  const title = clarify ? 'Clarification request' : !reply ? 'Human handoff' : 'Customer response';
  const verification = data.verification_passed === true ? 'Passed' : data.verification_passed === false ? 'Not passed' : 'Not available';
  const preGeneration = data.decision_reasons.includes('escalation_engine_pre_generation_short_circuit') && !reply && data.generation_attempts === 0;
  return `<section class="reply-panel" aria-labelledby="reply-title"><div class="section-heading"><h2 id="reply-title">${title}</h2><span class="eyebrow">WORK PRODUCT</span></div>
    ${clarify ? '<p class="response-context"><strong>Resolution withheld · more context requested.</strong> Clarification is a policy outcome; the verification and grounding values below still describe the returned reply.</p>' : ''}
    ${reply ? `<div class="reply-paper"><span class="eyebrow">${clarify ? 'REQUEST FOR INFORMATION' : 'CUSTOMER-FACING REPLY'} · NOT SENT</span><p class="reply-text">${escape(reply.text)}</p><div class="reply-tools"><span class="citation-count">Cited sources: ${data.evidence.filter(e => reply.used_evidence_ids.includes(e.evidence_id)).length}</span><button class="secondary" id="copy-reply" type="button">Copy reply <span aria-hidden="true">↗</span></button><span id="copy-status" role="status"></span></div></div>
    <dl class="reply-checks"><div><dt>Verification</dt><dd>${verification}</dd></div><div><dt>Grounded flag</dt><dd>${reply.grounded ? 'True' : 'False'}</dd></div><div><dt>Generator</dt><dd>${escape(reply.provider)} / offline</dd></div><div><dt>Generation attempts</dt><dd>${data.generation_attempts}</dd></div></dl><p class="footnote">Citation and URL checks do not establish technical correctness. Historical links may lead to contact or DM flows.</p>` : `<div class="no-reply"><span class="eyebrow">AUTOMATION STOPPED</span><h3>No reply constructed</h3><p>${criticalShortCircuit(data) || preGeneration ? 'Human review was selected before response construction.' : 'The controller returned no customer response.'}</p><div class="handoff"><span>HANDOFF DESTINATION</span><strong>${escape(humanize(data.target_queue))}</strong></div><p class="footnote">Routing recommendation only. No ticket or message is sent.</p></div>`}</section>`;
}

export function renderResult(data) {
  const action = actions[data.final_action];
  if (!action || data.version !== 1) throw new Error('Unsupported response contract');
  return `<div class="analysis ${action.style}">
    <section class="decision-head" aria-labelledby="decision-title"><div><span class="eyebrow" id="result-title" tabindex="-1">FINAL DECISION / ${data.final_action}</span><h2 id="decision-title">${action.title}</h2></div><div class="decision-context"><p>${decisionExplanation(data)}</p>${data.decision_reasons.length ? `<p class="primary-reason">Policy · ${escape(humanize(data.decision_reasons[0]))}</p>` : ''}<p class="routing">Route to <strong>${escape(humanize(data.target_queue))}</strong></p></div></section>
    ${renderTrace(data)}
    <div class="workbench"><section class="evidence-panel" aria-labelledby="evidence-title"><div class="section-heading"><h2 id="evidence-title">Historical evidence <span class="source-count" aria-label="${data.evidence.length} sources returned">${data.evidence.length.toString().padStart(2, '0')}</span></h2><span class="eyebrow">SOURCE INSPECTOR</span></div><div class="evidence-caption"><span>Sufficiency <strong>${escape(data.retrieval.sufficiency)}</strong></span><span>Top similarity <b>${metric(data.retrieval.top_similarity)}</b></span></div>${['insufficient','empty','borderline'].includes(data.retrieval.sufficiency) ? '<p class="evidence-warning">Limited evidence: ' + escape(data.retrieval.sufficiency) + '. The retrieval score alone does not establish a supported resolution.</p>' : ''}${renderEvidence(data)}</section>${renderResponse(data)}</div>
    <details class="policy-codes"><summary>Why this decision · policy reasons &amp; exact codes</summary><ul>${data.decision_reasons.map(r => `<li><span>${escape(humanize(r))}</span><code>${escape(r)}</code></li>`).join('')}</ul></details>
  </div>`;
}
export async function analyze(message, fetcher = fetch) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetcher('/api/analyze', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message}), signal: controller.signal});
    if (!response.ok) throw new Error(response.status === 409 ? 'Another analysis is running. Try again shortly.' : 'Analysis unavailable. Check the local server and model artifacts, then try again.');
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The local analysis timed out. Check the server before trying again.');
    if (error instanceof TypeError) throw new Error('Cannot reach the local API. Start the server and reconnect.');
    throw error;
  } finally {clearTimeout(timeout);}
}


