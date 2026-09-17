import {analyze, canSubmit, demos, renderResult, renderPending} from './view.js';

const message = document.querySelector('#message');
const submit = document.querySelector('#analyze');
const results = document.querySelector('#results');
const error = document.querySelector('#error');
const status = document.querySelector('#request-status');
const reconnect = document.querySelector('#reconnect');
let busy = false;
let ready = false;

function update() {
  submit.disabled = !canSubmit(message.value, busy, ready);
  for (const button of document.querySelectorAll('#demos button')) button.setAttribute('aria-pressed', String(button.dataset.message === message.value));
}

function inputChanged() {
  // Never leave a previous decision beneath a different customer message.
  results.hidden = true;
  error.hidden = true;
  document.querySelector('#empty').hidden = false;
  update();
}

async function health() {
  reconnect.disabled = true;
  ready = false; update();
  document.querySelector('#health').textContent = 'Connecting…';
  try {
    const response = await fetch('/api/health', {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error();
    const data = await response.json();
    ready = data.ready === true;
    document.querySelector('#health').textContent = ready ? '● Local system ready' : '○ Model artifacts unavailable';
  } catch {
    ready = false;
    document.querySelector('#health').textContent = '○ Local API unavailable';
  } finally {
    const notice = document.querySelector('#connection-notice');
    notice.hidden = ready;
    notice.textContent = ready ? '' : 'Analysis is paused. Check that the console server is running and its model artifacts are available, then select Reconnect. Your message stays here.';
    document.querySelector('.connection').dataset.state = ready ? 'ready' : 'unavailable';
    reconnect.hidden = ready; reconnect.disabled = false; update();
  }
}

for (const demo of demos) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = demo.label;
  button.dataset.message = demo.message;
  button.setAttribute('aria-pressed', 'false');
  button.addEventListener('click', () => {message.value = demo.message; message.focus(); inputChanged();});
  document.querySelector('#demos').append(button);
}

message.addEventListener('input', inputChanged);
message.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !submit.disabled) {
    event.preventDefault(); document.querySelector('#analyze-form').requestSubmit();
  }
});
reconnect.addEventListener('click', health);

document.querySelector('#analyze-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!canSubmit(message.value, busy, ready)) return;
  busy = true; update(); message.disabled = true;
  for (const button of document.querySelectorAll('#demos button')) button.disabled = true;
  error.hidden = true; results.hidden = true;
  document.querySelector('#empty').hidden = true;
  status.innerHTML = renderPending();
  status.className = 'loading';
  results.setAttribute('aria-busy', 'true');
  submit.textContent = 'Analyzing…';
  submit.setAttribute('aria-busy', 'true');
  try {
    const data = await analyze(message.value.trim());
    results.innerHTML = renderResult(data);
    results.hidden = false;
    for (const selector of results.querySelectorAll('[data-evidence-index]')) {
      selector.addEventListener('click', () => {
        for (const button of results.querySelectorAll('[data-evidence-index]')) {
          button.setAttribute('aria-pressed', String(button === selector));
          button.querySelector('.selection-label').textContent = button === selector ? 'Selected' : 'Inspect';
        }
        for (const panel of results.querySelectorAll('.source-document')) {
          panel.hidden = panel.id !== selector.getAttribute('aria-controls');
        }
      });
    }
    const title = document.querySelector('#result-title');
    title.focus({preventScroll: true});
    if (results.getBoundingClientRect().top < 0) results.scrollIntoView({block: 'start'});
    const copy = document.querySelector('#copy-reply');
    copy?.addEventListener('click', async () => {
      try {await navigator.clipboard.writeText(data.reply.text); document.querySelector('#copy-status').textContent = 'Copied';}
      catch {document.querySelector('#copy-status').textContent = 'Copy unavailable. Select the reply text to copy.';}
    });
  } catch (failure) {
    // Error messages are generated locally; never show a raw backend body/stack.
    error.textContent = ['Unsupported response contract', 'Unexpected end of JSON input'].includes(failure.message) ? 'The API returned an unsupported response. Restart the local server.' :
      /^(Analysis unavailable|Another analysis|Cannot reach|The local analysis)/.test(failure.message) ? failure.message : 'The response could not be displayed. Check the local server and try again.';
    error.hidden = false;
    if (/^Cannot reach/.test(failure.message)) await health();
  } finally {
    busy = false; message.disabled = false;
    for (const button of document.querySelectorAll('#demos button')) button.disabled = false;
    status.textContent = ''; status.className = ''; results.setAttribute('aria-busy', 'false');
    submit.removeAttribute('aria-busy');
    submit.textContent = 'Analyze message ↗'; update();
  }
});

// Tab Navigation (Agent Console / Evaluation Benchmark / Failure Lab)
const tabIds = ['console', 'evaluation', 'failure-lab'];

export function switchTab(tabId) {
  for (const id of tabIds) {
    const button = document.querySelector?.(`#tab-${id}`);
    const panel = document.querySelector?.(`#panel-${id}`);
    const active = id === tabId;
    button?.setAttribute?.('aria-selected', String(active));
    if (button?.classList?.toggle) button.classList.toggle('active', active);
    if (panel) {
      panel.hidden = !active;
      if (panel.classList?.toggle) panel.classList.toggle('active', active);
    }
  }
  try {
    if (typeof history !== 'undefined' && history.replaceState) {
      history.replaceState(null, '', `#${tabId}`);
    }
  } catch {}
}

for (const id of tabIds) {
  const button = document.querySelector?.(`#tab-${id}`);
  button?.addEventListener?.('click', () => switchTab(id));
  button?.addEventListener?.('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const idx = tabIds.indexOf(id);
      if (idx >= 0) {
        const nextIdx = e.key === 'ArrowRight' ? (idx + 1) % tabIds.length : (idx - 1 + tabIds.length) % tabIds.length;
        document.querySelector?.(`#tab-${tabIds[nextIdx]}`)?.focus?.();
        switchTab(tabIds[nextIdx]);
      }
    }
  });
}

try {
  if (typeof window !== 'undefined' && window.location && window.location.hash) {
    const initialHash = window.location.hash.replace('#', '');
    if (tabIds.includes(initialHash)) {
      switchTab(initialHash);
    }
  }
} catch {}

health();
