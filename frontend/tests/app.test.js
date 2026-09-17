import test from 'node:test';
import assert from 'node:assert/strict';

// Minimal DOM ports for testing the real app's asynchronous state transitions.
// Rendering and actual browser layout are checked separately.
class Element {
  value = ''; hidden = false; disabled = false; textContent = ''; innerHTML = '';
  dataset = {}; attributes = {}; listeners = {}; children = [];
  addEventListener(type, callback) { this.listeners[type] = callback; }
  async fire(type, event = {}) { return this.listeners[type]?.({preventDefault() {}, ...event}); }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  append(node) { this.children.push(node); }
  focus() {}
  getBoundingClientRect() { return {top: 1}; }
  querySelectorAll() { return []; }
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('real app reconnects, blocks repeated submissions and recovers from safe errors', async () => {
  const originalDocument = globalThis.document;
  const originalFetch = globalThis.fetch;
  const elements = new Map();
  const node = key => {
    if (!elements.has(key)) elements.set(key, new Element());
    return elements.get(key);
  };
  let healthy = false, posts = 0, finish;
  globalThis.document = {
    querySelector: node,
    querySelectorAll: () => node('#demos').children,
    createElement: () => new Element(),
  };
  globalThis.fetch = async url => {
    if (url === '/api/health') return {ok: true, json: async () => ({ready: healthy})};
    posts++;
    return new Promise(resolve => {finish = resolve;});
  };
  try {
    await import(`../app.js?test=${Date.now()}`);
    await settle();
    assert.equal(node('#analyze').disabled, true);
    assert.equal(node('#connection-notice').hidden, false);
    node('#message').value = 'A synthetic inquiry';
    await node('#message').fire('input');
    assert.equal(node('#analyze').disabled, true);
    healthy = true;
    const reconnecting = node('#reconnect').fire('click');
    assert.equal(node('#analyze').disabled, true);
    await reconnecting;
    assert.equal(node('#analyze').disabled, false);
    assert.equal(node('#connection-notice').hidden, true);
    const first = node('#analyze-form').fire('submit');
    assert.equal(node('#analyze').disabled, true);
    assert.match(node('#request-status').innerHTML, /Awaiting result/);
    await node('#analyze-form').fire('submit');
    assert.equal(posts, 1);
    finish({ok: false, status: 503});
    await first;
    assert.equal(node('#analyze').disabled, false);
    assert.equal(node('#error').hidden, false);
    assert.match(node('#error').textContent, /^Analysis unavailable/);
    assert.equal(node('#message').value, 'A synthetic inquiry');
    assert.equal(node('#request-status').textContent, '');
    const demo = node('#demos').children[0];
    await demo.fire('click');
    assert.equal(demo.attributes['aria-pressed'], 'true');
    node('#message').value = 'Edited inquiry';
    await node('#message').fire('input');
    assert.equal(demo.attributes['aria-pressed'], 'false');
    node('#message').value = '  ';
    await node('#message').fire('input');
    assert.equal(node('#analyze').disabled, true);
  } finally {
    globalThis.document = originalDocument;
    globalThis.fetch = originalFetch;
  }
});
