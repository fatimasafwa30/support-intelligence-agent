# Support Decision Console — Milestone 20

## Integration boundary

The repository is a Python inference/evaluation project with no existing web
stack. `AgentController.process_query()` returns an `AgentState` dataclass.
The console calls this unchanged controller with an explicitly injected existing
`MockReplyGenerator` (the deterministic offline template provider used by the
evaluation runner). Classification and retrieval use the real frozen artifacts.

Browser-native ES modules, semantic HTML and CSS avoid a second dev server and
framework dependencies. Node is only used for reproducible asset copying,
syntax checks and tests. Python's standard-library HTTP server is bound to
127.0.0.1. This is a local evaluator prototype, not an internet-facing deployment.

## Contract

- `GET /api/health`: actual artifact readiness and offline mode.
- `GET /api/schema`: the version-1 response JSON Schema.
- `POST /api/analyze`: exactly `{"message": "..."}`, nonblank, at most 4,000
  characters. Extra fields are rejected. Responses are validated with the
  already-installed `jsonschema` package and reject nonfinite numbers.
- HTTP 400: invalid input; 403: nonlocal host or cross-origin request; 409:
  another analysis is running; 503: artifact/controller failure. Error responses
  never include exception strings, request bodies or stack traces.

The adapter serializes individual fields, never `AgentState.to_dict()` wholesale.
Exposed fields are customer message, predicted intent/confidence, final action,
target queue, policy reason codes, risk level/categories, retrieval attempt count,
top similarity, evidence sufficiency, evidence IDs/similarities/customer problems/
brand resolutions/URLs, post-guard reply text, grounded flag, cited evidence IDs/
URLs, provider, verification flag, generation attempts and measured latency.

`top_similarity` is null when retrieval did not run; `verification_passed` is null
when no reply exists. These are unexecuted operations, not zero-score outcomes.
Confidence is an uncalibrated 0–1 model score. Similarity is a retrieval score,
not a success probability.

Not exposed: environment/configuration secrets, Golden annotations, judge scores,
human ratings, conversation identifiers, full execution traces, trace details,
risk/generator rationales, internal exception messages or hidden reasoning.
Human-readable policy reasons only replace underscores and case; exact returned
codes remain available in a disclosure.

Only the four built assets are served. The server does not serve the repository
filesystem, reports or `.env`. It requires a local Host and a same-origin Origin
when present, has no CORS allowance, uses a self-only content security policy and
disables caching. User/history text is HTML-escaped; only HTTP(S) evidence URLs
become links. No request history is stored and no customer reply is sent.

## Workspace structure

`index.html` owns the shell and message composer. `app.js` owns request/health
state, demo selection, keyboard submission and copy feedback. `view.js` owns
escaped presentation and the API request helper; `styles.css` owns responsive
layout and status semantics.

The workspace uses a compact composer, dominant decision headline and policy
basis, observable Decision Trace, then an asymmetric evidence/reply workbench.
There is no navigation rail. The trace follows actual controller order: Understand,
Assess risk, Retrieve, Verify, Decide. Skipped operations are explicitly marked;
clarification does not imply verification failed.

Ranked evidence selectors show similarity and customer previews. The selected
source separates CUSTOMER and APPLESUPPORT text, with expandable source metadata.
Citation indicators derive exclusively from returned used_evidence_ids membership.
The response is presented as support work product; no-reply outcomes show an
intentional automation stop and handoff destination.

Empty, waiting, complete, unavailable, failed-request, no-evidence and no-reply
states are explicit. While awaiting the synchronous controller, all trace stages
say Awaiting result; there are no invented stage completions or percentages.
Submitting clears the old result so it cannot be mistaken for a new response.

Accessibility: labelled textarea, native buttons and details, skip link, focus
outlines, text labels in addition to action colors, status/alert live regions,
Ctrl/Cmd+Enter submission, focus on completion, copy feedback, reduced-motion
support and stacked narrow-screen layout. Source links open with noopener and
noreferrer. No custom keyboard traps or hover-only controls.

## Demo provenance

All four shortcuts are labelled synthetic. Battery drain reuses synthetic
smoke-test wording present in the retrieval scripts; security, damage and
greeting prompts are synthetic product inputs. No Golden rows or labels are read
to select or run them. No expected action is attached to a prompt; the current
controller decides every time.

Observed with existing local artifacts: battery drain auto-handled; account
security escalated to account_security; greeting requested clarification. These
observations are not evaluation metrics or guaranteed general behavior.

## Validation and limits

Python tests cover request/response schema, all action contracts, null semantics,
private-field omission, runtime errors, busy-state rejection, real loopback HTTP,
cross-origin rejection and blocked filesystem routes. Frontend tests cover input
validation, all action presentations, real field interpolation, null/no-evidence
states, citation membership, HTML/URL safety, private-field omission, API failures,
layout order, semantic controls and reduced motion. Browser checks exercise real
offline inference, keyboard submission, disclosures, copy feedback and responsive
layouts; test fixtures are never used as operational fallback data.

Limitations:

- Original pre-guard drafts are not retained by the controller. Only its actual
  post-guard reply is shown; no invented draft is displayed.
- The controller does not stream events, so live per-stage progress is unavailable.
- Policy/grounding flags are displayed faithfully. A URL check does not establish
  semantic correctness, and historical short links can lead to DM/contact flows.
- Existing historical text can contain profanity; it is displayed verbatim.
- Frozen local artifacts are required and trusted. A fresh clone without those
  ignored artifacts cannot perform inference.
- There is no authentication, persistent history, database, message delivery or
  external LLM provider. Keep this server local.
- Python's standard-library server and syntax-only JS linting are deliberately
  limited to a local take-home demonstration.
