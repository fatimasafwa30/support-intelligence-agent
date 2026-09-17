# Support Intelligence Agent

> **Production-Grade Automated Support Decision System & Evaluation Benchmark**
> *Deterministic intent classification, historical resolution retrieval, grounded reply drafting, and conservative safety routing on the AppleSupport Twitter dataset.*

---

## 1. Project Overview

### Problem Being Solved
Enterprise customer support teams face high ticket volumes with strict response SLAs, where repetitive technical questions consume specialist bandwidth while mission-critical safety or account security issues risk delayed handling. Building an automated agent for customer service requires solving a fundamental tension:
- **Automation ROI**: Resolving common, well-documented inquiries immediately via self-service guidance.
- **Safety and Trust**: Preventing hallucinated troubleshooting steps, catching user-stated tool failures, and strictly escalating credential theft, hardware emergencies, and legal hazards to human specialists.

This project implements an end-to-end support orchestration pipeline that ingests customer messages, classifies underlying intent with confidence scoring and abstention, retrieves relevant historical resolution evidence, synthesizes grounded replies with verified links, and enforces deterministic, policy-governed routing.

### AppleSupport as Selected Brand
From the multi-brand Kaggle Customer Support dataset, **AppleSupport** (`@AppleSupport`) was selected as the operational domain:
1. **High Inbound Volume**: 792,572 tweets across 107,314 multi-turn conversations provided statistical scale for robust evaluation.
2. **Technical Depth & Clear Boundaries**: Requests span diverse, unambiguous technical topics (battery health, iCloud sync, iOS software updates, hardware repair, Two-Factor Authentication).
3. **Rich Multi-Turn Resolutions**: Historical support threads frequently document customer problem statements paired with verified official agent resolution steps and support links.

### The Dataset
The system is built on the Kaggle *Customer Support on Twitter* dataset (~2.8M tweets total). The corpus was filtered to `@AppleSupport`, reconstructed into threaded multi-turn dialogues using `first_tweet_id`, and partitioned at the conversation level into strict, non-overlapping train, development, test, and evaluation sets.

> [!NOTE]
> **Dataset Acquisition**: Raw dataset files are excluded from version control. To rebuild conversations and historical resolution indices from scratch, download `twcs.csv` directly from [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it under `data/raw/twcs.csv`. The evaluation benchmark (`data/golden/golden_annotation.csv`) and frozen model artifacts are pre-packaged for immediate reproduction.

### What the Agent Does
For each incoming customer query, the system executes an automated decision pipeline:

```
[Inbound Customer Message]
           │
           ▼
 [Intent Classification]  ──(Confidence < 0.25)──► [Abstain to other_unclear]
           │                                                │
           ▼                                                ▼
   [Risk Assessment]  ─────(Critical Safety Hazard)──────► [ESCALATE] (Short-Circuit)
           │
           ▼
[Resolution Retrieval] ──(Borderline / Weak)──► [Query Reformulation Retry]
           │
           ▼
  [Grounded Generation] ──(Citation Guard)──► [Verified Reply / Strip Links]
           │
           ▼
[Deterministic Policy] ──► AUTO_HANDLE | ASK_CLARIFICATION | ESCALATE
```

1. **Classify Intent**: Maps the customer query to a 15-intent domain taxonomy using TF-IDF and calibrated logistic regression. Queries falling below confidence threshold $\tau = 0.25$ abstain safely to `other_unclear`.
2. **Detect Risk & Critical Hazards**: Identifies hardware risks (battery swelling, electrical sparks), account security boundaries, and angry escalations. Critical safety hazards trigger immediate escalation short-circuits.
3. **Retrieve Historical Evidence**: Searches a corpus of 74,415 verified training resolution pairs using cosine similarity. If initial retrieval is borderline ($0.25 \le \text{similarity} < 0.35$), the system reformulates the query by stripping noise tokens and retries retrieval.
4. **Draft Grounded Reply**: Constructs an evidence-grounded response citing official Apple support resources. The programmatic citation guard verifies that cited URLs exist verbatim in historical brand records.
5. **Decide Action & Target Queue**:
   - **`AUTO_HANDLE`**: Automated self-service guidance dispatched directly to the customer (requires high intent confidence $\ge 0.50$, strong retrieval similarity $\ge 0.35$, and a verified support link).
   - **`ASK_CLARIFICATION`**: Requests targeted diagnostic information (device model, OS version) when intent or retrieval is borderline.
   - **`ESCALATE`**: Transfers the case to specialized human queues (`general_support`, `hardware_repair`, `account_security`, `safety_team`) when technical uncertainty, high risk, or policy boundaries mandate human handling.

---

## 2. System Architecture

The system is engineered with strict separation between probabilistic models and deterministic safety policies:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     AgentController                                    │
│                                                                                        │
│  ┌──────────────────────┐   ┌──────────────────────┐   ┌────────────────────────────┐  │
│  │  Intent Classifier   │   │  Historical Corpus   │   │   Deterministic Policy     │  │
│  │  • TF-IDF N-Grams    │   │  • 74,415 TRAIN Pairs│   │   • Risk Scanning Engine   │  │
│  │  • Logistic Regr.    │   │  • Cosine Similarity │   │   • Threshold Gating       │  │
│  │  • Calibrated Probs  │   │  • Query Retry Loop  │   │   • Queue Routing Logic    │  │
│  └──────────┬───────────┘   └──────────┬───────────┘   └─────────────┬──────────────┘  │
│             │                          │                             │                 │
│             ▼                          ▼                             ▼                 │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                               Execution State Machine                            │  │
│  │  1. Risk & Hazard Scan ──► 2. Intent & Confidence ──► 3. Resolution Retrieval    │  │
│  │  4. Sufficiency Check  ──► 5. Grounded Draft     ──► 6. Action & Queue Dispatch  │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Conversation Reconstruction**: Raw tweet records are grouped into conversations using `first_tweet_id`, establishing chronological ordering and distinguishing inbound customer queries from official brand responses.
- **Leakage-Safe Partitioning**: Splitting is executed strictly by conversation ID. No conversation appears across multiple splits. All retrieval indices are built **strictly from TRAIN split data**; neither DEV, TEST, nor the Golden Set are ever ingested into the retrieval index.
- **Intent Classifier**: TF-IDF n-gram feature extractor (1–2 n-grams, sublinear scaling, 50,000 features) coupled with a multinomial Logistic Regression model. Outputs calibrated probability distributions over 15 intents. If $\max P(\text{intent}) < 0.25$, the prediction is abstained to `other_unclear`.
- **Resolution Retriever**: Inverted index of 74,415 customer problem $\to$ AppleSupport resolution pairs from the TRAIN split. Uses TF-IDF cosine similarity. If top similarity falls in the borderline band ($[0.25, 0.35)$), a query reformulation engine strips conversational noise and punctuation, re-ranking candidate resolutions.
- **Grounded Generation & Citation Guard**: Offline templated reply synthesizer that incorporates verified evidence steps. The `verify_and_filter_reply` guard verifies every extracted link against retrieved evidence text. Ungrounded or hallucinated URLs are stripped.
- **Centralized Escalation Policy**: An orchestrator (`AgentController`) coordinates all sub-components. Routing decisions require multi-factor conjunctions (confidence $\ge 0.50$, similarity $\ge 0.35$, verified link) rather than single-model trust.
- **Independent Evaluation Subsystem**: Decoupled offline test harnesses evaluating operational 3-way distribution, binary action accuracy, critical safety recall, under-escalation, intent accuracy, and LLM-as-a-judge quality rubrics.

---

## 3. Data + Intent Taxonomy

### Dataset Scale
The full Kaggle Twitter Customer Support dataset encompasses 2,811,774 tweets across 20+ corporations. Filtering to AppleSupport yields:
- **Total AppleSupport Tweets**: 792,572
- **Reconstructed Multi-Turn Conversations**: 107,314
- **TRAIN Split**: 60,460 conversations (74,415 customer-brand resolution pairs indexed for retrieval)
- **DEV Split**: 12,955 conversations (used for threshold tuning and hyperparameter selection)
- **TEST Split**: 33,649 conversations (held out for model generalization checks)
- **Golden Evaluation Set**: 250 conversations manually curated, annotated, and verified

### The 15-Intent Taxonomy
Developed through heuristic discovery over customer vocabulary and official support workflows:

| Intent Key | Description & Scope | Golden Support ($N=250$) |
| :--- | :--- | :---: |
| `device_hardware` | Physical defects, screen/casing damage, speaker/mic, button failure | 46 |
| `software_update` | iOS / macOS installation failures, boot loops, update storage errors | 44 |
| `battery_power` | Rapid drain, unexpected shutdown, charging failure, battery health | 38 |
| `other_unclear` | Ambiguous complaints, conversational banter, unparseable queries | 25 |
| `connectivity` | Wi-Fi disconnects, Bluetooth pairing, cellular "No Service" / searching | 19 |
| `apple_id_account` | Two-Factor Authentication, account lockouts, Apple ID name/email changes | 16 |
| `repair_service` | AppleCare claims, repair booking, warranty status, store appointments | 13 |
| `icloud` | iCloud storage capacity, Photo Library sync, backup restoration errors | 8 |
| `app_store` | App installation errors, download loops, App Store account credentials | 7 |
| `billing_payments` | Credit card declines, unrecognized charges, invoice inquiries | 7 |
| `security_privacy` | Suspicious phishing, unauthorized Apple ID logins, device passcode locks | 7 |
| `subscriptions_media`| Apple Music, iTunes match, recurring service cancellations | 7 |
| `orders_delivery` | Online order tracking, shipping address corrections, delivery delays | 6 |
| `how_to_information` | Operating system how-to questions, feature usage instructions | 5 |
| `setup_activation` | New device setup, iCloud Activation Lock, data migration | 2 |

### Golden Set Isolation Invariant
> [!IMPORTANT]
> **Strict Anti-Leakage Guarantee**: The 250-example Golden Set was sampled across stratified risk, difficulty, and intent slices. It was **strictly isolated from all training, silver-label heuristic tuning, retrieval indexing, and few-shot prompt construction**. No component of the system has ever trained on or indexed Golden Set conversations.

---

## 4. Evaluation Methodology

The system is evaluated against baseline models and measured across operational, intent, safety, and retrieval metrics.

### Evaluated Systems & Baselines
1. **Trivial Baseline (Always-Escalate)**: Routes 100% of queries to `ESCALATE`. Achieves 0.00% under-escalation and 100% critical recall, but delivers 0.00% automation yield. Due to the 180/250 Gold ESCALATE base rate, it achieves 72.00% binary action accuracy.
2. **Simple Baseline (Linear Action Classifier)**: Direct TF-IDF + Logistic Regression model trained to predict binary `AUTO_HANDLE` vs `ESCALATE` directly from text without intent, retrieval, or policy stages.
3. **Current Final Candidate (`AgentController`)**: Multi-stage orchestration integrating intent classification with confidence abstention ($\tau = 0.25$), TF-IDF historical resolution retrieval with retry, URL citation verification, and deterministic multi-tier risk policy.

### Metric Definitions
- **Operational 3-Way Routing Distribution**: Share of queries routed to `AUTO_HANDLE`, `ASK_CLARIFICATION`, and `ESCALATE`.
- **Secondary Binary Action Accuracy**: Matches between predicted action and gold action, where `AUTO_HANDLE` maps to `AUTO` and both `ASK_CLARIFICATION` and `ESCALATE` map to `ESCALATE` (reflecting operational withholding from immediate automation).
- **Under-Escalation Rate (Dangerous False Negatives)**: Proportion of gold `ESCALATE` cases incorrectly auto-handled ($\frac{\text{False AUTO}}{\text{Gold ESCALATE}}$). Target: $< 5.0\%$.
- **False Escalation Rate (Withholding Rate)**: Proportion of gold `AUTO_HANDLE` cases withheld from automated self-service ($\frac{\text{Withheld AUTO}}{\text{Gold AUTO}}$).
- **Critical Safety Recall**: Recall on critical safety, physical hazard, and severe security cases. Target: $100.0\%$.
- **Full Intent Accuracy**: Multiclass accuracy over all 250 Golden cases, including `other_unclear`.
- **Specific Intent Accuracy**: Accuracy over the 225 specific-intent queries excluding gold `other_unclear`, measuring classifier performance post-abstention.
- **Evidence URL Citation Integrity**: Programmatic verification rate of URLs in generated replies against retrieved historical brand evidence.

### Reply-Quality Rubric (LLM-as-a-Judge)
Generated replies are evaluated across five dimensions on a 1–5 integer scale:
1. **Groundedness**: Faithfulness of claims to retrieved historical evidence (no ungrounded troubleshooting steps).
2. **Correctness**: Technical accuracy and adherence to Apple support domain standards.
3. **Relevance**: Direct address of the customer's specific problem without extraneous tangents.
4. **Helpfulness**: Actionability and clarity of troubleshooting steps.
5. **Tone**: Professionalism, empathy, and brand alignment.

> [!WARNING]
> **Gemini Judge Evaluation Status: PARTIAL**:
> Out of 142 generated replies, LLM-as-a-judge scoring via `gemini-2.5-flash` was executed on an interim batch of 46 records (**39 SUCCESS**, **7 FAILED** due to API quotas/rate-limits). Across the 39 successfully judged records, the interim mean composite score is **3.14 / 5.00** (Groundedness: 2.28, Correctness: 3.46, Relevance: 3.00, Helpfulness: 2.90, Tone: 4.08).
> **Judge-Human Agreement Status**: 23 independent human ratings were intentionally collected (`reports/human_agreement_ratings.jsonl`). The clean intersection between the 23 human ratings and the 39 successful Gemini judge records yields exactly **N=9 examples** (excluding `golden_candidate_0066`, which is contaminated by prior score guidance). Because the offline evaluation harness does not compute inter-rater agreement on partial intersections, formal correlation metrics ($\kappa_w$, Spearman $\rho$) are deferred rather than fabricated over an incomplete sample. Zero agreement figures are fabricated.

---

## 5. Headline Results

Authoritative metrics verified against the frozen evaluation artifacts (`reports/agent_evaluation_summary.json`, `reports/failure_analysis.json`):

| Evaluation Dimension | Metric | Measured Frozen Value | Baseline Reference | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Benchmark Scale** | Evaluated Golden Sample ($N$) | **250** | 250 | Complete human Golden set |
| **Operational 3-Way** | `AUTO_HANDLE` Count / Share | **16** (6.40%) | — | Selective self-service |
| | `ASK_CLARIFICATION` Count / Share | **126** (50.40%) | — | Conservative diagnostic triage |
| | `ESCALATE` Count / Share | **108** (43.20%) | — | Direct specialist routing |
| **Action Routing** | Secondary Binary Action Accuracy | **72.00%** (180/250) | 70.80% (Linear) | Operational binary match |
| | Routing Macro-F1 | **50.85%** | 49.50% (Linear) | Reflects heavy class caution |
| **Safety Invariants** | Critical Hazard Recall | **100.00%** (10/10) | 100.00% (Linear) | Zero critical hazards missed |
| | Under-Escalation Rate (Dangerous FN) | **4.44%** (8/180) | 3.89% (Linear) | Within $<5\%$ safety budget |
| | Gold AUTO Withheld (False Escalation) | **88.57%** (62/70) | 94.29% (Linear) | 38 to ASK, 24 to ESCALATE |
| **Intent Diagnostics**| Full Golden Intent Accuracy | **51.20%** (128/250) | — | Includes `other_unclear` |
| | Specific-Intent Accuracy (Post-Abstention)| **48.00%** (108/225) | 56.44% (Raw No-Abstain)| 67 abstained to `other_unclear` |
| | High-Confidence Intent Accuracy | **81.48%** (66/81) | — | Precision on confidence $\ge 0.80$ |
| **Execution Traces** | Generated Reply Population | **142** | — | 16 AUTO + 126 ASK |
| | Pre-Generation Short-Circuits | **108** | — | 7 Pre-Retrieval + 101 Pre-Gen Policy |
| | Evidence URL Verification Rate | **40.85%** (58/142) | — | Verified links in generated text |
| **Retrieval Dynamics**| Initial $\to$ Final Mean Similarity | **0.3983 $\to$ 0.3997** | — | Modest lift from retry loop |
| | Retry Triggered Rate | **15.60%** (39/250) | — | Queries in borderline band |
| | Borderline Upgrade Rate | **3.88%** (4/103) | — | Upgraded to sufficient |

---

## 6. What is misleading about my headline number?

> [!CAUTION]
> ### Unusually Candid Production-Grade Appraisal
> At first glance, a **72.00% binary action accuracy** and **100.00% critical safety recall** suggest an agent that is ready for customer-facing deployment. **Marketing this headline number as production readiness would be profoundly misleading.**

Here is what the headline number hides:

1. **The Headline Accuracy is Heavily Buoyed by Class Imbalance**:
   In the Golden Set, 180 out of 250 customer queries (72.00%) have a gold label of `ESCALATE`. This means that a brainless **"Always-Escalate" trivial baseline** that refuses to answer every customer and forwards every ticket to a human agent achieves **exactly 72.00% binary action accuracy** without inspecting a single word. Our agent matches this accuracy (72.00%) through actual classification and retrieval, but its overall accuracy is anchored to the dominant class.

2. **Severe Self-Service Starvation (88.57% Withheld)**:
   Of the 70 cases in the Golden Set that human annotators verified were safe, clear, and appropriate for automated self-service resolution (`AUTO_HANDLE`), the system successfully auto-handled **only 8 of them (11.43% yield)**. It withheld **62 out of 70 valid auto-handle opportunities (88.57%)**:
   - 38 cases were diverted into `ASK_CLARIFICATION`
   - 24 cases were diverted into `ESCALATE`
   In production, an automation tool that deflects only 6.4% of total volume (16/250) while failing to resolve 88.6% of automatable inquiries fails its primary economic objective.

3. **Conservative Routing Hides Residual Dangerous Under-Escalations**:
   The agent's conservative stance (demanding confidence $\ge 0.50$, similarity $\ge 0.35$, and an extracted link) successfully shields users from many bad answers. However, it still committed **8 under-escalation failures (4.44% error rate)** on queries marked `ESCALATE`. These were not minor edge cases:
   - Shipping address change on an existing order where online chat was offline (`golden_candidate_0035`).
   - Customer requesting to disable Two-Factor Authentication (`golden_candidate_0132`).
   - iPhone trapped in an unrecoverable 30-second crash loop after an update (`golden_candidate_0062`).
   In all three cases, the system dispatched an automated reply promising self-service steps when live human intervention was mandatory.

4. **100% Critical Recall is Statistically Fragile**:
   The agent correctly escalated 10 out of 10 critical safety hazards (swelling batteries, sparks, account takeover). While essential, achieving 10/10 over a sample of size $N=10$ yields wide binomial confidence intervals ($95\%\text{ CI: }[69.2\%, 100.0\%]$). Relying purely on lexical keyword triggers does not guarantee that a novel formulation of physical danger will be caught in production.

**Summary**: 72.00% binary accuracy conceals both **excessive conservatism (automation starvation)** and **residual high-risk misses**. The system represents a solid safety-first baseline, but requires architectural refinements before handling live traffic.

---

## 7. Exactly Five Failure Modes

Condensed from the comprehensive failure analysis (`reports/failure_analysis.md`), preserving measured counts vs. causal hypotheses:

### Failure Mode 1: Under-Escalation on High-Risk / Complex Requests (Dangerous False Self-Service)
- **Impact Priority**: `CRITICAL (Safety, Security & Brand Trust)`
- **Measured Frequency**: 8 cases out of 180 Gold ESCALATE (4.44% under-escalation rate).
- **Concrete Golden Examples**:
  - `golden_candidate_0035`: Customer stated *"order may be ineligible for changes online and chat is unavailable"*; agent predicted `orders_delivery` (conf=0.909) and issued `AUTO_HANDLE` linking to a generic portal.
  - `golden_candidate_0132`: Customer complained Two-Factor Authentication was making the phone unusable and asked to turn it off; agent predicted `apple_id_account` (conf=0.636) and auto-handled.
  - `golden_candidate_0062`: Customer reported phone restarting every 30 seconds despite updating; agent predicted `software_update` (conf=0.863) and auto-handled.
- **Root-Cause Hypothesis**: The escalation policy permits `AUTO_HANDLE` whenever intent confidence $\ge 0.50$, retrieval similarity $\ge 0.35$, and an evidence URL exists, unless an explicit critical-hazard keyword fires. The system lacks negative semantic guards detecting statements of prior self-service failure or sensitive account alteration requests.

### Failure Mode 2: Over-Conservative Withholding & False Escalation of Automatable Queries
- **Impact Priority**: `HIGH (Operational Automation Deficit)`
- **Measured Frequency**: 62 cases out of 70 Gold AUTO withheld (88.57% false escalation rate; 38 to `ASK_CLARIFICATION`, 24 to `ESCALATE`).
- **Concrete Golden Examples**:
  - `golden_candidate_0006`: Customer asked how to change personal name associated with Apple ID; agent predicted `apple_id_account` with 0.968 confidence and 0.384 similarity, but escalated because no URL link was extracted from evidence.
  - `golden_candidate_0015`: Informational inquiry on intermittent vs full charging; agent predicted `battery_power` (conf=0.795), but escalated because lexical similarity was 0.271 (< 0.35 threshold).
  - `golden_candidate_0026`: Customer reported phone saying "Searching" / "No Service"; agent escalated due to similarity 0.330 (< 0.35).
- **Root-Cause Hypothesis**: Two rigid gates cause severe withholding: (1) requiring an extracted URL link forces escalation even when confidence is near-certain ($\ge 0.95$) and text troubleshooting steps exist; (2) rigid 0.35 TF-IDF cosine cutoff rejects conceptual questions lacking exact keyword overlap.

### Failure Mode 3: Monolithic Clarification Loop & Premature Triage Trap (Static Canned Responses)
- **Impact Priority**: `HIGH (Customer Dialogue Friction)`
- **Measured Frequency**: 126 queries routed to `ASK_CLARIFICATION` (50.40% of Golden Set; 88.73% of all 142 generated replies used identical static canned templates).
- **Concrete Golden Examples**:
  - `golden_candidate_0191`: Customer stated *"Will not take CC credentials on an iPhone 6s"*; agent replied with canned template: *"Could you let us know your exact device model and iOS version?"*
  - `golden_candidate_0001`: Customer opening sentence: *"When trying to install free apps on an iPhone 6s..."*; agent asked for exact device model already provided.
  - `golden_candidate_0025`: Customer asking about an old work computer / Mac; agent replied asking for their iOS version.
- **Root-Cause Hypothesis**: The offline reply generator uses a single unparameterized clarification template whenever retrieval is weak or intent is borderline. Zero entity/slot extraction is performed on the customer's message prior to formulating the question.

### Failure Mode 4: Link Validity Degradation & Repurposing of Ephemeral Twitter DM Shortlinks
- **Impact Priority**: `HIGH (Grounding Fidelity & Citation Integrity)`
- **Measured Frequency**: 16 out of 16 AUTO_HANDLE replies (100.0%) cited raw Twitter `t.co` shortlinks from historical agent tweets; 7 of 16 cited `https://t.co/GDrqU22YpT` (a Twitter DM invite link). Exactly **zero** AUTO_HANDLE replies cited canonical `support.apple.com` documentation.
- **Concrete Golden Examples**:
  - `golden_candidate_0066`: Customer trying to run update from App Store; agent promised *"You can find the steps to resolve this here"* while linking to a Twitter DM link.
  - `golden_candidate_0039`: Autocorrect bug inquiry paired with historical DM invitation shortlink.
  - `golden_candidate_0193`: Battery drain complaint answered with shortened `t.co` tweet link.
- **Distinction (Measured vs Hypothesis)**:
  - *Measured Fact*: 100% of auto-handle replies use `t.co` links; 43.75% cite a DM link; 0% cite canonical Apple support domains.
  - *Root-Cause Hypothesis*: TF-IDF retrieval matches high-frequency brand tokens (`@AppleSupport`, `iPhone`) rather than technical resolution content. The programmatic URL guard checks verbatim presence in evidence, but does not distinguish canonical documentation from ephemeral invite links.

### Failure Mode 5: Conversational Anaphora & Context Boundary Deficit in Multi-Turn Threads
- **Impact Priority**: `MEDIUM-HIGH (Intent Diagnostic Accuracy)`
- **Measured Frequency**:
  - Exactly 250 of 250 Golden examples (100.0%) belong to conversation threads with length $> 1$ (mean thread length = 4.1 turns).
  - In 96 cases (38.40%), preceding conversational context was available in the thread; 154 cases (61.60%) were Turn 0.
  - Among 122 intent misclassifications, 66 (54.10%) contained anaphoric markers (*"it"*, *"this"*, *"that"*, *"still"*, *"again"*).
  - In 29 of those anaphoric misclassifications (23.77% of all intent errors), prior conversational context was available in the thread that could resolve the referent.
- **Concrete Golden Examples**:
  - `golden_candidate_0017`: *"it asks for a user and a passwort, like a remote login... why did it appear all of a sudden?"*; Turn 1 established unauthorized new user login appeared after Mac update.
  - `golden_candidate_0034`: *"Took maybe an hour to recover all photos... Is there a setting I can turn off to stop it happening"*; prior turn established customer was discussing iCloud photo recovery.
  - `golden_candidate_0059`: *"Updated apps reset device still very glitchy"*; prior turns showed ongoing complaints about UI freezing.
- **Distinction (Measured vs Hypothesis)**:
  - *Measured Fact*: 250/250 threads are multi-turn; 96 have prior context; 29 intent errors combine anaphoric markers with available prior context.
  - *Root-Cause Hypothesis*: Single-turn inference depresses classifier confidence and triggers abstention to `other_unclear`. However, the causal improvement from thread context remains an unmeasured hypothesis until experimentally evaluated.

---

## 8. One-More-Week Plan

*Prioritized engineering roadmap addressing measured failure modes (labeled strictly as future work):*

1. **Negative-Intent Escalation Guards for High-Risk Cases**:
   - Implement regex guards detecting statements of prior failure (*"already tried"*, *"ineligible.*online"*, *"chat.*unavailable"*, *"still restarting"*).
   - Enforce mandatory human escalation on 2FA credential changes and persistent hardware bootloops.
2. **Reduce Excessive Withholding Without Sacrificing Safety**:
   - Permit text-only self-service replies for high-confidence predictions ($\ge 0.80$) with similarity $\ge 0.30$, removing the rigid requirement for an external link.
   - Lower the retrieval similarity threshold from 0.35 to 0.30 specifically for informational intents (`battery_power`, `how_to_information`).
3. **Dynamic Slot-Aware Clarification**:
   - Implement lightweight entity extractors for hardware model (*"iPhone 6s"*, *"Mac"*) and OS version (*"iOS 11"*, *"High Sierra"*).
   - Branch clarification questions based on missing slots: if device model is already present, ask only for OS version or specific error codes.
4. **Canonical Support URL Whitelisting & Dense Retrieval**:
   - Enforce strict domain whitelisting on evidence URLs (`support.apple.com`, `appleid.apple.com`); reject or strip raw `t.co` shortlinks and DM invitations.
   - Supplement lexical TF-IDF retrieval with a dense bi-encoder (e.g., `all-MiniLM-L6-v2`) to capture semantic paraphrasing without keyword overlap.
5. **Experiment with Multi-Turn Conversation History**:
   - Extend `AgentController.process_query` to accept an optional `conversation_history` parameter. Prepend prior customer/agent turns with turn boundary tokens before vectorization.
   - Benchmark intent accuracy lift specifically on the 96 multi-turn Golden examples to evaluate causal impact.

---

## 9. Decision Log

Key architectural and methodological decisions made during development:

1. **Focus Exclusively on AppleSupport**
   - *Why*: Provides the largest volume, richest multi-turn resolutions, and deepest technical taxonomy in the dataset.
   - *Tradeoff*: Models and heuristic rules are tailored to Apple's ecosystem and do not generalize zero-shot to other brands.
2. **Split Data at the Conversation Level (`first_tweet_id`)**
   - *Why*: Prevents cross-turn data leakage between train, development, and evaluation splits.
   - *Tradeoff*: Conversation-level splitting requires multi-turn reconstruction and produces slightly variable sample sizes across splits.
3. **Strict Isolation of the 250-Example Golden Set**
   - *Why*: Guarantees that evaluation reflects true generalization without indirect data contamination.
   - *Tradeoff*: Limits the available training and retrieval pool by 250 high-quality annotated conversations.
4. **Freeze Silver Labeler Rules Before Training**
   - *Why*: Avoids iterative label-hacking and maintains clean provenance between heuristic discovery and supervised learning.
   - *Tradeoff*: Silver training data retains residual heuristic label noise.
5. **TF-IDF + Logistic Regression Baseline Over Deep Neural Models**
   - *Why*: Ensures sub-millisecond CPU inference latency, transparent interpretable feature weights, and zero GPU dependencies.
   - *Tradeoff*: Vulnerable to vocabulary mismatch and unable to capture deep semantic paraphrasing.
6. **Index Retrieval Corpus Strictly from TRAIN Split**
   - *Why*: Eliminates knowledge leakage into the historical retrieval index.
   - *Tradeoff*: Evaluation queries cannot match identical historical answers that originated in DEV, TEST, or Golden splits.
7. **Verification-Grounded Generation Over Open-Ended LLM Drafting**
   - *Why*: Prevents hallucinated troubleshooting steps and ensures all cited URLs exist in verified brand history.
   - *Tradeoff*: Offline replies rely on structured templates and lack conversational fluidity.
8. **Centralized Deterministic Orchestration (`AgentController`)**
   - *Why*: Guarantees that safety, risk, and escalation policies override probabilistic classifier outputs.
   - *Tradeoff*: Introduces multi-stage coordination overhead and heuristic threshold coupling.
9. **Tune Intent Abstention Threshold ($\tau = 0.25$) on DEV Split**
   - *Why*: Optimizes the tradeoff between coverage and precision without overfitting to the evaluation benchmark.
   - *Tradeoff*: Abstained queries drop into `other_unclear`, slightly reducing raw specific-intent accuracy.
10. **Conservative Escalation Policy (Verified Link + 0.35 Similarity)**
    - *Why*: Prioritizes customer safety and brand trust by withholding automated replies when evidence is weak.
    - *Tradeoff*: Causes severe automation withholding (88.57% of automatable queries diverted).
11. **LLM-as-a-Judge Validated Against Independent Human Ratings**
    - *Why*: Establishes multi-dimensional quality auditing while requiring empirical validation against human agreement.
    - *Tradeoff*: LLM judge execution is constrained by API rate limits, requiring explicit tracking of partial evaluation runs.
12. **Freeze Final Candidate Pipeline Before Failure Analysis**
    - *Why*: Prevents post-hoc parameter tweaking and ensures failure analysis reflects true candidate behavior.
    - *Tradeoff*: Known failure modes are documented and analyzed rather than patched late in the cycle.
13. **Zero-Dependency Native Browser Frontend for Support Decision Console**
    - *Why*: Eliminates complex web development server setups and enables direct evaluator inspection via a single Python process.
    - *Tradeoff*: UI is intentionally minimalist and relies on browser-native CSS and vanilla JavaScript.

---

## 10. Reproduce in <15 Minutes

The following instructions allow an evaluator to reproduce all results and launch the interactive Support Decision Console from a clean environment on Windows PowerShell in under 5 minutes without retraining models.

### Step 1: Clone Repository & Set Up Virtual Environment
```powershell
git clone https://github.com/fatimasafwa30/support-intelligence-agent.git
cd support-intelligence-agent

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install runtime and test dependencies (~1-2 minutes)
pip install -r requirements.txt
```

### Step 2: Verify Precomputed Frozen Artifacts
The repository includes all frozen models and evaluation benchmarks:
- `data/processed/models/intent_classifier_baseline.joblib` (frozen TF-IDF + Logistic Regression model)
- `data/processed/models/tfidf_retriever.joblib` (frozen historical resolution index)
- `data/golden/golden_annotation.csv` (250-example human-annotated Golden Set)

### Step 3: Run Deterministic Golden Set Evaluation (~25 seconds)
Execute the complete end-to-end agent pipeline across all 250 Golden Set queries:
```powershell
.venv\Scripts\python.exe scripts\evaluate_agent_golden.py
```
*Output*: Generates `reports/agent_golden_evaluation.md`, `reports/agent_evaluation_summary.json`, and `reports/agent_misclassifications.csv`.

### Step 4: Run Machine-Readable Failure Analysis (~15 seconds)
Audit the five failure modes and generate failure analysis artifacts:
```powershell
.venv\Scripts\python.exe scripts\analyze_failures.py
```
*Output*: Generates `reports/failure_analysis.md` and `reports/failure_analysis.json`.

### Step 5: Run Full Test Suite (~30 seconds)
Verify all 245 unit, integration, and contract tests:
```powershell
.venv\Scripts\python.exe -m unittest discover tests
```

### Step 6: Build Frontend & Launch Support Decision Console (<1 minute)
```powershell
cd frontend
npm ci --ignore-scripts --no-audit --no-fund
npm run build
cd ..

# Launch single-process loopback console
.venv\Scripts\python.exe scripts\serve_console.py
```
Open **http://127.0.0.1:8765** in any modern web browser.

---

## 11. Repository Structure

```
support-intelligence-agent/
├── configs/                        # YAML configurations for agent, evaluation, and models
│   ├── agent_controller.yaml       # Thresholds, timeouts, and retrieval retry limits
│   ├── agent_evaluation.yaml       # Evaluation benchmark parameters and baseline configs
│   └── reply_evaluation.yaml       # Rubric dimensions and LLM judge configurations
├── data/                           # Data storage (split by isolation tier)
│   ├── golden/                     # 250-example human-annotated benchmark
│   │   └── golden_annotation.csv
│   └── processed/models/           # Precomputed frozen models and indices
│       ├── intent_classifier_baseline.joblib
│       └── tfidf_retriever.joblib
├── docs/                           # Extended technical specifications and rubrics
│   ├── human_agreement_rating.md   # Human rating protocol and CLI instructions
│   └── support_decision_console.md # Console architectural contract and API spec
├── frontend/                       # Accessible single-workspace browser console
│   ├── dist/                       # Production-built browser-native assets
│   ├── src/                        # Vanilla JavaScript state and DOM controllers
│   └── package.json                # Zero third-party runtime dependencies
├── reports/                        # Authoritative evaluation and failure reports
│   ├── agent_evaluation_summary.json # Machine-readable Golden evaluation metrics
│   ├── agent_golden_evaluation.md  # Detailed markdown evaluation report
│   ├── failure_analysis.json       # Structured 5 failure modes with real examples
│   ├── failure_analysis.md         # Comprehensive failure analysis report
│   └── human_agreement_ratings.jsonl # Collected human rating records
├── scripts/                        # Deterministic CLI runners and evaluation harnesses
│   ├── analyze_failures.py         # Failure analysis extractor and validator
│   ├── evaluate_agent_golden.py    # Main Golden Set evaluation runner
│   ├── rate_human_agreement.py     # Interactive terminal rating harness
│   └── serve_console.py            # Loopback HTTP server for Support Decision Console
├── src/                            # Modular Python source packages
│   ├── agent/                      # Orchestration, state machine, and escalation policy
│   │   ├── agent_state.py          # State tracking and trace serialization
│   │   ├── controller.py           # Core AgentController pipeline
│   │   ├── escalation_policy.py    # Deterministic risk and queue routing rules
│   │   ├── grounded_generator.py   # Evidence-grounded template synthesis
│   │   └── grounding_guard.py      # Citation and URL verification guards
│   ├── console/                    # Local HTTP adapter and validation schemas
│   ├── evaluation/                 # Evaluator metrics and LLM judge harnesses
│   ├── intents/                    # TF-IDF vectorization and Logistic Regression
│   └── retrieval/                  # Historical resolution corpus and TF-IDF search
├── tests/                          # 245 automated tests (unit, regression, contract)
├── README.md                       # Comprehensive system and evaluation report
└── requirements.txt                # Pinned Python dependencies
```

---

## 12. Limitations

A candid summary of system boundaries and operational constraints:

1. **Historical 2017 Twitter Dataset**: The underlying data reflects 2017 customer behavior (iPhone 6s/7, iOS 11), Twitter's character limitations, and informal colloquial grammar. Contemporary multimodal support (screenshots, video) and current iOS 17/18 workflows are not represented.
2. **Single-Brand Domain Scope**: Intent categories, vocabulary, and heuristic rules are tailored specifically to `@AppleSupport`. Transferring the system to banking, telecom, or retail requires building new intent taxonomies and indexing domain-specific historical corpora.
3. **Silver Training Label Noise**: The training corpus was labeled using rule-based keyword matching and pattern heuristics. While high-volume, silver labels introduce label noise that limits supervised classifier precision.
4. **Lexical Retrieval Limitations**: The TF-IDF retriever relies on exact word and n-gram overlap. It struggles with conceptual paraphrasing, novel technical phrasing, and colloquial synonyms that lack lexical overlap with historical resolutions.
5. **Static Offline Reply Generation**: Grounded replies rely on deterministic templated drafting rather than dynamic neural text generation. While preventing hallucinations, replies can feel rigid and lack fluid conversational personalization.
6. **Partial LLM-as-a-Judge Evaluation**: Due to external API rate limits, LLM-as-a-judge scoring is complete for an interim cohort of 39 generated replies. Inter-rater agreement on the clean 9-example intersection with independent human ratings is deferred pending full cohort adjudication.
7. **Single-Turn Live Controller Inference**: While conversations are reconstructed into multi-turn threads in the dataset, the live `AgentController` currently evaluates each inbound query as a single turn, discarding antecedent conversational context.
8. **Golden Sample Size Constraints**: With $N=250$, the Golden Set provides a solid evaluation benchmark, but contains only 10 critical safety cases. Rare catastrophic events carry wide statistical confidence intervals.

---

## 13. Live Demo Examples

Three representative examples evaluated on the frozen system demonstrating distinct operational paths:

### 1. Safe Escalation: Hardware Repair Inquiry (`golden_candidate_0010`)
- **Customer Message**:
  `"@AppleSupport Will Apple reject to fix my iPhone under warranty if it has minor dents on it? https://t.co/icS6yynki7"`
- **Gold Labels**: Intent: `repair_service` | Risk: `high` | Action: `ESCALATE`
- **Agent Output**:
  - **Predicted Intent**: `repair_service` (Confidence: `0.784`)
  - **Final Action**: `ESCALATE`
  - **Target Queue**: `hardware_repair`
  - **Decision Reasons**: `repair_service_mandates_human_booking`, `escalation_engine_pre_generation_short_circuit`
  - **Generated Text**: None (Pre-generation short-circuit prevents ungrounded bot replies; ticket routed directly to authorized repair personnel).

### 2. Successful Auto-Handle: Account Administration (`golden_candidate_0029`)
- **Customer Message**:
  `"@AppleSupport is it possible to merge accounts? I have a new Apple ID on a recent iPhone and utilizing an older ID on iPad...1/2"`
- **Gold Labels**: Intent: `apple_id_account` | Risk: `medium` | Action: `AUTO_HANDLE`
- **Agent Output**:
  - **Predicted Intent**: `apple_id_account` (Confidence: `0.804`)
  - **Retrieval Evidence**: Similarity: `0.378` | Evidence URL: `https://t.co/SDIe7UiyJN`
  - **Final Action**: `AUTO_HANDLE`
  - **Target Queue**: `self_service`
  - **Generated Reply**:
    *"We'd be glad to help with this. You can find the steps to resolve this here: https://t.co/SDIe7UiyJN . Let us know if you need any further assistance!"*
  - **Verification**: Cited URL was verified against historical AppleSupport resolution records and passed the programmatic citation guard.

### 3. Diagnostic Clarification: Vague Reboot Freeze (`golden_candidate_0007`)
- **Customer Message**:
  `"welp, @115858 iPhone 7 has officially screwed me over. trying to do factory reset and it won't let me get thru to end. every time i try getting to finish the factory reset, it will power itself down and restart mode. i just need a new iPhone, this one obvi doesn't work. plz help"`
- **Gold Labels**: Intent: `device_hardware` | Risk: `high` | Action: `ESCALATE`
- **Agent Output**:
  - **Predicted Intent**: `other_unclear` (Confidence: `0.229` $\to$ Abstained below 0.25 threshold)
  - **Retrieval Evidence**: Similarity: `0.240` (Below 0.35 threshold)
  - **Final Action**: `ASK_CLARIFICATION`
  - **Target Queue**: `general_support`
  - **Decision Reasons**: `unclear_intent_requires_resolution`, `intent_confidence_below_threshold (0.229 < 0.5)`, `retrieval_similarity_below_threshold (0.240 < 0.35)`, `vague_inquiry_requested_clarification`
  - **Generated Reply**:
    *"We'd like to look into this with you. Could you let us know your exact device model and iOS version? You can also meet us in DM with additional details so we can help."*

---

## 14. Technologies Used

### Backend & Machine Learning
- **Python 3.11+**: Core programming language.
- **Scikit-learn**: TF-IDF feature extraction, Logistic Regression classification, and cosine similarity metric calculations.
- **Joblib**: Efficient serialization of frozen vectorizers and classifiers.
- **Pandas & NumPy**: Data processing, stratified sampling, and metric computations.
- **PyYAML**: Strict schema-validated configuration loading.

### LLM Evaluation & Adjudication
- **Google GenAI SDK (`google-genai`)**: LLM-as-a-judge scoring via `gemini-2.5-flash` with structured Pydantic output schemas.
- **Pydantic v2**: Strict schema definition and validation for judge records and evaluation outputs.

### Frontend & Console
- **HTML5 & Vanilla CSS**: Modern, responsive, accessible interface with zero CSS framework bloat.
- **Vanilla JavaScript (ES6+)**: Browser-native event handling, state coordination, and API communication.
- **Node.js (Built-in Test Runner)**: Frontend syntax checking and test automation with zero external npm dependencies.
- **Python `http.server`**: Local single-process HTTP loopback server routing both API requests and static UI assets.
