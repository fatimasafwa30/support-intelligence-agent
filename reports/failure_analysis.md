# Failure Analysis Report: Final-Candidate Golden Evaluation

## 1. Executive Summary & Verified Frozen Benchmarks

This report provides a comprehensive failure analysis over the frozen final-candidate Golden Set evaluation (250 examples). All calculations were executed in deterministic, read-only offline mode without mutating any model weights, thresholds, or evaluation artifacts.

### Key Benchmark Invariants Verification Table
| Metric | Measured Golden Value | Benchmark Reference | Provenance / Verification Status |
| :--- | :---: | :---: | :--- |
| **Evaluated Golden Sample ($N$)** | **250** | 250 | Verified; matches `golden_annotation.csv` |
| **Operational 3-Way Routing** | **AUTO: 16**, **ASK: 126**, **ESC: 108** | 16 / 126 / 108 | Verified; matches Milestone 18 distribution |
| **Secondary Binary Action Accuracy** | **72.00%** | 72.00% | Verified; 180 of 250 binary matches |
| **Under-Escalation Rate (Dangerous FN)** | **4.44%** (8/180) | 4.44% (8/180) | Verified; exactly 8 Gold ESCALATE routed to AUTO |
| **False Escalation Rate on Auto Candidates** | **88.57%** (62/70) | 88.57% (62/70) | Verified; 62 Gold AUTO withheld from automation |
| **Critical Safety Recall** | **100.00%** (10/10) | 100.00% (10/10) | Verified; zero critical safety hazards missed |
| **Full Golden Intent Accuracy** | **51.20%** | 51.20% (128/250) | Verified; includes 25 Gold other_unclear queries |
| **Specific-Intent Accuracy (Post-Abstention)** | **48.00%** | 48.00% (108/225) | Verified; 67 specific-intent cases abstained to other_unclear |
| **Generated Reply Population** | **142** | 142 | Verified; 16 AUTO_HANDLE + 126 ASK_CLARIFICATION |
| **Pre-Generation Short-Circuits** | **108** | 108 | Verified; 7 Pre-Retrieval Critical + 101 Pre-Gen Policy |

> **Provenance & Accounting Clarifications**:
> 1. **Pre-Generation Short-Circuits Accounting**: Exactly 108 of 250 queries bypass text generation ($250 - 142 = 108$). This is composed of **7 pre-retrieval critical hazard short-circuits** plus **101 escalation-engine pre-generation boundary short-circuits**.
> 2. **Intent Accuracy Baseline Comparison**: In Milestone 14/16, the standalone intent classifier scored **56.44%** on specific intents without abstention. In the orchestrator, introducing the 0.25 confidence threshold abstains 67 low-confidence queries, yielding **48.00%** on specific intents. However, on the full 250 set, correctly abstaining 20 of the 25 Gold `other_unclear` cases lifts full accuracy to **51.20%**.
> 3. **Reply Quality Population**: In `reports/reply_quality_judge_report.md` Section 4, a figure of 135 generated replies was cited from an earlier Milestone 19.1 snapshot; the frozen Milestone 18 AgentController evaluation produces exactly **142 generated replies**.

---

## 2. Exactly Five Evaluator-Facing Failure Modes

The failure modes below are prioritized by **practical customer support and safety impact**, rather than raw frequency.

### Failure Mode 1: Under-Escalation on High-Risk / Complex Technical Requests (Dangerous False Self-Service)
- **Impact Priority**: `CRITICAL (Safety, Security & Brand Trust)`
- **Measured Frequency**: {"count": 8, "total_gold_escalate": 180, "under_escalation_rate": 0.0444}

#### Concrete Real Golden Examples

**Example `golden_candidate_0035`**:
- **Customer Message**: *"@AppleSupport I've tried since I placed my order - it says "Your order may be ineligible for changes online" and your chat line is unavailable."*
- **Gold Labels**: Intent=`orders_delivery`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`orders_delivery` (conf=0.909), Action=`AUTO_HANDLE`
- **Retrieval Evidence**: Similarity=`0.378`, URL=`https://t.co/8yjRd1Xo0i`
- **Generated Reply**: *"We'd be glad to help with this. You can find the steps to resolve this here: https://t.co/8yjRd1Xo0i . Let us know if you need any further assistance!"*
- **Human Annotation Notes**: Urgent request to modify order shipping address while online change portal and chat are unavailable; requires live human support.

**Example `golden_candidate_0132`**:
- **Customer Message**: *"@AppleSupport Your two factor authentication is awful! It makes using the phone very unpleasant. And turning it off is impossible!"*
- **Gold Labels**: Intent=`security_privacy`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`apple_id_account` (conf=0.636), Action=`AUTO_HANDLE`
- **Retrieval Evidence**: Similarity=`0.351`, URL=`https://t.co/GDrqU22YpT`
- **Generated Reply**: *"We'd be glad to help with this. You can find the steps to resolve this here: https://t.co/GDrqU22YpT . Let us know if you need any further assistance!"*
- **Human Annotation Notes**: Customer complaining about Two-Factor Authentication and asking to disable it; account security modifications require escalation.

**Example `golden_candidate_0062`**:
- **Customer Message**: *"@AppleSupport I’ve downloaded the suggested software update and my phone is still restarting every 30 seconds"*
- **Gold Labels**: Intent=`device_hardware`, Action=`ESCALATE`, Risk=`medium`
- **Model Output**: Predicted Intent=`software_update` (conf=0.863), Action=`AUTO_HANDLE`
- **Retrieval Evidence**: Similarity=`0.351`, URL=`https://t.co/GDrqU22YpT`
- **Generated Reply**: *"We'd be glad to help with this. You can find the steps to resolve this here: https://t.co/GDrqU22YpT . Let us know if you need any further assistance!"*
- **Human Annotation Notes**: Repeated rebooting every 30 seconds persisting despite iOS 11.2 update; hardware/crash investigation required.

#### Root-Cause Hypothesis
The escalation policy permits AUTO_HANDLE whenever intent confidence >= 0.50, retrieval similarity >= 0.35, and an extracted evidence URL exists, unless an explicit critical-hazard keyword fires. The system lacks semantic guards for: (a) explicit customer statements that self-service or online tools have already failed, (b) sensitive account security boundaries (e.g. 2FA disabling), and (c) recurring hardware reboot loops.

#### Concrete One-More-Week Mitigation
1. Implement a negative-intent escalation guard regex detecting statements of prior failure (e.g., 'already tried', 'ineligible.*online', 'chat.*unavailable', 'still restarting').
2. Classify 2FA/credential alteration requests as high-risk security boundaries requiring human escalation.
3. Require customer confirmation before auto-resolving hardware issues that mention recurring bootloops.

---

### Failure Mode 2: Over-Conservative Withholding & False Escalation of Automatable Self-Service Queries
- **Impact Priority**: `HIGH (Operational Efficiency & Automation Deficit)`
- **Measured Frequency**: {"count": 62, "total_gold_auto": 70, "false_escalation_rate": 0.8857, "diverted_to_clarification": 38, "diverted_to_escalate": 24}

#### Concrete Real Golden Examples

**Example `golden_candidate_0006`**:
- **Customer Message**: *"@AppleSupport Hello! How to change my personal name associated with my apple ID? on the community FAQ sections, it always direct me to how to change apple ID name instead."*
- **Gold Labels**: Intent=`apple_id_account`, Action=`AUTO_HANDLE`, Risk=`high`
- **Model Output**: Predicted Intent=`apple_id_account` (conf=0.968), Action=`ESCALATE`
- **Retrieval Evidence**: Similarity=`0.384`, URL=`None`
- **Decision Reasons**: `lacks_verified_self_service_link_escalated_to_agent; escalation_engine_pre_generation_short_circuit`
- **Human Annotation Notes**: Customer asking for self-service instructions to change personal name associated with Apple ID; appropriate for automated guidance.

**Example `golden_candidate_0015`**:
- **Customer Message**: *"@AppleSupport Hello
I have a question whether the intermittent charging is better for the battery better than the full charge"*
- **Gold Labels**: Intent=`battery_power`, Action=`AUTO_HANDLE`, Risk=`high`
- **Model Output**: Predicted Intent=`battery_power` (conf=0.795), Action=`ESCALATE`
- **Retrieval Evidence**: Similarity=`0.271`, URL=`None`
- **Decision Reasons**: `retrieval_similarity_below_threshold (0.271 < 0.35); uncertain_technical_inquiry_escalated`
- **Human Annotation Notes**: Informational inquiry regarding battery charging habits and device health; suitable for automated knowledge-grounded response.

**Example `golden_candidate_0026`**:
- **Customer Message**: *"@AppleSupport Since last night - no, it says 'searching' and occasionally says 'no service'"*
- **Gold Labels**: Intent=`connectivity`, Action=`AUTO_HANDLE`, Risk=`high`
- **Model Output**: Predicted Intent=`connectivity` (conf=0.655), Action=`ESCALATE`
- **Retrieval Evidence**: Similarity=`0.33`, URL=`None`
- **Decision Reasons**: `retrieval_similarity_below_threshold (0.330 < 0.35); uncertain_technical_inquiry_escalated`
- **Human Annotation Notes**: iPhone displaying 'searching' / 'no service' cellular connection issue; standard self-service troubleshooting applies.

#### Root-Cause Hypothesis
Dual rigid gates suppress self-service resolution: (1) requiring an extracted URL link in evidence forces escalation even when confidence is near-certain (e.g. 0.968 in candidate 0006) and text steps exist; (2) rigid 0.35 TF-IDF cosine similarity cutoff rejects conceptual or conversational queries that lack exact keyword overlap (e.g. battery charging best practices at 0.271).

#### Concrete One-More-Week Mitigation
1. Permit grounded text-only self-service replies for high-confidence predictions (>= 0.80) when evidence similarity is >= 0.30, even without an extracted link.
2. Lower the similarity gating threshold from 0.35 to 0.30 for top informational intents (battery_power, how_to_information).
3. Implement semantic dense bi-encoder embeddings (e.g. all-MiniLM-L6-v2) to capture paraphrased technical questions.

---

### Failure Mode 3: Monolithic Clarification Loop & Premature Triage Trap (Static Canned Responses)
- **Impact Priority**: `HIGH (Customer Experience & Dialogue Friction)`
- **Measured Frequency**: {"clarification_actions_count": 126, "clarification_share_of_golden": 0.504, "clarification_share_of_generated": 0.8873, "unique_templates_used": 9, "static_canned_reply_count": 126}

#### Concrete Real Golden Examples

**Example `golden_candidate_0191`**:
- **Customer Message**: *"@AppleSupport Yes we are but this is not working. Will not take husbands CC credentials on an iPhone 6s. Now none of us can update even free apps"*
- **Gold Labels**: Intent=`billing_payments`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`app_store` (conf=0.269), Action=`ASK_CLARIFICATION`
- **Generated Reply**: *"We'd like to look into this with you. Could you let us know your exact device model and iOS version? You can also meet us in DM with additional details so we can help."*
- **Observed Defect**: Customer explicitly stated 'on an iPhone 6s'; the agent blindly asks for their exact device model.

**Example `golden_candidate_0001`**:
- **Customer Message**: *"@AppleSupport When trying to install free apps on an iPhone 6s I had to finish the setup of the Apple ID and enter payment info which I can't do without a CC"*
- **Gold Labels**: Intent=`app_store`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`apple_id_account` (conf=0.316), Action=`ASK_CLARIFICATION`
- **Generated Reply**: *"We'd like to look into this with you. Could you let us know your exact device model and iOS version? You can also meet us in DM with additional details so we can help."*
- **Observed Defect**: Customer explicitly stated 'on an iPhone 6s'; the template requests device information already provided in the opening sentence.

**Example `golden_candidate_0025`**:
- **Customer Message**: *"@AppleSupport Hi, guys. It’s me again. I bought my old work computer. It was wiped. It turns on but gives me a password hint that doesn’t work, so I can’t get it open. Ideas? Thanks!"*
- **Gold Labels**: Intent=`apple_id_account`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`other_unclear` (conf=0.232), Action=`ASK_CLARIFICATION`
- **Generated Reply**: *"We'd like to look into this with you. Could you let us know your exact device model and iOS version? You can also meet us in DM with additional details so we can help."*
- **Observed Defect**: Inquiry is regarding a work computer / Mac; the canned mobile triage prompt inappropriately asks for an iOS version.

#### Root-Cause Hypothesis
The offline reply generator uses a single unparameterized CLARIFICATION_TEMPLATE whenever retrieval_status is weak or confidence is low. There is zero entity/slot extraction for device type or operating system before formulating the clarification request.

#### Concrete One-More-Week Mitigation
1. Implement regex entity extractors for hardware ('iPhone 6s', 'Mac', 'iPad', 'Apple Watch') and OS ('High Sierra', 'iOS 11').
2. Condition clarification prompts on missing slots: if device model is present, ask only for OS version or specific error codes.
3. Branch clarification templates by platform: ask for macOS version on Mac queries rather than iOS version.

---

### Failure Mode 4: Link Validity Degradation & Repurposing of Ephemeral Twitter DM Shortlinks
- **Impact Priority**: `HIGH (Grounding Fidelity & Citation Integrity)`
- **Measured Frequency**: {"auto_handle_replies_count": 16, "auto_replies_using_t_co_links": 16, "auto_replies_t_co_share": 1.0, "initial_borderline_or_insufficient_retrieval_count": 139, "retry_borderline_upgrade_rate": 0.0388}

#### Concrete Real Golden Examples

**Example `golden_candidate_0066`**:
- **Customer Message**: *"@AppleSupport I was trying to run the update from the App Store"*
- **Gold Labels**: Intent=`software_update`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`app_store` (conf=0.979), Action=`AUTO_HANDLE`
- **Retrieval Evidence**: Similarity=`0.389`, URL=`https://t.co/GDrqU22YpT`
- **Generated Reply**: *"We'd be glad to help with this. You can find the steps to resolve this here: https://t.co/GDrqU22YpT . Let us know if you need any further assistance!"*
- **Observed Defect**: Reply promises 'steps to resolve this here', but links to a Twitter DM invitation link.

**Example `golden_candidate_0039`**:
- **Customer Message**: *"Dear @AppleSupport - on my iPhone "is" keeps autocorrecting to I.S - I've factory reset the phone and it didn't help. Is this an iOS 11 bug?"*
- **Gold Labels**: Intent=`device_hardware`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`device_hardware` (conf=0.561), Action=`AUTO_HANDLE`
- **Retrieval Evidence**: Similarity=`0.361`, URL=`https://t.co/GDrqU22YpT`
- **Generated Reply**: *"We'd be glad to help with this. You can find the steps to resolve this here: https://t.co/GDrqU22YpT . Let us know if you need any further assistance!"*
- **Observed Defect**: Lexical keyword hit on 'iPhone' and 'iOS' pairs autocorrect bug with historical DM invitation.

**Example `golden_candidate_0193`**:
- **Customer Message**: *"The battery drain is real @115858 #ios11 #neverfails"*
- **Gold Labels**: Intent=`battery_power`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`battery_power` (conf=0.996), Action=`AUTO_HANDLE`
- **Retrieval Evidence**: Similarity=`0.478`, URL=`https://t.co/bivpdfBNJ6`
- **Generated Reply**: *"We'd be glad to help with this. You can find the steps to resolve this here: https://t.co/bivpdfBNJ6 . Let us know if you need any further assistance!"*
- **Observed Defect**: Shortened t.co URL from historical tweet used as autonomous self-service guidance.

#### Measured Facts vs. Hypothesis
- **MEASURED FACTS**:
16 of 16 AUTO_HANDLE replies (100.0%) cite raw Twitter 't.co' shortlinks extracted from historical agent tweets. In 7 of those 16 cases, the exact link cited is 'https://t.co/GDrqU22YpT' (a Twitter DM invite link). Zero AUTO_HANDLE replies cite canonical 'support.apple.com' documentation.

- **ROOT-CAUSE HYPOTHESIS**:
TF-IDF lexical retrieval matches high-frequency brand tokens (e.g. '@AppleSupport', 'iPhone', 'iOS 11') rather than exact troubleshooting content. The programmatic URL verification guard checks that the URL exists verbatim in retrieved evidence, but fails to distinguish canonical documentation URLs from ephemeral conversational shortlinks.

#### Concrete One-More-Week Mitigation
1. Enforce strict domain whitelisting on evidence URLs (permitting only support.apple.com, appleid.apple.com, getsupport.apple.com); strip or reject t.co shortlinks.
2. When evidence contains only a DM link, route the case to ASK_CLARIFICATION or agent escalation instead of claiming steps exist.
3. Enhance retrieval index by indexing curated Apple Knowledge Base articles rather than raw historical tweet replies.

---

### Failure Mode 5: Conversational Anaphora & Context Boundary Deficit in Multi-Turn Threads
- **Impact Priority**: `MEDIUM-HIGH (Intent Diagnostic Accuracy & Contextual Grounding)`
- **Measured Frequency**: {"total_golden_examples": 250, "examples_in_threads_gt_1_turn": 250, "examples_in_threads_gt_1_share": 1.0, "examples_with_prior_context_available": 96, "examples_with_prior_context_share": 0.384, "examples_as_first_turn_no_prior_context": 154, "examples_as_first_turn_share": 0.616, "total_intent_misclassifications": 122, "misclassifications_with_anaphoric_markers": 66, "misclassifications_anaphora_with_prior_context": 29, "misclassifications_anaphora_at_turn_0": 37}

#### Concrete Real Golden Examples

**Example `golden_candidate_0017`**:
- **Customer Message**: *"@AppleSupport yeah i know, it asks for a user and a passwort, like a remote login or something, but why did it appear all of a sudden?
because i used ssh once? :D"*
- **Gold Labels**: Intent=`security_privacy`, Action=`ESCALATE`, Risk=`critical`
- **Model Output**: Predicted Intent=`other_unclear` (conf=0.127), Action=`ASK_CLARIFICATION`
- **Preceding Thread Context**: *"[customer] my mac just pulled a microsoft and updated straight after power on, now i have this new user on my laptop, is this normal or what is going on? D: https://t.co/LqlkV1Btzg"*
- **Contextual Analysis**: Single-turn message has pronoun 'it' and colloquial spelling ('passwort'); antecedent context in turn 1 clearly establishes unauthorized new user login screen after update.

**Example `golden_candidate_0034`**:
- **Customer Message**: *"@AppleSupport 11.0.3. Took maybe an hour to recover all photos. Thought it might have been quicker. Is there a setting I can turn off to stop it happening"*
- **Gold Labels**: Intent=`icloud`, Action=`AUTO_HANDLE`, Risk=`high`
- **Model Output**: Predicted Intent=`how_to_information` (conf=0.317), Action=`ASK_CLARIFICATION`
- **Preceding Thread Context**: *"[customer] @AppleSupport ios11 not letting me recover photos to be deleted. How do I solve this? [AppleSupport] @247114 What version of iOS 11 are you on?"*
- **Contextual Analysis**: Message contains 'it might have been quicker' and 'stop it happening'; antecedent turn establishes that 'it' refers to iCloud photo recovery.

**Example `golden_candidate_0059`**:
- **Customer Message**: *"@AppleSupport Updated apps reset device still very glitchy"*
- **Gold Labels**: Intent=`device_hardware`, Action=`ESCALATE`, Risk=`high`
- **Model Output**: Predicted Intent=`other_unclear` (conf=0.223), Action=`ASK_CLARIFICATION`
- **Preceding Thread Context**: *"[customer] @AppleSupport Please get back to your roots and make the iPhone reliable. I’m ready to ditch it! [AppleSupport] What kind of issues are you having?"*
- **Contextual Analysis**: Message uses 'still very glitchy'; antecedent turns show ongoing complaints about persistent hardware/UI freezing.

#### Measured Facts vs. Hypothesis
- **MEASURED FACTS**:
1. Thread Breadth: Exactly 250 of 250 Golden examples (100.0%) belong to conversation threads with length > 1 (mean thread length = 4.1 turns).
2. Context Availability: In 96 of 250 cases (38.40%), the evaluated customer message is a subsequent turn with preceding conversational context available in the thread. In the remaining 154 cases (61.60%), the evaluated message is Turn 0 (first inbound tweet, no prior context).
3. Anaphoric Density: Among 122 intent misclassifications, exactly 66 (54.10%) contain anaphoric markers ('it', 'this', 'that', 'still', 'again').
4. Resolvable Subset: Of those 66 anaphoric misclassifications, exactly 29 cases (23.77% of all intent errors) had prior conversational context available in the thread, while 37 occurred at Turn 0 without prior Twitter thread history.

- **ROOT-CAUSE HYPOTHESIS**:
Single-turn inference isolates queries from conversational referents, which plausibly depresses classifier confidence and triggers abstention to other_unclear on multi-turn replies. However, causal improvement from thread context remains an unmeasured hypothesis until experimentally evaluated on this subset.

#### Concrete One-More-Week Mitigation
1. Extend AgentController.process_query to accept an optional conversation_history parameter.
2. When preceding turns are present, prepend prior customer/agent turns with turn boundary tokens before TF-IDF vectorization.
3. Experimentally benchmark intent classification accuracy with and without multi-turn thread history on the 96 multi-turn Golden examples.

---

## 3. Machine-Generated Failure Summaries

### 3.1 Top Intent Confusion Pairs (Specific-Intent Benchmark)
| Rank | Gold Intent (Human) | Predicted Intent (ML) | Mismatch Count | Primary Confusion Driver |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `device_hardware` | `other_unclear` | 18 | False abstention on low confidence |
| 2 | `software_update` | `other_unclear` | 17 | False abstention on low confidence |
| 3 | `device_hardware` | `software_update` | 6 | Lexical term overlap |
| 4 | `battery_power` | `other_unclear` | 6 | False abstention on low confidence |
| 5 | `subscriptions_media` | `other_unclear` | 6 | False abstention on low confidence |
| 6 | `repair_service` | `other_unclear` | 5 | False abstention on low confidence |
| 7 | `software_update` | `device_hardware` | 5 | Lexical term overlap |
| 8 | `connectivity` | `other_unclear` | 4 | False abstention on low confidence |
| 9 | `apple_id_account` | `icloud` | 3 | Lexical term overlap |
| 10 | `security_privacy` | `apple_id_account` | 3 | Lexical term overlap |
| 11 | `device_hardware` | `connectivity` | 3 | Lexical term overlap |
| 12 | `app_store` | `apple_id_account` | 2 | Lexical term overlap |
| 13 | `connectivity` | `software_update` | 2 | Lexical term overlap |
| 14 | `apple_id_account` | `other_unclear` | 2 | False abstention on low confidence |
| 15 | `repair_service` | `device_hardware` | 2 | Lexical term overlap |

### 3.2 Full Audit of Under-Escalation Cases (8 Cases)
| Example ID | Customer Query Excerpt | Gold Intent | Predicted Intent | Top Sim | Retr URL | Decision Failure |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `golden_candidate_0035` | @AppleSupport I've tried since I placed my order - it says "... | `orders_delivery` | `orders_delivery` | 0.378 | Yes (t.co) | Missed boundary in high-confidence inquiry |
| `golden_candidate_0039` | Dear @AppleSupport - on my iPhone "is" keeps autocorrecting ... | `device_hardware` | `device_hardware` | 0.361 | Yes (t.co) | Missed boundary in high-confidence inquiry |
| `golden_candidate_0062` | @AppleSupport I’ve downloaded the suggested software update ... | `device_hardware` | `software_update` | 0.351 | Yes (t.co) | Missed boundary in high-confidence inquiry |
| `golden_candidate_0066` | @AppleSupport I was trying to run the update from the App St... | `software_update` | `app_store` | 0.389 | Yes (t.co) | Missed boundary in high-confidence inquiry |
| `golden_candidate_0132` | @AppleSupport Your two factor authentication is awful! It ma... | `security_privacy` | `apple_id_account` | 0.351 | Yes (t.co) | Missed boundary in high-confidence inquiry |
| `golden_candidate_0193` | The battery drain is real @115858 #ios11 #neverfails... | `battery_power` | `battery_power` | 0.478 | Yes (t.co) | Missed boundary in high-confidence inquiry |
| `golden_candidate_0208` | @116333 plz do something for iphone 7plus battery. Its getti... | `battery_power` | `battery_power` | 0.474 | Yes (t.co) | Missed boundary in high-confidence inquiry |
| `golden_candidate_0231` | @115858 thanks for all the updates, bit@can we do something ... | `battery_power` | `battery_power` | 0.383 | Yes (t.co) | Missed boundary in high-confidence inquiry |

### 3.3 Sample False-Escalation & Withholding Cases (62 Total)
| Example ID | Customer Query | Gold Intent | Pred Action | Pred Intent | Sim | Triggered Short-Circuit / Withholding Reason |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| `golden_candidate_0006` | @AppleSupport Hello! How to change my personal nam... | `apple_id_account` | `ESCALATE` | `apple_id_account` | 0.384 | `lacks_verified_self_service_link_escalated_to_agent; escalation_e...` |
| `golden_candidate_0015` | @AppleSupport Hello I have a question whether the ... | `battery_power` | `ESCALATE` | `battery_power` | 0.271 | `retrieval_similarity_below_threshold (0.271 < 0.35); uncertain_te...` |
| `golden_candidate_0024` | @AppleSupport who's bright fucking idea was it to ... | `apple_id_account` | `ESCALATE` | `apple_id_account` | 0.285 | `retrieval_similarity_below_threshold (0.285 < 0.35); uncertain_te...` |
| `golden_candidate_0026` | @AppleSupport Since last night - no, it says “sear... | `connectivity` | `ESCALATE` | `connectivity` | 0.330 | `retrieval_similarity_below_threshold (0.330 < 0.35); uncertain_te...` |
| `golden_candidate_0034` | @AppleSupport 11.0.3. Took maybe an hour to recove... | `icloud` | `ASK_CLARIFICATION` | `how_to_information` | 0.223 | `intent_confidence_below_threshold (0.317 < 0.5); retrieval_simila...` |
| `golden_candidate_0037` | @AppleSupport I turned it off through control cent... | `connectivity` | `ASK_CLARIFICATION` | `connectivity` | 0.303 | `intent_confidence_below_threshold (0.389 < 0.5); retrieval_simila...` |
| `golden_candidate_0040` | @AppleSupport do u send email like this one? PS, t... | `security_privacy` | `ESCALATE` | `apple_id_account` | 0.411 | `lacks_verified_self_service_link_escalated_to_agent; escalation_e...` |
| `golden_candidate_0042` | Hey @115858 - how about adding a feature to the #H... | `how_to_information` | `ESCALATE` | `how_to_information` | 0.256 | `retrieval_similarity_below_threshold (0.256 < 0.35); uncertain_te...` |

### 3.4 Lowest Retrieval Quality Cases
| Example ID | Customer Query | Gold Intent | Pred Intent | Final Sim | Sufficiency | Resulting Action |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `golden_candidate_0020` | @115858 says my iPhone 7 Plus has not been hacked, but ... | `security_privacy` | `security_privacy` | 0.000 | `unknown` | `ESCALATE` |
| `golden_candidate_0057` | @115858 I️ think you guys have been hacked I’ve NEVER h... | `security_privacy` | `security_privacy` | 0.000 | `unknown` | `ESCALATE` |
| `golden_candidate_0104` | @ATT @AppleSupport @115858 https://t.co/e3gs0IMTUR... | `orders_delivery` | `other_unclear` | 0.000 | `empty` | `ASK_CLARIFICATION` |
| `golden_candidate_0109` | @AppleSupport @67109 @505673... | `other_unclear` | `other_unclear` | 0.000 | `empty` | `ASK_CLARIFICATION` |
| `golden_candidate_0111` | @AppleSupport... | `other_unclear` | `other_unclear` | 0.000 | `empty` | `ASK_CLARIFICATION` |
| `golden_candidate_0120` | @AppleSupport ????... | `device_hardware` | `other_unclear` | 0.000 | `empty` | `ASK_CLARIFICATION` |
| `golden_candidate_0163` | Hmmm. I've updated to iOS 11.1 but my emojis haven't up... | `software_update` | `other_unclear` | 0.000 | `unknown` | `ESCALATE` |
| `golden_candidate_0168` | @AppleSupport so my iPhone charger literally became so ... | `battery_power` | `battery_power` | 0.000 | `unknown` | `ESCALATE` |
| `golden_candidate_0177` | As much as I love @115858 I’m about to switch... my @11... | `security_privacy` | `security_privacy` | 0.000 | `unknown` | `ESCALATE` |
| `golden_candidate_0186` | Is anyone else’s iPhone making a high pitched alarm noi... | `device_hardware` | `device_hardware` | 0.000 | `unknown` | `ESCALATE` |

### 3.5 Unique Reply Template Distribution (142 Generated Replies)
| Reply Template Excerpt | Count | Share % | Classification Type |
| :--- | :---: | :---: | :--- |
| *"We'd like to look into this with you. Could you let us know your exact device mo..."* | **126** | 88.73% | Static Clarification Prompt |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **7** | 4.93% | Grounded Self-Service Reply |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **3** | 2.11% | Grounded Self-Service Reply |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **1** | 0.70% | Grounded Self-Service Reply |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **1** | 0.70% | Grounded Self-Service Reply |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **1** | 0.70% | Grounded Self-Service Reply |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **1** | 0.70% | Grounded Self-Service Reply |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **1** | 0.70% | Grounded Self-Service Reply |
| *"We'd be glad to help with this. You can find the steps to resolve this here: htt..."* | **1** | 0.70% | Grounded Self-Service Reply |

### 3.6 Conversational Context Audit Metrics (Measured vs. Hypothesized)
- **MEASURED: Total Golden Benchmark Examples**: 250
- **MEASURED: Examples in Threads with >1 Turn (`conversation_length > 1`)**: **250** (100.00%)
- **MEASURED: Evaluated Message is Subsequent Turn (Has Preceding Context in Thread)**: **96** (38.40%)
- **MEASURED: Evaluated Message is Initial Turn (No Preceding Context in Thread)**: **154** (61.60%)
- **MEASURED: Total Intent Misclassifications**: **122**
- **MEASURED: Intent Misclassifications Containing Anaphoric Markers**: **66** (54.10%)
- **MEASURED: Anaphoric Errors with Preceding Context Available**: **29** (23.77% of all intent errors)
- **MEASURED: Anaphoric Errors with No Prior Thread Context (Turn 0)**: **37**
- **HYPOTHESIS**: Single-turn inference caused the intent error on cases with prior thread context; causal improvement from concatenating history remains to be experimentally verified.

---

## 4. Methodological Boundaries & Anti-Regression Invariants
1. **Zero Data Leakage**: Evaluation harness processes only inbound customer queries; ground truth labels are attached strictly post-inference.
2. **Zero Inventions**: All counts are derived deterministically from the 250 evaluated Golden records. All example IDs are verified Golden instances.
3. **Zero Frozen Component Alterations**: Classifier weights, TF-IDF indices, AgentController logic, escalation thresholds, and prompt templates were completely untouched during this analysis milestone.