# AppleSupport Customer Support Golden Evaluation Set

This directory houses the ground-truth **250-example Golden Set** created for evaluating the Hiver Customer Support Agent.

---

## 1. Dataset Provenance & Selection

* **Source Dataset**: Kaggle Customer Support on Twitter ([`twcs.csv`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)) containing ~2.81M customer service tweets across 108 commercial brands.
* **Selected Brand**: `@AppleSupport`
  * Represents the largest single-brand dialogue cluster in the dataset (~119,681 eligible customer inbound messages across reconstructed multi-turn conversations).
  * High complexity: rich taxonomy covering consumer electronics hardware, cloud services, OS updates, app ecosystems, and account security.

---

## 2. Sampling Methodology: Stratified vs. Random Sampling

Pure random sampling in customer support datasets causes severe class imbalance, over-indexing on high-frequency boilerplate greetings while under-representing critical security incidents, rare edge cases, and complex multi-turn failures.

To establish an evaluation benchmark, candidates were selected via **stratified candidate sampling** (seed: `20260909`, total: 250 examples) across four dimensions:
1. **Conversation Length**: Balanced across short ($\le 2$ turns), medium (3–4 turns), and long ($\ge 5$ turns) dialogue trajectories.
2. **Intent Stratification**: Ensuring coverage of rare topic clusters, high-frequency topics, and ambiguous queries.
3. **Difficulty Tiers**:
   - `specific_candidate` (92): Specific, well-formed issues.
   - `likely_multi_intent` (77): Messages exhibiting competing issue signals (e.g. update + battery + billing).
   - `ambiguous_vague` (42): Emotion-driven or underspecified queries requiring contextual clarification.
   - `low_context` (39): Brief messages (e.g. "Yes", "DM is referred to?", screenshots) where true intent is only recoverable through conversation history.
4. **Risk Tiers**: Stratified representation across low, medium, high, and critical safety/financial risks.

*Note: Exploratory regex hints (`candidate_intent_hint`) were strictly used as stratification scaffolding and never as ground truth.*

---

## 3. Human Annotation Protocol

All 250 candidates were **100% manually reviewed and hand-labelled** in verified batches with full multi-turn conversational context (`conversation_context`).

### Gold Annotation Schema

| Column | Type | Allowed Values | Description |
| :--- | :---: | :--- | :--- |
| `gold_intent` | String | 15 taxonomy intents | The customer's single primary actionable problem. |
| `gold_risk` | String | `low`, `medium`, `high`, `critical` | Assessment of customer impact, safety, financial exposure, or urgency. |
| `gold_action` | String | `AUTO_HANDLE`, `ESCALATE` | Operational routing decision for the agent pipeline. |
| `annotation_notes`| String | Free text | Concrete rationale explaining primary intent, boundary decisions, and routing logic. |

### Ground-Truth Intent Taxonomy (15 Intents)

Defined in [`configs/intents.yaml`](../../configs/intents.yaml):
1. **`device_hardware`** (46, 18.4%): Physical hardware, screen/touch, buttons, audio/speakers, camera, freezing, or total system reboots.
2. **`software_update`** (44, 17.6%): iOS/macOS installation failures, update verification errors, OS rollbacks, or regressions directly tied to an update.
3. **`battery_power`** (38, 15.2%): Battery drainage, charging failures, cable/adapter issues, or device overheating.
4. **`other_unclear`** (25, 10.0%): Insufficient context, isolated image links, generic frustration, or unsupported queries.
5. **`connectivity`** (19, 7.6%): Wi-Fi, Bluetooth, CarPlay drops, cellular data toggles, or network throughput degradation.
6. **`apple_id_account`** (16, 6.4%): Account locks, password resets, 2FA verification codes, credential authentication, or disabled IDs.
7. **`repair_service`** (13, 5.2%): In-person Genius Bar bookings, AppleCare coverage disputes, or post-repair failure inquiries.
8. **`icloud`** (8, 3.2%): iCloud Photo Library sync stalls, cloud backup errors, or storage quota allocation.
9. **`app_store`** (7, 2.8%): Inability to download apps, pending download spinners, or App Store compatibility errors.
10. **`billing_payments`** (7, 2.8%): Payment method rejection (CVV errors), unexpected charges, or refund requests.
11. **`security_privacy`** (7, 2.8%): Phishing emails, account takeovers, unauthorized charges, or stolen credentials.
12. **`subscriptions_media`** (7, 2.8%): Apple Music / iTunes library access, subscription sharing, or playback control bugs.
13. **`orders_delivery`** (6, 2.4%): Online order status, courier delivery failures (UPS), or product return policy windows.
14. **`how_to_information`** (5, 2.0%): Informational settings inquiries, feature how-tos, or device capability questions.
15. **`setup_activation`** (2, 0.8%): Device activation lock, preorder carrier number activation, or setup assistant.

---

## 4. Operational Principle: Intent vs. Action Independence

A central architectural requirement of the support agent is that **Action Decision (`AUTO_HANDLE` vs `ESCALATE`) is decoupled from Intent Classification**:

* **Risk $\ne$ Action**: A high-risk inquiry does not automatically mandate escalation if an established official self-service resolution exists (e.g. fast charging documentation or return policy confirmation).
* **Intent $\ne$ Action**: An inquiry classified under `device_hardware` can be `AUTO_HANDLE` if a known workaround exists (e.g. the iOS keyboard "I" autocorrect replacement), or `ESCALATE` if physical diagnostic intervention is required.
* **Separation of Concerns**:
  - `gold_intent` captures **what** the customer's technical problem is.
  - `gold_action` captures **how** the customer support organization safely processes that problem given business boundaries, safety hazards, and self-service capabilities.

---

## 5. Strict Evaluation Integrity Policy (No Training Data Contamination)

> [!CAUTION]
> **STRICT USAGE RESTRICTION**:
> The Golden Set (`golden_annotation.csv` and `golden_candidates.csv`) is a permanently frozen **test and evaluation benchmark**.
>
> 1. It must **NEVER** be used to train or fine-tune classification models.
> 2. It must **NEVER** be sampled for few-shot prompt examples.
> 3. It must **NEVER** be ingested into the knowledge retrieval vector database.
>
> Doing so causes data leakage, rendering automated evaluation metrics statistically invalid.

---

## 6. Files & Checksums

| File | Purpose | Rows | SHA256 Checksum |
| :--- | :--- | :---: | :--- |
| [`golden_candidates.csv`](golden_candidates.csv) | Stratified candidate set | 250 | `d30649967307aae70633ec56b7c9fafecf3246929862ef4383ce6529620319a4` |
| [`golden_annotation.csv`](golden_annotation.csv) | Hand-labelled ground truth | 250 | `854cda964c0f63f9268ba00e3e4e28a8d212cbea3b5ef9aa3fa99e773c8258f7` |
| [`golden_metadata.json`](golden_metadata.json) | Machine-readable metadata | - | Complete distribution and provenance metadata |
