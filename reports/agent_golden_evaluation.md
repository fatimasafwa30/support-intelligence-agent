# AgentController Golden Set Evaluation Report (Milestone 18)

## 1. Executive Summary & Core Results
- **Benchmark Dataset**: AppleSupport Human-Labelled Golden Set (250 examples)
- **Evaluation Nature**: Read-only, deterministic, offline/mock execution without data leakage.
- **Primary Operational Decisions** (3-Way Routing):
  - **AUTO_HANDLE**: 16 (6.40%) — Grounded autonomous resolutions
  - **ASK_CLARIFICATION**: 126 (50.40%) — Interactive clarification requests
  - **ESCALATE**: 108 (43.20%) — Human support queue transfers
- **Critical Safety Recall**: **100.00%** (10 of 10 critical hazard cases escalated)

---

## 2. Primary 3-Way Operational Action Breakdown
| Operational Action | Definition | Cases | Share % |
| :--- | :--- | :---: | :---: |
| **`AUTO_HANDLE`** | Direct self-service response with grounded evidence & citation integrity | 16 | 6.40% |
| **`ASK_CLARIFICATION`** | First-class clarification prompt for ambiguous or low-confidence queries | 126 | 50.40% |
| **`ESCALATE`** | Transfer to human specialist queue based on policy boundary or risk | 108 | 43.20% |

### Queue Routing Breakdown
| Target Queue | Case Count | Share % |
| :--- | :---: | :---: |
| `general_support` | 218 | 87.20% |
| `hardware_repair` | 9 | 3.60% |
| `self_service` | 16 | 6.40% |
| `account_security` | 4 | 1.60% |
| `safety_team` | 3 | 1.20% |

---

## 3. Secondary Binary Action Comparison (vs. Historical 2-Way Gold Action)
> *Note: Compares `AUTO_HANDLE` against Non-AUTO (`ESCALATE` + `ASK_CLARIFICATION`) to directly benchmark against the 2-way `gold_action`.*

| Metric | Agent Result | Linear Baseline (M16) | Delta |
| :--- | :---: | :---: | :---: |
| **Action Accuracy** | **72.00%** | 70.80% | +1.20% |
| **Macro F1-Score** | **50.85%** | 46.23% | — |
| **Under-Escalation Rate** (Dangerous FN) | **4.44%** (8/180) | 3.89% (7/180) | +0.55% |
| **False Escalation Rate on Auto** | **88.57%** (62/70) | 94.29% (66/70) | -5.72% |
| **Overall Escalation Rate** | **93.60%** (234/250) | 95.60% (239/250) | — |
| **Critical Safety Recall** | **100.00%** | 100.00% (10/10) | 0.00% |

### Binary Confusion Matrix
- **Gold ESCALATE -> Pred ESCALATE**: 172
- **Gold ESCALATE -> Pred AUTO_HANDLE**: 8 (Under-Escalation)
- **Gold AUTO_HANDLE -> Pred ESCALATE**: 62 (Over-Escalation / Clarification)
- **Gold AUTO_HANDLE -> Pred AUTO_HANDLE**: 8

---

## 4. Intent Classification Performance
| Slice | Agent Result | Standalone Classifier Baseline | Delta |
| :--- | :---: | :---: | :---: |
| **Specific-Intent Accuracy** (225 cases) | **48.00%** | 52.00% | -4.00% |
| **Specific-Intent Macro F1** | **54.09%** | 50.19% | — |
| **Full Golden Accuracy** (250 cases) | **51.20%** | 46.80% | — |
| **High-Confidence Accuracy** (>= 0.50) | **81.48%** | 82.05% | — |

---

## 5. Agentic Retrieval & Retry Dynamics
- **Attempt 1 Mean Similarity**: 0.3983 (Median: 0.3168)
- **Final Mean Similarity**: 0.3997
- **Retrieval Retries Triggered**: 39 (15.60%)
- **Retrieval Retries Adopted**: 8
- **Borderline Cases Upgraded** ($0.25 \le \text{sim} < 0.35 \rightarrow \ge 0.35$): 4 of 103 (3.88%)

---

## 6. Evidence URL Verification (Citation Integrity)
> *Clarification: Evaluates programmatic citation and link integrity against retrieved evidence; does not measure complete semantic reply correctness.*

- **Replies Reaching Generation**: 142
- **Evidence URL Verification Pass Rate**: **40.85%** (58 passed)
- **Hallucinated / Ungrounded URLs Stripped**: 0

---

## 7. Zero-Tolerance Critical Safety Audit
- **Critical Safety Hazard Recall**: **100.00%** (10 / 10)
- **Pre-Retrieval Hazard Short-Circuits**: 7 cases
- **Pre-Generation Boundary Short-Circuits**: 101 cases
- **Missed Critical Cases Count**: 0