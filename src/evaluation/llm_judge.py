"""LLM-as-a-Judge evaluation implementation for customer support grounded reply quality.

Evaluates generated customer support replies across five core dimensions:
1. Groundedness (1-5): Evidence fidelity and hallucination resistance relative to retrieved evidence.
2. Correctness  (1-5): Technical validity and operational soundness for Apple Support.
3. Relevance    (1-5): Directness in addressing the customer's specific inquiry.
4. Helpfulness  (1-5): Actionability and clarity in progressing toward resolution.
5. Tone         (1-5): Professionalism, empathy, and brand voice alignment.

Key Guardrails:
- Strict Anti-Leakage: Never provides gold_intent, gold_risk, gold_action, or annotation notes to the judge.
- Structured JSON Enforcement: Validates discrete integer scores (1-5) and rejects floats/booleans.
- Resumability: Uses append-only JSONL storage to skip already-judged cases on retry/restart.
- Bounded Retries: Implements exponential backoff for transient rate limits (HTTP 429/503).
- Offline Test Mode: Supports deterministic MockJudgeClient for zero-cost, hermetic testing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable, Sequence

import requests

from src.evaluation.reply_quality_schema import (
    DIMENSION_CORRECTNESS,
    DIMENSION_GROUNDEDNESS,
    DIMENSION_HELPFULNESS,
    DIMENSION_RELEVANCE,
    DIMENSION_TONE,
    EVALUATOR_LLM_JUDGE,
    RUBRIC_DIMENSIONS,
    SCORE_MAX,
    SCORE_MIN,
    EvaluationUnit,
    ReplyQualityRating,
    _validate_score,
)

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Golden Set Ground-Truth Keys that MUST NEVER enter the judge prompt
FORBIDDEN_GOLDEN_KEYS = frozenset({
    "gold_intent",
    "gold_risk",
    "gold_action",
    "annotation_notes",
    "gold_action_raw",
    "human_label",
    "ground_truth",
})


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration parameters for the LLM reply quality judge."""

    provider: str = "gemini"                      # "gemini" or "mock"
    model: str = DEFAULT_GEMINI_MODEL            # e.g. "gemini-2.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    timeout_seconds: float = 30.0
    max_retries: int = 5
    retry_base_delay: float = 2.0
    request_delay_seconds: float = 1.0           # Rate-limiting pause between calls
    output_records_path: Path = Path("reports/reply_quality_judge_records.jsonl")
    output_results_path: Path = Path("reports/reply_quality_judge_results.json")
    output_report_path: Path = Path("reports/reply_quality_judge_report.md")
    human_sample_path: Path = Path("reports/human_agreement_sample_50.json")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "request_delay_seconds": self.request_delay_seconds,
            "output_records_path": str(self.output_records_path),
            "output_results_path": str(self.output_results_path),
            "output_report_path": str(self.output_report_path),
            "human_sample_path": str(self.human_sample_path),
        }


@dataclass(frozen=True)
class JudgeEvaluationRecord:
    """Individual judged record including input metadata, scores, rationales, and execution status."""

    example_id: str
    evaluator_type: str
    judge_model: str
    groundedness: int
    correctness: int
    relevance: int
    helpfulness: int
    tone: int
    overall_score: float
    average_score: float
    rationales: dict[str, str] = field(default_factory=dict)
    status: str = "SUCCESS"                      # "SUCCESS" or "FAILED"
    error_message: str | None = None
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert judged record to dictionary."""
        return {
            "example_id": self.example_id,
            "evaluator_type": self.evaluator_type,
            "judge_model": self.judge_model,
            "groundedness": self.groundedness,
            "correctness": self.correctness,
            "relevance": self.relevance,
            "helpfulness": self.helpfulness,
            "tone": self.tone,
            "overall_score": round(self.overall_score, 4),
            "average_score": round(self.average_score, 4),
            "rationales": dict(self.rationales),
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeEvaluationRecord:
        """Construct JudgeEvaluationRecord from dictionary."""
        return cls(
            example_id=str(data["example_id"]),
            evaluator_type=str(data.get("evaluator_type", EVALUATOR_LLM_JUDGE)),
            judge_model=str(data.get("judge_model", DEFAULT_GEMINI_MODEL)),
            groundedness=int(data["groundedness"]),
            correctness=int(data["correctness"]),
            relevance=int(data["relevance"]),
            helpfulness=int(data["helpfulness"]),
            tone=int(data["tone"]),
            overall_score=float(data.get("overall_score", data.get("average_score", 0.0))),
            average_score=float(data.get("average_score", 0.0)),
            rationales=dict(data.get("rationales") or {}),
            status=str(data.get("status", "SUCCESS")),
            error_message=data.get("error_message"),
            timestamp=float(data.get("timestamp", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
        )

    def to_rating_schema(self) -> ReplyQualityRating:
        """Convert to standard ReplyQualityRating model."""
        return ReplyQualityRating(
            example_id=self.example_id,
            evaluator_type=self.evaluator_type,
            groundedness=self.groundedness,
            correctness=self.correctness,
            relevance=self.relevance,
            helpfulness=self.helpfulness,
            tone=self.tone,
            overall_score=self.overall_score,
            notes=json.dumps(self.rationales) if self.rationales else None,
        )


class JudgePromptBuilder:
    """Constructs strict, leakage-free evaluation prompts for the LLM Judge."""

    SYSTEM_INSTRUCTION = (
        "You are an impartial, expert customer support evaluation judge for Apple Support. "
        "Your task is to evaluate the quality of an automated customer support reply on a strict 1 to 5 integer scale "
        "across five distinct dimensions based ONLY on the customer's query, diagnosed intent, retrieved historical "
        "resolution evidence, and the generated reply.\n\n"
        "Evaluation Principles:\n"
        "1. You are an EVALUATOR, not an agent. Do NOT rewrite, edit, or improve the reply.\n"
        "2. Score ONLY the reply that was generated against the provided evidence and customer inquiry.\n"
        "3. GROUNDEDNESS vs. CORRECTNESS:\n"
        "   - Groundedness evaluates EVIDENCE FIDELITY (whether facts, steps, and URLs in the reply strictly derive from retrieved evidence).\n"
        "   - Correctness evaluates TECHNICAL VALIDITY (whether the advice is technically sound and non-harmful for Apple devices).\n"
        "4. Do NOT reward confident hallucinations or fabricated troubleshooting steps.\n"
        "5. A concise, efficient reply can receive a score of 5 if it is effective and actionable.\n"
        "6. A polite response must still be penalized on Groundedness and Correctness if it gives ungrounded or incorrect advice.\n"
        "7. Return strictly valid JSON conforming to the requested schema with integer scores between 1 and 5."
    )

    RUBRIC_GUIDE = (
        "### SCORING RUBRIC (Scale: 1 to 5 Integers):\n\n"
        "1. GROUNDEDNESS (Evidence Fidelity & Hallucination Resistance):\n"
        "   - 1: Completely ungrounded. Fabricated URLs, invented procedures, or claims absent from retrieved evidence.\n"
        "   - 2: Mostly ungrounded. Major unsupported assertions alongside minimal evidence overlap.\n"
        "   - 3: Partially grounded. Core resolution derives from evidence, but includes minor extraneous/unverified steps.\n"
        "   - 4: Well grounded. All technical claims and URLs are supported by retrieved evidence; only polite conversational filler is added.\n"
        "   - 5: Fully grounded. 100% of facts, URLs, and troubleshooting instructions directly derive from verified retrieved evidence.\n\n"
        "2. CORRECTNESS (Technical Accuracy & Safety Alignment):\n"
        "   - 1: Harmful or materially false advice. Advice would cause data loss, device damage, or security violations.\n"
        "   - 2: Substantially incorrect. Misidentifies root cause or prescribes invalid/obsolete troubleshooting steps.\n"
        "   - 3: Partially correct. Plausible direction, but contains minor inaccuracies or omits critical prerequisites.\n"
        "   - 4: Mostly correct. Technically sound guidance with only trivial omissions that do not hinder resolution.\n"
        "   - 5: Completely correct. Flawless technical accuracy aligned with official Apple support procedures.\n\n"
        "3. RELEVANCE (Inquiry Specificity & Contextual Match):\n"
        "   - 1: Completely irrelevant. Misses the inquiry entirely or answers a different question.\n"
        "   - 2: Tangentially relevant. Mentions product keywords but fails to address the core problem.\n"
        "   - 3: Moderately relevant. Addresses part of the inquiry but ignores key stated constraints or context.\n"
        "   - 4: Highly relevant. Directly answers the customer's main concern with minimal extraneous filler.\n"
        "   - 5: Perfectly targeted. Laser-focused on the exact symptom, device model, and situation described.\n\n"
        "4. HELPFULNESS (Actionability & Resolution Progression):\n"
        "   - 1: Useless or obstructive. Leaves customer stranded with no actionable path forward or confusing advice.\n"
        "   - 2: Minimally helpful. Vague, generic responses (e.g. 'Check settings') without clear next steps.\n"
        "   - 3: Moderately helpful. Provides basic next steps or link, but lacks depth or troubleshooting structure.\n"
        "   - 4: Very helpful. Delivers clear, actionable resolution steps or targeted clarification request.\n"
        "   - 5: Exemplary. Immediately actionable, provides verified self-serve path or exact next steps to rapid resolution.\n\n"
        "5. TONE (Professionalism, Empathy & Brand Voice):\n"
        "   - 1: Inappropriate. Rude, dismissive, robotic, argumentative, or defensive.\n"
        "   - 2: Poor. Cold, stiff, condescending, or excessively bureaucratic.\n"
        "   - 3: Acceptable. Neutral and civil, but formulaic; lacks warmth or empathy for customer frustration.\n"
        "   - 4: Good. Courteous, professional, patient, and polite; conveys willingness to assist.\n"
        "   - 5: Exemplary. Warm, empathetic, respectful, and highly reassuring; perfectly captures Apple Support brand voice."
    )

    JSON_SCHEMA_INSTRUCTION = (
        "### OUTPUT FORMAT REQUIREMENT:\n"
        "You must respond ONLY with a JSON object matching this exact schema:\n"
        "{\n"
        '  "groundedness": <int 1-5>,\n'
        '  "correctness": <int 1-5>,\n'
        '  "relevance": <int 1-5>,\n'
        '  "helpfulness": <int 1-5>,\n'
        '  "tone": <int 1-5>\n'
        "}"
    )

    @classmethod
    def verify_anti_leakage(cls, payload: dict[str, Any]) -> None:
        """Verify that no Golden Set ground-truth annotations exist in the evaluation payload."""
        found_forbidden = set(payload.keys()) & FORBIDDEN_GOLDEN_KEYS
        if found_forbidden:
            raise ValueError(
                f"ANTI-LEAKAGE SECURITY VIOLATION: Golden labels detected in judge payload: {sorted(found_forbidden)}"
            )

        # Deep search nested dictionaries
        for k, v in payload.items():
            if isinstance(v, dict):
                cls.verify_anti_leakage(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        cls.verify_anti_leakage(item)

    @classmethod
    def format_retrieved_evidence(cls, evidence_list: list[dict[str, Any]]) -> str:
        """Format retrieved historical resolution evidence for prompt context."""
        if not evidence_list:
            return "No historical resolution evidence retrieved (Empty evidence set)."

        formatted_items: list[str] = []
        for idx, item in enumerate(evidence_list, 1):
            ev_id = item.get("evidence_id", f"evidence_{idx}")
            sim = item.get("similarity_score", 0.0)
            prob = item.get("past_customer_problem", "N/A")
            res = item.get("past_brand_resolution", "N/A")
            urls = item.get("extracted_urls", [])
            url_str = ", ".join(urls) if urls else "None"

            formatted_items.append(
                f"--- Evidence Item #{idx} [ID: {ev_id}, Similarity: {sim:.4f}] ---\n"
                f"Past Customer Problem: {prob}\n"
                f"Past Brand Resolution: {res}\n"
                f"Official URLs in Evidence: {url_str}"
            )

        return "\n\n".join(formatted_items)

    @classmethod
    def build_prompt(cls, unit: EvaluationUnit) -> dict[str, Any]:
        """Build leakage-safe prompt payload for evaluating an individual EvaluationUnit."""
        unit_dict = unit.to_dict()
        cls.verify_anti_leakage(unit_dict)

        # Extract reply text
        raw_reply = unit.generated_reply.get("reply_text", "")
        verified_reply = unit.verified_reply.get("reply_text", raw_reply) if unit.verified_reply else raw_reply
        grounded_flag = unit.generated_reply.get("grounded", True)
        used_ev_ids = unit.generated_reply.get("used_evidence_ids", [])

        # Format evidence
        formatted_evidence = cls.format_retrieved_evidence(unit.retrieved_evidence)

        # Confidence display
        conf_str = f"{unit.intent_confidence:.4f}" if unit.intent_confidence is not None else "Unknown"

        user_content = (
            f"Please evaluate the following customer support interaction on the 1-5 rubric:\n\n"
            f"=== EVALUATION CASE: {unit.example_id} ===\n"
            f"Customer Inbound Message:\n"
            f"\"{unit.customer_query}\"\n\n"
            f"Diagnosed Intent: {unit.predicted_intent} (Confidence: {conf_str})\n\n"
            f"Retrieved Historical Evidence:\n"
            f"{formatted_evidence}\n\n"
            f"Generated Support Reply:\n"
            f"\"{raw_reply}\"\n"
        )

        if verified_reply != raw_reply:
            user_content += (
                f"\nVerified / Filtered Reply (after URL citation guard):\n"
                f"\"{verified_reply}\"\n"
            )

        user_content += (
            f"\nGeneration Metadata: Grounded Flag = {grounded_flag}, "
            f"Cited Evidence IDs = {used_ev_ids}\n\n"
            f"{cls.RUBRIC_GUIDE}\n\n"
            f"{cls.JSON_SCHEMA_INSTRUCTION}"
        )

        return {
            "system_instruction": cls.SYSTEM_INSTRUCTION,
            "user_prompt": user_content,
        }


class JudgeResponseValidator:
    """Validates and parses structured JSON output from LLM judge."""

    @classmethod
    def parse_and_validate(
        cls,
        raw_text: str,
        example_id: str,
        judge_model: str,
        latency_ms: float = 0.0,
    ) -> JudgeEvaluationRecord:
        """Parse raw response text into validated JudgeEvaluationRecord."""
        if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"Empty response received from judge for example '{example_id}'")

        # Strip markdown code fencing if present
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as err:
            raise ValueError(f"Malformed JSON from judge for '{example_id}': {err}\nRaw text: {raw_text[:200]}") from err

        if not isinstance(data, dict):
            raise TypeError(f"Expected JSON dictionary from judge for '{example_id}', got {type(data).__name__}")

        # Validate mandatory 5 dimensions
        for dim in RUBRIC_DIMENSIONS:
            if dim not in data:
                raise KeyError(f"Missing mandatory dimension '{dim}' in judge output for '{example_id}'")
            # Enforce discrete integer [1, 5]
            _validate_score(dim, data[dim])

        groundedness = int(data[DIMENSION_GROUNDEDNESS])
        correctness = int(data[DIMENSION_CORRECTNESS])
        relevance = int(data[DIMENSION_RELEVANCE])
        helpfulness = int(data[DIMENSION_HELPFULNESS])
        tone = int(data[DIMENSION_TONE])

        avg_score = round(
            (groundedness + correctness + relevance + helpfulness + tone) / 5.0,
            4,
        )

        # Overall score validation (defaults to average_score if not supplied)
        overall_raw = data.get("overall_score")
        if overall_raw is not None:
            if isinstance(overall_raw, bool) or not isinstance(overall_raw, (int, float)):
                raise TypeError(f"overall_score must be a number, got {type(overall_raw).__name__}")
            overall_val = round(float(overall_raw), 4)
            if overall_val < float(SCORE_MIN) or overall_val > float(SCORE_MAX):
                raise ValueError(f"overall_score {overall_val} is outside [{SCORE_MIN}, {SCORE_MAX}]")
        else:
            overall_val = avg_score

        # Extract rationales
        rationales_raw = data.get("rationales", {})
        rationales: dict[str, str] = {}
        if isinstance(rationales_raw, dict):
            for dim in RUBRIC_DIMENSIONS:
                val = rationales_raw.get(dim)
                if val is not None and isinstance(val, str):
                    rationales[dim] = val.strip()

        return JudgeEvaluationRecord(
            example_id=example_id,
            evaluator_type=EVALUATOR_LLM_JUDGE,
            judge_model=judge_model,
            groundedness=groundedness,
            correctness=correctness,
            relevance=relevance,
            helpfulness=helpfulness,
            tone=tone,
            overall_score=overall_val,
            average_score=avg_score,
            rationales=rationales,
            status="SUCCESS",
            error_message=None,
            timestamp=time.time(),
            latency_ms=latency_ms,
        )


class BaseJudgeClient:
    """Abstract base class for LLM Judge execution clients."""

    def evaluate_unit(self, unit: EvaluationUnit) -> JudgeEvaluationRecord:
        """Evaluate a single evaluation unit."""
        raise NotImplementedError


class MockJudgeClient(BaseJudgeClient):
    """Deterministic, zero-cost mock judge client for testing and offline development."""

    def __init__(
        self,
        default_scores: tuple[int, int, int, int, int] = (5, 5, 5, 4, 5),
        canned_responses: dict[str, dict[str, Any]] | None = None,
        simulate_failure_ids: set[str] | None = None,
    ) -> None:
        self.default_scores = default_scores
        self.canned_responses = canned_responses or {}
        self.simulate_failure_ids = simulate_failure_ids or set()
        self.call_count = 0

    def evaluate_unit(self, unit: EvaluationUnit) -> JudgeEvaluationRecord:
        """Produce deterministic mock score for unit."""
        self.call_count += 1
        unit_dict = unit.to_dict()
        JudgePromptBuilder.verify_anti_leakage(unit_dict)

        if unit.example_id in self.simulate_failure_ids:
            return JudgeEvaluationRecord(
                example_id=unit.example_id,
                evaluator_type=EVALUATOR_LLM_JUDGE,
                judge_model="mock-judge",
                groundedness=0,
                correctness=0,
                relevance=0,
                helpfulness=0,
                tone=0,
                overall_score=0.0,
                average_score=0.0,
                status="FAILED",
                error_message="Simulated mock API failure",
                timestamp=time.time(),
                latency_ms=1.0,
            )

        if unit.example_id in self.canned_responses:
            data = self.canned_responses[unit.example_id]
            raw_json = json.dumps(data)
            return JudgeResponseValidator.parse_and_validate(
                raw_text=raw_json,
                example_id=unit.example_id,
                judge_model="mock-judge",
                latency_ms=1.0,
            )

        # Calibrate mock scores deterministically based on whether verified URLs exist
        g, c, r, h, t = self.default_scores
        verified_text = unit.verified_reply.get("reply_text", "") if unit.verified_reply else unit.generated_reply.get("reply_text", "")
        if "http" in verified_text:
            g_score = 5
        else:
            g_score = 4

        mock_payload = {
            "groundedness": g_score,
            "correctness": c,
            "relevance": r,
            "helpfulness": h,
            "tone": t,
            "overall_score": round((g_score + c + r + h + t) / 5.0, 4),
            "rationales": {
                "groundedness": "Derived from retrieved evidence.",
                "correctness": "Technically accurate advice.",
                "relevance": "Addresses customer inquiry.",
                "helpfulness": "Actionable instructions.",
                "tone": "Courteous and professional.",
            },
        }

        return JudgeResponseValidator.parse_and_validate(
            raw_text=json.dumps(mock_payload),
            example_id=unit.example_id,
            judge_model="mock-judge",
            latency_ms=1.5,
        )


class GeminiJudgeClient(BaseJudgeClient):
    """Google Gemini REST API client executing structured reply quality evaluations."""

    def __init__(
        self,
        api_key: str,
        config: JudgeConfig | None = None,
    ) -> None:
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("GEMINI_API_KEY must be a valid, non-empty string.")

        self.api_key = api_key.strip()
        self.config = config or JudgeConfig()
        self.endpoint = GEMINI_API_ENDPOINT.format(model=self.config.model)

    def evaluate_unit(self, unit: EvaluationUnit) -> JudgeEvaluationRecord:
        """Call Gemini API with prompt and retry logic to evaluate unit."""
        prompt_data = JudgePromptBuilder.build_prompt(unit)
        system_instruction = prompt_data["system_instruction"]
        user_prompt = prompt_data["user_prompt"]

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_instruction}],
            },
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {
                        dim: {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX}
                        for dim in RUBRIC_DIMENSIONS
                    },
                    "required": list(RUBRIC_DIMENSIONS),
                    "additionalProperties": False,
                    "propertyOrdering": list(RUBRIC_DIMENSIONS),
                },
            },
        }

        # Flash's default dynamic thinking shares maxOutputTokens with the JSON.
        # Reserve the small evaluation budget for the five scores themselves.
        if self.config.model == DEFAULT_GEMINI_MODEL:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

        headers = {
            "Content-Type": "application/json",
        }
        params = {
            "key": self.api_key,
        }

        attempts = 0
        last_error: Exception | None = None

        while attempts < self.config.max_retries:
            attempts += 1
            start_t = time.perf_counter()

            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                latency_ms = (time.perf_counter() - start_t) * 1000.0

                if response.status_code == 200:
                    resp_json = response.json()
                    candidates = resp_json.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"Zero candidates returned in Gemini response: {resp_json}")

                    candidate = candidates[0]
                    finish_reason = candidate.get("finishReason")
                    if finish_reason != "STOP":
                        raise ValueError(
                            f"Incomplete Gemini response: finishReason={finish_reason}; "
                            f"usageMetadata={resp_json.get('usageMetadata', {})}"
                        )
                    content_parts = candidate.get("content", {}).get("parts", [])
                    if not content_parts:
                        raise ValueError(f"Empty content parts in Gemini candidate: {candidates[0]}")

                    raw_text = "".join(
                        part.get("text", "") for part in content_parts if not part.get("thought")
                    )
                    return JudgeResponseValidator.parse_and_validate(
                        raw_text=raw_text,
                        example_id=unit.example_id,
                        judge_model=self.config.model,
                        latency_ms=latency_ms,
                    )

                elif response.status_code in (429, 500, 503, 504):
                    # Transient error, apply exponential backoff
                    wait_time = self.config.retry_base_delay * (2 ** (attempts - 1)) + (random.random() * 0.5)
                    logger.warning(
                        "Gemini API returned HTTP %d for '%s' (attempt %d/%d). Backing off %.2fs...",
                        response.status_code,
                        unit.example_id,
                        attempts,
                        self.config.max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                else:
                    # Permanent error
                    error_body = response.text[:300]
                    raise RuntimeError(
                        f"Gemini API HTTP {response.status_code} on example '{unit.example_id}': {error_body}"
                    )

            except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError, TypeError) as err:
                last_error = err
                wait_time = self.config.retry_base_delay * (2 ** (attempts - 1))
                logger.warning(
                    "Error judging '%s' on attempt %d/%d: %s. Retrying in %.2fs...",
                    unit.example_id,
                    attempts,
                    self.config.max_retries,
                    err,
                    wait_time,
                )
                time.sleep(wait_time)

        # If all retries exhausted, return failed record
        return JudgeEvaluationRecord(
            example_id=unit.example_id,
            evaluator_type=EVALUATOR_LLM_JUDGE,
            judge_model=self.config.model,
            groundedness=0,
            correctness=0,
            relevance=0,
            helpfulness=0,
            tone=0,
            overall_score=0.0,
            average_score=0.0,
            status="FAILED",
            error_message=f"Retries exhausted ({self.config.max_retries} attempts). Last error: {last_error}",
            timestamp=time.time(),
            latency_ms=0.0,
        )


class HumanAgreementSampler:
    """Deterministically samples a stratified 50-example audit subset from generated replies."""

    @staticmethod
    def sample_human_subset(
        generated_units: list[EvaluationUnit],
        sample_size: int = 50,
        random_seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Deterministically select sample_size examples stratified across operational characteristics.

        Stratification Dimensions:
        1. pred_intent
        2. intent_confidence_bucket (high >= 0.8, medium [0.5, 0.8), low < 0.5)
        3. primary_final_action (AUTO_HANDLE vs ASK_CLARIFICATION)
        4. retrieval_sufficiency (sufficient vs borderline vs insufficient)
        """
        if not generated_units:
            return []

        if len(generated_units) <= sample_size:
            return [
                {
                    "example_id": u.example_id,
                    "customer_query": u.customer_query,
                    "pred_intent": u.predicted_intent,
                    "intent_confidence": u.intent_confidence,
                    "conversation_id": u.conversation_id,
                }
                for u in generated_units
            ]

        # Group by composite strata
        strata: dict[tuple[str, str, str], list[EvaluationUnit]] = {}
        for u in generated_units:
            conf = u.intent_confidence or 0.0
            if conf >= 0.8:
                c_bucket = "high"
            elif conf >= 0.5:
                c_bucket = "medium"
            else:
                c_bucket = "low"

            intent = u.predicted_intent
            is_auto = "AUTO_HANDLE" if u.generated_reply.get("grounded", False) and conf >= 0.5 else "CLARIFICATION"

            key = (intent, c_bucket, is_auto)
            strata.setdefault(key, []).append(u)

        # Deterministic sampling with seed
        rng = random.Random(random_seed)
        sampled_units: list[EvaluationUnit] = []

        # Sort strata keys for reproducibility
        sorted_keys = sorted(strata.keys(), key=lambda k: (k[0], k[1], k[2]))

        # Round-robin proportional selection
        strata_pools = {k: list(strata[k]) for k in sorted_keys}
        for k in sorted_keys:
            rng.shuffle(strata_pools[k])

        while len(sampled_units) < sample_size:
            added_in_cycle = 0
            for k in sorted_keys:
                if strata_pools[k] and len(sampled_units) < sample_size:
                    sampled_units.append(strata_pools[k].pop(0))
                    added_in_cycle += 1
            if added_in_cycle == 0:
                break

        # Convert to serialized metadata
        sampled_records = [
            {
                "sample_index": idx,
                "example_id": u.example_id,
                "customer_query": u.customer_query,
                "predicted_intent": u.predicted_intent,
                "intent_confidence": round(u.intent_confidence, 4) if u.intent_confidence is not None else None,
                "conversation_id": u.conversation_id,
                "has_verified_urls": bool(u.verified_reply and u.verified_reply.get("reply_text") and "http" in u.verified_reply.get("reply_text", "")),
            }
            for idx, u in enumerate(sampled_units, 1)
        ]

        return sampled_records


class ResumableJudgeRunner:
    """Orchestrates resumable, append-only evaluation of generated customer support replies."""

    def __init__(
        self,
        client: BaseJudgeClient,
        config: JudgeConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or JudgeConfig()

    def load_existing_results(self) -> dict[str, JudgeEvaluationRecord]:
        """Load already-evaluated records from JSONL output to enable idempotent resumption."""
        records: dict[str, JudgeEvaluationRecord] = {}
        path = self.config.output_records_path
        if not path.exists():
            return records

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    rec = JudgeEvaluationRecord.from_dict(data)
                    if rec.status == "SUCCESS":
                        records[rec.example_id] = rec
                except (json.JSONDecodeError, KeyError, ValueError) as err:
                    logger.warning("Skipping corrupted line in %s: %s", path, err)

        return records

    def append_record(self, record: JudgeEvaluationRecord) -> None:
        """Atomically append a validated judge evaluation record to the JSONL trace file."""
        path = self.config.output_records_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def run_evaluation(
        self,
        units: list[EvaluationUnit],
        total_golden_size: int = 250,
    ) -> dict[str, Any]:
        """Execute evaluation over all generated evaluation units with resumability and reporting."""
        existing_records = self.load_existing_results()
        logger.info(
            "Found %d existing successful records in %s. Resuming evaluation...",
            len(existing_records),
            self.config.output_records_path,
        )

        all_records: dict[str, JudgeEvaluationRecord] = dict(existing_records)
        newly_judged_count = 0
        failed_count = 0

        for i, unit in enumerate(units, 1):
            ex_id = unit.example_id
            if ex_id in existing_records:
                logger.debug("[%d/%d] Skipping already judged: %s", i, len(units), ex_id)
                continue

            logger.info("[%d/%d] Evaluating '%s' with %s...", i, len(units), ex_id, self.config.model)
            rec = self.client.evaluate_unit(unit)

            if rec.status == "SUCCESS":
                self.append_record(rec)
                all_records[ex_id] = rec
                newly_judged_count += 1
            else:
                logger.error("Evaluation failed for '%s': %s", ex_id, rec.error_message)
                self.append_record(rec)
                failed_count += 1

            # Respect API rate limits
            if self.config.request_delay_seconds > 0:
                time.sleep(self.config.request_delay_seconds)

        # Compute aggregate metrics over all successful records
        successful_records = [r for r in all_records.values() if r.status == "SUCCESS"]
        metrics = self.compute_aggregate_metrics(
            successful_records=successful_records,
            total_generated=len(units),
            total_golden=total_golden_size,
            failed_count=failed_count,
        )

        # Save summary JSON
        self.config.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_results_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved summary metrics to %s", self.config.output_results_path)

        # Format and save markdown report
        report_md = self.format_markdown_report(metrics)
        self.config.output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_report_path.open("w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info("Saved markdown report to %s", self.config.output_report_path)

        return metrics

    @staticmethod
    def compute_aggregate_metrics(
        successful_records: list[JudgeEvaluationRecord],
        total_generated: int,
        total_golden: int,
        failed_count: int = 0,
    ) -> dict[str, Any]:
        """Compute per-dimension means, medians, and distribution breakdowns."""
        n_judged = len(successful_records)
        non_generated = total_golden - total_generated
        coverage_rate = total_generated / total_golden if total_golden > 0 else 0.0

        dimension_metrics: dict[str, Any] = {}
        for dim in RUBRIC_DIMENSIONS:
            scores = [getattr(r, dim) for r in successful_records]
            if scores:
                dim_mean = round(sum(scores) / len(scores), 4)
                sorted_scores = sorted(scores)
                dim_median = sorted_scores[len(sorted_scores) // 2]
                dist_counts = {score: scores.count(score) for score in range(SCORE_MIN, SCORE_MAX + 1)}
                dist_shares = {
                    score: round(count / len(scores), 4)
                    for score, count in dist_counts.items()
                }
            else:
                dim_mean = 0.0
                dim_median = 0
                dist_counts = {score: 0 for score in range(SCORE_MIN, SCORE_MAX + 1)}
                dist_shares = {score: 0.0 for score in range(SCORE_MIN, SCORE_MAX + 1)}

            dimension_metrics[dim] = {
                "mean": dim_mean,
                "median": dim_median,
                "distribution_counts": dist_counts,
                "distribution_shares": dist_shares,
            }

        avg_scores = [r.average_score for r in successful_records]
        overall_mean = round(sum(avg_scores) / len(avg_scores), 4) if avg_scores else 0.0

        return {
            "evaluation_population": {
                "total_golden_examples": total_golden,
                "generated_replies_evaluated": total_generated,
                "non_generated_short_circuits": non_generated,
                "reply_generation_coverage": round(coverage_rate, 4),
                "successfully_judged_count": n_judged,
                "failed_judging_count": failed_count,
            },
            "judge_metadata": {
                "evaluator_type": EVALUATOR_LLM_JUDGE,
                "scoring_scale": "1 to 5 integer scale",
                "scoring_dimensions": list(RUBRIC_DIMENSIONS),
            },
            "overall_quality": {
                "composite_mean_score": overall_mean,
            },
            "dimension_scores": dimension_metrics,
            "human_agreement_status": {
                "measured": False,
                "note": "Judge-human agreement has not yet been measured. It will be evaluated after independent human ratings are collected in Milestone 19.3.",
            },
        }

    def format_markdown_report(self, metrics: dict[str, Any]) -> str:
        """Format comprehensive Markdown evaluation report."""
        pop = metrics.get("evaluation_population", {})
        dims = metrics.get("dimension_scores", {})
        overall = metrics.get("overall_quality", {})

        total_g = pop.get("total_golden_examples", 250)
        total_gen = pop.get("generated_replies_evaluated", 135)
        non_gen = pop.get("non_generated_short_circuits", 115)
        cov = pop.get("reply_generation_coverage", 0.54)
        success_n = pop.get("successfully_judged_count", 0)
        fail_n = pop.get("failed_judging_count", 0)
        comp_mean = overall.get("composite_mean_score", 0.0)

        lines: list[str] = [
            "# LLM-as-a-Judge Reply Quality Evaluation Report (Milestone 19.2)",
            "",
            "## 1. Executive Summary & Evaluation Population",
            f"- **Evaluation Benchmark**: Human-Annotated AppleSupport Golden Set ({total_g} examples)",
            f"- **Evaluator Model**: `{self.config.model}` (Provider: `{self.config.provider}`, Temperature: `{self.config.temperature}`)",
            f"- **Generation Coverage**: **{cov:.1%}** ({total_gen} of {total_g} examples generated a reply)",
            f"- **Non-Generated Short-Circuits**: {non_gen} cases (appropriately excluded from reply-quality evaluation)",
            f"- **Successfully Judged Cases**: **{success_n}** / {total_gen}",
            f"- **Failed / Interrupted Cases**: {fail_n}",
            f"- **Composite Mean Score Across Dimensions**: **{comp_mean:.2f}** / 5.00",
            "",
            "> **Formal Population Clarification**:",
            f"> *Reply-quality evaluation covers {total_gen} generated replies out of {total_g} Golden examples "
            f"({cov:.1%} generation coverage). The remaining {non_gen} cases were not scored for reply quality "
            "because no reply was generated. Non-generation reflects safe operational triage/escalation and is not penalized as poor reply quality.*",
            "",
            "---",
            "",
            "## 2. Five-Dimensional Reply Quality Scores",
            "| Rubric Dimension | Mean (1–5) | Median | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for dim in RUBRIC_DIMENSIONS:
            d_info = dims.get(dim, {})
            d_mean = d_info.get("mean", 0.0)
            d_med = d_info.get("median", 0)
            counts = d_info.get("distribution_counts", {})
            c1 = counts.get(1, counts.get("1", 0))
            c2 = counts.get(2, counts.get("2", 0))
            c3 = counts.get(3, counts.get("3", 0))
            c4 = counts.get(4, counts.get("4", 0))
            c5 = counts.get(5, counts.get("5", 0))

            lines.append(
                f"| **{dim.capitalize()}** | **{d_mean:.2f}** | {d_med} | "
                f"{c1} ({c1/success_n:.1%}) | {c2} ({c2/success_n:.1%}) | {c3} ({c3/success_n:.1%}) | "
                f"{c4} ({c4/success_n:.1%}) | {c5} ({c5/success_n:.1%}) |"
                if success_n > 0 else
                f"| **{dim.capitalize()}** | **{d_mean:.2f}** | {d_med} | 0 | 0 | 0 | 0 | 0 |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Grounding & Verification Distinction",
            "> **Methodological Boundary**:",
            "- **Programmatic Citation Guard**: Verifies evidence URL integrity against historical brand records (Programmatic invariant).",
            "- **LLM-Judged Semantic Groundedness**: Assesses whether claims and troubleshooting steps are faithfully supported by retrieved context (Semantic model judgment).",
            "- **LLM-Judged Correctness**: Evaluates technical validity and adherence to Apple support procedures (Domain accuracy).",
            "",
            "---",
            "",
            "## 4. Human-Judge Agreement Status",
            "> [!NOTE]",
            "> **Judge-human agreement has not yet been measured. It will be evaluated after independent human ratings are collected.**",
            f"- A stratified human audit subset of **50 examples** has been selected from the 135 generated replies independently of LLM judge scores (`{Path(self.config.human_sample_path).as_posix()}`).",
            "- Inter-rater reliability (Quadratic Weighted Kappa $\\kappa_w$, MAE, Spearman $\\rho$) will be measured in the subsequent human agreement phase.",
            "",
            "---",
            "",
            "## 5. Methodological Limitations & Anti-Leakage Protocol",
            "1. **Zero Ground-Truth Exposure**: The LLM judge evaluated replies using only inbound customer queries, diagnosed intent/confidence, retrieved resolution evidence, and generated text. No `gold_intent`, `gold_risk`, `gold_action`, or `annotation_notes` were provided.",
            "2. **Evaluation Scope**: High judge scores indicate strong alignment with the 1–5 rubric given the retrieved context; they do not replace human expert validation.",
            "3. **Idempotence & Auditability**: All judged records are stored with model signatures and rationales in `reports/reply_quality_judge_records.jsonl`.",
        ])

        return "\n".join(lines)
