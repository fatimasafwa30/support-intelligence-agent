"""Deterministic risk detection engine for customer support inquiries.

Evaluates customer messages for physical safety hazards, security breaches,
physical hardware damage, financial disputes, and legal escalations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import re
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "escalation.yaml"

# Risk Categories
CRITICAL_SAFETY = "CRITICAL_SAFETY"
CRITICAL_SECURITY = "CRITICAL_SECURITY"
PHYSICAL_DAMAGE = "PHYSICAL_DAMAGE"
FINANCIAL_DISPUTE = "FINANCIAL_DISPUTE"
LEGAL_REGULATORY = "LEGAL_REGULATORY"


@dataclass(frozen=True)
class RiskAssessment:
    """Structured assessment of operational and safety risk."""

    risk_level: str  # "low", "medium", "high", "critical"
    risk_categories: list[str]
    matched_signals: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Convert assessment to dictionary."""
        return asdict(self)


class RiskDetector:
    """Deterministic risk detection module based on compiled regex patterns."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        cfg_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.config = self._load_config(cfg_path)
        self._compiled_patterns = self._compile_patterns()

        # General customer distress signals (elevate low -> medium)
        self._distress_pattern = re.compile(
            r"\b(?:furious|unacceptable|useless|horrible|terrible|worst|disaster|fed up|pissed|ridiculous)\b|"
            r"[!?]{2,}",
            re.IGNORECASE,
        )

    def _load_config(self, path: Path) -> dict[str, Any]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.warning("Config file not found at %s; using hardcoded fallback patterns.", path)
        return {}

    def _compile_patterns(self) -> dict[str, list[tuple[re.Pattern, str]]]:
        """Compile regex patterns from configuration with raw pattern preservation."""
        risk_signals = self.config.get("risk_signals", {})
        compiled: dict[str, list[tuple[re.Pattern, str]]] = {
            CRITICAL_SAFETY: [],
            CRITICAL_SECURITY: [],
            PHYSICAL_DAMAGE: [],
            FINANCIAL_DISPUTE: [],
            LEGAL_REGULATORY: [],
        }

        cat_mapping = {
            "critical_safety": CRITICAL_SAFETY,
            "critical_security": CRITICAL_SECURITY,
            "physical_damage": PHYSICAL_DAMAGE,
            "financial_dispute": FINANCIAL_DISPUTE,
            "legal_regulatory": LEGAL_REGULATORY,
        }

        # Fallback defaults if config is missing
        fallbacks = {
            "critical_safety": [
                r"\b(?:swelling|swollen|bulging)\b",
                r"\b(?:smoke|smoking|fire)\b",
                r"\b(?:exploded|explosion|sparking|sparks)\b",
                r"\bburn(?:ed|ing|t)?\s+(?:my\s+)?(?:hand|skin|finger|leg)\b",
                r"\b(?:shocked me|electrocuted)\b",
                r"\b(?:melting|melted|caught on fire)\b",
            ],
            "critical_security": [
                r"\b(?:hacked|account takeover)\b",
                r"\b(?:extortion|blackmail)\b",
                r"\b(?:unauthorized|fraudulent)\s+(?:charge|transaction)s?\b",
                r"\b(?:stolen\s+(?:phone|identity|credit card))\b",
            ],
            "physical_damage": [
                r"\b(?:cracked|shattered|broken)\s+(?:screen|glass|display)\b",
                r"\b(?:water damage|liquid spill|dropped in water|dropped in toilet)\b",
                r"\b(?:bent (?:iphone|ipad|phone))\b",
                r"\b(?:button (?:fell off|broken|jammed|stuck))\b",
                r"\b(?:charging port (?:broken|damaged|corroded))\b",
            ],
            "financial_dispute": [
                r"\b(?:charged twice|double charged)\b",
                r"\bdisput(?:e|ing)\s+(?:the\s+)?charge\b",
                r"\bunrecognized charge\b",
                r"\brefund (?:refused|rejected|denied)\b",
            ],
            "legal_regulatory": [
                r"\b(?:lawyer|lawsuit|legal action)\b",
                r"\bsu(?:e|ing)\s+you\b",
                r"\b(?:police report|consumer court)\b",
            ],
        }

        for cfg_key, cat_name in cat_mapping.items():
            pattern_list = risk_signals.get(cfg_key) or fallbacks.get(cfg_key, [])
            for pat_str in pattern_list:
                compiled[cat_name].append((re.compile(pat_str, re.IGNORECASE), pat_str))

        return compiled

    def assess_risk(self, text: str) -> RiskAssessment:
        """Evaluate an inbound customer query and determine risk level and categories.

        Args:
            text: Inbound customer message text.

        Returns:
            Structured RiskAssessment.
        """
        clean_text = str(text or "").strip()
        if not clean_text:
            return RiskAssessment(
                risk_level="low",
                risk_categories=[],
                matched_signals=[],
                rationale="Empty text; default low risk.",
            )

        detected_categories: set[str] = set()
        matched_signals: list[str] = []

        for category, pattern_tuples in self._compiled_patterns.items():
            for pat, raw_str in pattern_tuples:
                match = pat.search(clean_text)
                if match:
                    detected_categories.add(category)
                    matched_signals.append(f"{category}:{match.group(0)}")

        # Decision hierarchy for risk tier
        if CRITICAL_SAFETY in detected_categories or CRITICAL_SECURITY in detected_categories:
            risk_level = "critical"
            rationale = "Critical safety hazard or security compromise detected."
        elif (
            PHYSICAL_DAMAGE in detected_categories
            or FINANCIAL_DISPUTE in detected_categories
            or LEGAL_REGULATORY in detected_categories
        ):
            risk_level = "high"
            rationale = f"High-risk operational conditions detected: {', '.join(sorted(detected_categories))}."
        elif self._distress_pattern.search(clean_text):
            risk_level = "medium"
            rationale = "Elevated customer friction or distress signals detected."
        else:
            risk_level = "low"
            rationale = "Routine inquiry; no critical, physical, or dispute risk signals detected."

        return RiskAssessment(
            risk_level=risk_level,
            risk_categories=sorted(detected_categories),
            matched_signals=matched_signals,
            rationale=rationale,
        )
