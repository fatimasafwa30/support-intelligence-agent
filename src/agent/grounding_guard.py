"""Programmatic verification for exact evidence IDs and URLs in generated replies.

Strictly verifies that:
1. Every evidence ID reported as used exists in the provided retrieval evidence set.
2. Every URL included in the generated reply text exists in the whitelisted URLs
   extracted from the provided retrieval evidence.
Does NOT claim deterministic verification of general factual/technical claims.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.agent.reply_schemas import EvidenceItem, GenerationRequest, GroundedReply, extract_urls

logger = logging.getLogger(__name__)


def verify_and_filter_reply(
    reply: GroundedReply,
    request: GenerationRequest,
) -> GroundedReply:
    """Verify exact evidence IDs and URLs in the reply against the provided evidence.

    Args:
        reply: The raw generated GroundedReply.
        request: The original GenerationRequest containing provided evidence.

    Returns:
        Verified GroundedReply with ungrounded evidence IDs and URLs filtered or flagged.
    """
    valid_evidence_ids = {e.evidence_id for e in request.retrieved_evidence}
    whitelisted_urls: set[str] = set()
    for e in request.retrieved_evidence:
        whitelisted_urls.update(e.extracted_urls)

    # 1. Exact Evidence ID Verification
    verified_evidence_ids = [
        eid for eid in reply.used_evidence_ids if eid in valid_evidence_ids
    ]
    invalid_ids = set(reply.used_evidence_ids) - valid_evidence_ids
    if invalid_ids:
        logger.warning(
            "Filtered %d ungrounded evidence IDs not in retrieved evidence: %s",
            len(invalid_ids),
            sorted(invalid_ids),
        )

    # 2. Exact URL Verification in reply_text
    urls_in_text = extract_urls(reply.reply_text)
    ungrounded_urls = [u for u in urls_in_text if u not in whitelisted_urls]

    clean_reply_text = reply.reply_text
    grounded = reply.grounded
    action = reply.action
    rationale = reply.rationale

    if ungrounded_urls:
        logger.warning(
            "Detected %d ungrounded URL(s) in reply text: %s",
            len(ungrounded_urls),
            ungrounded_urls,
        )
        # Strip ungrounded URLs from the reply text
        for fake_url in ungrounded_urls:
            clean_reply_text = clean_reply_text.replace(fake_url, "").strip()

        # If the reply relied on an ungrounded URL and has no valid evidence, demote to clarification
        if not verified_evidence_ids:
            action = "ASK_CLARIFICATION"
            grounded = False
            rationale = (
                f"{rationale}; Demoted due to ungrounded URL(s) without valid evidence."
            ).lstrip("; ")

    # Whitelisted URLs actually retained in the final text
    final_verified_urls = [
        u for u in extract_urls(clean_reply_text) if u in whitelisted_urls
    ]

    return GroundedReply(
        reply_text=clean_reply_text,
        grounded=grounded and (len(verified_evidence_ids) > 0 or action == "ASK_CLARIFICATION"),
        used_evidence_ids=verified_evidence_ids,
        used_urls=final_verified_urls,
        action=action,
        rationale=rationale,
        provider=reply.provider,
        latency_ms=reply.latency_ms,
    )
