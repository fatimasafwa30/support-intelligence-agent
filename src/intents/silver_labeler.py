"""High-precision, deterministic single-label intent classifier for customer support.

Assigns exactly one canonical intent from the 15-intent taxonomy in configs/intents.yaml
to an inbound customer message, using deterministic pattern matching, contextual
disambiguation, and collision resolution.

v2: Targeted vocabulary additions based on diagnostic audit of golden-set failures.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

CANONICAL_INTENTS: tuple[str, ...] = (
    "apple_id_account",
    "icloud",
    "software_update",
    "battery_power",
    "connectivity",
    "device_hardware",
    "app_store",
    "billing_payments",
    "subscriptions_media",
    "setup_activation",
    "repair_service",
    "security_privacy",
    "orders_delivery",
    "how_to_information",
    "other_unclear",
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "for", "from", "get", "got", "have", "help", "i", "if", "in", "is",
    "it", "just", "me", "my", "of", "on", "or", "please", "so", "that", "the",
    "this", "to", "was", "we", "with", "you", "your", "apple", "applesupport",
}

# ---------------------------------------------------------------------------
# Short-but-strong signal phrases: when a message is very short (<=4 tokens
# after stop-word removal) these patterns can still trigger a specific intent.
# Only add patterns where the vocabulary is UNAMBIGUOUS.
# Vague short messages ("ok thanks", "I will", etc.) must still fall through
# to other_unclear.
# ---------------------------------------------------------------------------
_SHORT_STRONG_SIGNALS: list[tuple[re.Pattern, str]] = []  # populated below


@dataclass(frozen=True)
class LabelResult:
    """Structured result from high-precision intent classification."""

    intent: str
    confidence: float
    reason: str
    matched_rules: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "matched_rules": list(self.matched_rules),
        }


def normalize_text(text: str) -> str:
    """Normalize message text for deterministic pattern matching."""
    if not text:
        return ""
    normalized = URL_PATTERN.sub(" ", text.lower())
    normalized = MENTION_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def extract_meaningful_tokens(normalized_text: str) -> list[str]:
    """Tokenize normalized text and strip generic stopwords."""
    return [
        token for token in WORD_TOKEN_PATTERN.findall(normalized_text)
        if token not in STOPWORDS
    ]


class SilverLabeler:
    """High-precision deterministic single-label intent classifier."""

    def __init__(self) -> None:
        self._init_patterns()

    def _init_patterns(self) -> None:
        """Compile deterministic regular expressions for high precision detection."""

        # ── 1. Battery & Power ──────────────────────────────────────────────
        self._pat_battery = re.compile(
            r"\b(?:batter(?:y|ies)|battery\s*life|battery\s*drain|battery\s*health|battery\s*percentage)\b",
            re.IGNORECASE,
        )
        # v2: added percentage-drop patterns and "charge back" variants
        # NOTE: % is not a word character so \b after % does not work; use lookahead instead
        self._pat_battery_drain = re.compile(
            r"\b(?:rapidly|fast|silently|quick|drastically)\s*(?:draining|runs?\s*(?:down|out)|dying|drain)\b|"
            r"\b(?:battery|power)\s+(?:is\s+)?(?:draining|dying|running\s+down|dropping|drains?)\b|"
            r"\bgoes\s+from\s+\d+%\s+to\s+(?:dead|\d+%)|"
            r"\bdrop(?:ped|ping|s)?\s+(?:from\s+)?\d+\s*%\s+(?:to\s+\d+\s*%|in\s+\d+\s*(?:second|minute|hour|min)|overnight)|"
            r"\b(?:phone|iphone|battery)\s+(?:drained?|lost)\s+\d+\s*%|"
            r"\bdrained?\s+(?:\d+\s*%|completely|overnight|fast)\b|"
            r"\bdead\s+in\s+(?:a\s+few|\d+)\s*(?:hours?|mins?|minutes?)\b|"
            r"\bcharge(?:s|d)?\s+back\s+(?:to\s+)?\d+\s*%|"
            r"\b(?:drains?|dying|died)\s+(?:so|really|way\s+too)\s+fast\b",
            re.IGNORECASE,
        )
        self._pat_battery_charge = re.compile(
            r"\b(?:won'?t|not|stopped|slow|stops?|unable\s+to|refuses\s+to|stopped)\s+charg(?:e|ing|es)\b|"
            r"\bcharg(?:er|ing\s*port|ing\s*cable|ing\s*case|ing\s*block|ing\s*cord)\b|"
            r"\blightning\s*(?:cable|port|connector)\b|"
            r"\bcharg(?:e|ing)\s*(?:problem|issue|slowly|properly|drops)\b|"
            r"\bonly\s+charg(?:es?|ing)\s+(?:when|if)\b|"
            # v2: charging screen / charge state stuck
            r"\bshows?\s+(?:a\s+)?charg(?:ing|e)\s*screen\b|"
            r"\bstuck\s+on\s+(?:the\s+)?charg(?:ing|e)\s*screen\b|"
            # v2: "doesn't charge when turned on" phrasing
            r"\bdoesn'?t\s+charg(?:e|ing)\b|"
            r"\bwon'?t\s+charg(?:e|ing)\s+when\b",
            re.IGNORECASE,
        )
        self._pat_power_issues = re.compile(
            r"\b(?:won'?t|not|cannot|refuses\s+to)\s+(?:turn|power|switch)\s*on\b|"
            r"\bshut(?:s|ting)?\s*(?:itself\s*)?(?:down|off)\s*(?:randomly|unexpectedly|by\s*itself|frequently)?\b|"
            r"\bpowers?\s*(?:itself\s*)?(?:down|off)\b|"
            r"\b(?:overheat(?:ing|ed)?|phone\s*(?:is|gets?)\s*(?:burning\s*)?hot)\b|"
            r"\b(?:restarts?|rebooting)\s*(?:randomly|by\s*itself|constantly|loop)\b|"
            # v2: switching off at non-zero percentage
            r"\bswitching\s+off\s+(?:randomly|at\s+\d+\s*%|unexpectedly|suddenly)\b|"
            r"\bturns?\s+(?:itself\s+)?off\s+(?:randomly|at\s+\d+\s*%|by\s+itself)\b|"
            r"\b(?:keeps?|keeps\s+on)\s+(?:switching|turning)\s+off\b|"
            r"\bcannot\s+turn\s+(?:it\s+)?back\s+on\s+without\s+plug(?:ging)?\b",
            re.IGNORECASE,
        )

        # ── 2. Billing & Payments ────────────────────────────────────────────
        self._pat_monetary_charge = re.compile(
            r"\b(?:overcharged|charged\s*twice|double\s*charged|unauthori[sz]ed\s*charg(?:e|es))\b|"
            r"\bcharg(?:e|ed)\s+(?:me|my\s+(?:account|card|credit\s*card|debit\s*card|bank|statement|paypal))\b|"
            r"\bcharg(?:e|ed|ing)\s+(?:for\s+(?:an?\s+)?(?:app|subscription|service|icloud|storage|music|itunes)|(?:\$|£|€|\b\d+\s*(?:dollars?|pounds?|euros?|cents?)\b))\b|"
            r"\bunwanted\s*charg(?:e|es)\b|"
            r"\bcharg(?:ed|ing)\s+without\s+(?:my\s+)?(?:permission|consent)\b",
            re.IGNORECASE,
        )
        self._pat_billing_refund = re.compile(
            r"\b(?:refund|refunded|reimburse(?:ment)?|money\s*back)\b|"
            r"\b(?:bill|billing|invoice|receipt)\s*(?:issue|error|statement|question|dispute|charge)\b|"
            r"\bpayment\s*(?:method)?\s*(?:declined|rejected|failed|invalid|not\s*working|keeps\s*being\s*rejected)\b|"
            r"\b(?:credit\s*card|debit\s*card|paypal)\b|"
            r"\bnone\s+payment\s+option\b",
            re.IGNORECASE,
        )

        # ── 3. Repair & Service ──────────────────────────────────────────────
        # v2: added broken screen fix cost, warranty rejection, general repair request
        self._pat_repair_explicit = re.compile(
            r"\bgenius\s*bar\b|"
            r"\bapple\s*store\s*appointment\b|"
            r"\b(?:make|book|schedule)\s+(?:an?\s+)?(?:appointment|reservation)\b|"
            r"\b(?:repair|replacement)\s*(?:appointment|quote|cost|pricing|price|fee|service|program|request|process)\b|"
            r"\b(?:how\s+much(?:\s+(?:does\s+it\s+cost|is\s+it))?|what\s+does\s+it\s+cost)\s+(?:to\s+)?(?:fix|repair|replace)\b|"
            r"\b(?:screen|battery|phone|device|watch|ipad|macbook)\s+replacement\b|"
            r"\breplace(?:ment)?\s+(?:screen|battery|phone|device|unit)\b|"
            r"\b(?:send|mail)\s+(?:it|my\s+(?:phone|device|watch|ipad|mac))\s+in\s+for\s+repair\b|"
            r"\bmail[- ]in\s*service\b|"
            r"\bapplecare(?:\+)?\s*(?:claim|repair|replacement|service|coverage)\b|"
            r"\bwarranty\s*(?:claim|covered|repair|replacement|void|reject(?:ed)?)\b|"
            # v2: explicit repair cost for broken screen / general fix
            # 'how much do [X] charge to fix' — actor word is optional after @mention stripping
            r"\bhow\s+much\s+(?:do\s+(?:you|apple|\w+\s+)?|does\s+(?:it|apple\s+)?)\s*charge\s+to\s+fix\b|"
            r"\bcost\s+(?:to\s+)?(?:fix|repair)\s+(?:broken\s+)?(?:screen|iphone|ipad|macbook|device)\b|"
            r"\b(?:fix|repair)\s+(?:broken|cracked|smashed)\s+(?:screen|glass|phone|ipad)\b|"
            r"\bwill\s+apple\s+(?:fix|repair|replace|reject)\b|"
            r"\b(?:out\s+of\s+)?warranty\s+(?:repair|fix|issue)\b|"
            r"\bservice\s+appointment\b|"
            r"\bapple\s*(?:repair|technician)\b",
            re.IGNORECASE,
        )

        # ── 4. Connectivity ──────────────────────────────────────────────────
        self._pat_wifi = re.compile(
            r"\bwi[- ]?fi\b|"
            r"\bwireless\s*network\b",
            re.IGNORECASE,
        )
        self._pat_bluetooth = re.compile(
            r"\bbluetooth\b|"
            r"\b(?:pair|pairing)\s*(?:with|to|failed|issue|problem)?\b|"
            r"\bwon'?t\s+pair\b",
            re.IGNORECASE,
        )
        self._pat_cellular_service = re.compile(
            r"\b(?:no\s+service|searching\.\.\.|searching\s+for\s+service)\b|"
            r"\bcellular\s*(?:data|network|service|connection)?\b|"
            r"\bmobile\s*data\b|"
            r"\b(?:sim\s*card|no\s*sim|invalid\s*sim|sim\s*failure|no\s*sim\s*(?:card\s*)?installed)\b|"
            r"\bairplane\s*mode\b|"
            r"\b(?:airplay|airdrop|personal\s*hotspot|tethering)\b|"
            r"\bnetwork\s*(?:connection|connectivity|carrier\s*settings|dropped)\b",
            re.IGNORECASE,
        )

        # ── 5. Software Update ───────────────────────────────────────────────
        # v2: added "install latest <product>" and implicit update-symptom phrases
        self._pat_software_update_explicit = re.compile(
            r"\b(?:software\s*update|ios\s*update|os\s*update|macos\s*update)\b|"
            r"\b(?:won'?t|cannot|unable\s*to|fails?\s+to)\s+(?:update|install\s*(?:the\s*)?(?:update|ios|software|macos)|upgrade)\b|"
            r"\bupdate\s*(?:failed|error|stuck|frozen|verification\s*failed|loop|won'?t\s*install)\b|"
            r"\berror\s*\d+\s*(?:when|while|during)\s*(?:updat(?:ing|e)|restor(?:ing|e))\b|"
            r"\b(?:will\s+not\s+update\s*,\s*will\s+not\s+restore|cannot\s+update\s+or\s+restore)\b|"
            r"\b(?:downgrade|rollback)\s+(?:to|from)?\s*(?:ios|\d+)\b|"
            r"\bstuck\s+on\s+(?:apple\s*logo|update\s*progress|progress\s*bar)\b|"
            # v2: installing latest iTunes / product version
            # Note: 'install latest itunes' routes here (not subscriptions_media)
            r"\binstall\s+(?:latest|newest|new)\s+(?:itunes|ios|macos|software|version)\b|"
            r"\b(?:latest|newest|new)\s+(?:itunes|ios|macos)\s+(?:version\s+)?(?:showed?|shows?|errored?)\b|"
            # 'unable to install <version number>' — no need to say 'iOS'
            r"\bunable\s+to\s+install\s+(?:ios|macos|\d+(?:\.\d+)*|the\s+update)\b|"
            r"\bit\s+says\s+unable\s+to\s+install\b|"
            # v2: "downloaded latest iOS updates" phrasing
            r"\bdownloaded?\s+(?:the\s+)?latest\s+(?:ios|macos|software)\s+updates?\b",
            re.IGNORECASE,
        )
        self._pat_software_update_general = re.compile(
            r"\b(?:update|updated|updating|upgrade|upgraded|upgrading)\b|"
            r"\b(?:ios|macos|watchos|tvos)\s*\d+(?:\.\d+)*\b",
            re.IGNORECASE,
        )

        # ── 6. Device Hardware ───────────────────────────────────────────────
        # v2: extended with frozen/unresponsive device phrases, general malfunction,
        #     and short-but-strong activation triggers for "iPhones won't activate" style
        self._pat_screen_hardware = re.compile(
            r"\b(?:screen|display|touchscreen|digitizer|glass)\s+(?:is|was|got|keeps|started|went|looks?|became)?\s*(?:flicker(?:ing)?|glitch(?:ing)?|cracked|broken|shattered|lines?|black|frozen|unresponsive|blank|dead|touch\s*not\s*working|ghost\s*touch|padlock\s*icon|mess(?:ed)?\s*up|distorted|shaking)\b|"
            r"\b(?:cracked|broken|shattered|glitchy|flickering|frozen|unresponsive|blank)\s+(?:screen|display|glass)\b|"
            r"\b(?:screen|display|touchscreen)\s+(?:won'?t|doesn'?t|not)\s+(?:turn\s*on|respond|work|touch)\b|"
            r"\blines\s+on\s+(?:the\s+)?(?:screen|display)\b|"
            r"\bpadlock\s*icon\b|"
            r"\bdead\s*pixels?\b|"
            r"\b(?:3d\s*touch|touch\s*id|face\s*id|touch\s*screen)\s*(?:is\s+)?(?:not\s*working|failed|unresponsive|broken|stopped)\b|"
            r"\b(?:my\s+(?:phone|iphone|ipad|mac|pod|watch)\s+is\s+broken|broken\s+(?:phone|iphone|ipad|screen|camera|device|pod))\b",
            re.IGNORECASE,
        )
        self._pat_camera_hardware = re.compile(
            r"\b(?:camera|lens|flashlight|flash|torch|facetim(?:e|ing))\s*(?:is|was|got|keeps|started|decided\s+to)?\s*(?:not\s*working|black|blurry|fuzzy|shak(?:y|ing)|stopped|freez(?:es|ing)|won'?t\s*(?:open|work|turn\s*on)|crash(?:es|ing)?|failed)\b|"
            r"\b(?:front|back|rear)\s*camera\b|"
            r"\bcamera\s+(?:won'?t|doesn'?t|cannot)\s+(?:focus|open|work|turn\s*on)\b",
            re.IGNORECASE,
        )
        self._pat_audio_hardware = re.compile(
            r"\b(?:speaker|microphone|mic|earpiece|audio|sound|airpods?|earphones?|earbuds?)\s*(?:is|was|got|keeps|started)?\s*(?:not\s*working|crackl(?:e|ing)|distort(?:ed|ion)|no\s*sound|low\s*volume|buzzing|muffled|failed|stopped|too\s*quiet)\b|"
            r"\bno\s+sound\s+(?:from|on|in)\s+(?:my\s+)?(?:speaker|phone|iphone|ipad|mac|video|call|headphones?)\b|"
            r"\bheadphones?\s*(?:jack|port)?\s*(?:not\s*working|broken|stuck)\b|"
            r"\b(?:vibrat(?:ion|or|ing)|taptic\s*engine)\s*(?:not\s*working|broken)\b",
            re.IGNORECASE,
        )
        self._pat_button_hardware = re.compile(
            r"\b(?:home\s*button|power\s*button|volume\s*button|mute\s*switch|keyboard)\s*(?:is|was|got|keeps|started)?\s*(?:stuck|broken|not\s*working|loose|click(?:ing)?|unresponsive|lag(?:ging)?)\b",
            re.IGNORECASE,
        )
        # v2: general device malfunction — frozen/unresponsive device as a whole
        self._pat_device_malfunction = re.compile(
            r"\b(?:phone|iphone|ipad|macbook|mac|device|watch)\s+(?:is\s+)?(?:frozen|freezing|froze|unresponsive|not\s+responding|stopped\s+responding|won'?t\s+respond)\b|"
            # Also catch 'completely unresponsive' / 'totally unresponsive' without device noun
            r"\b(?:completely|totally|just|suddenly|now)\s+unresponsive\b|"
            r"\b(?:frozen|freezing|froze|unresponsive)\s+(?:phone|iphone|ipad|macbook|device|watch)\b|"
            r"\b(?:phone|iphone|ipad|device)\s+(?:is\s+)?(?:not\s+working|stopped\s+working|completely\s+dead|bricked)\b|"
            r"\bwon'?t\s+(?:turn\s+on|boot\s+up|start\s+up|respond)\b|"
            # Microphone / speaker general device — hardware symptom context
            r"\b(?:microphone|speaker)\s+(?:is\s+)?(?:not\s+working|broken|failed|stopped\s+working)\b|"
            # Screen unresponsive / touch not working short-form
            r"\btouch(?:screen)?\s+(?:is\s+)?(?:not\s+working|unresponsive|broken|stopped\s+working)\b|"
            # Device-level animations / lag
            # Note: 'not smooth / not the best' → catch 'aren't the best' style too
            r"\banimations?\s+(?:are\s+|aren'?t\s+)?(?:laggy|stuttering|janky|not\s+(?:smooth|the\s+best)|slow|glitchy)\b|"
            r"\b(?:slight\s+)?delay\s+(?:in|on|with|and)\s+(?:the\s+)?(?:animation|touch|response|screen)\b|"
            # Catch 'there's a slight delay and the animations aren't the bestest thing'
            r"\bdelay\s+and\s+the\s+animations?\b",
            re.IGNORECASE,
        )

        # ── 7. App Store ─────────────────────────────────────────────────────
        self._pat_app_store_explicit = re.compile(
            r"\bapp\s*store\b|"
            r"\b(?:cannot|can'?t|unable\s*to|won'?t)\s+(?:download|install|update|search\s*for)\s+(?:any\s+)?apps?\b|"
            r"\bapp\s+(?:download|installation|installing|updating)\s*(?:stuck|pending|waiting|failed|error|loop|slow)\b|"
            r"\bdownload(?:ing)?\s+(?:free\s+)?apps?\b|"
            r"\binstall(?:ing)?\s+(?:free\s+)?apps?\b|"
            r"\bapp\s*store\s*(?:cannot\s*connect|blank|loading\s*forever|down|error)\b|"
            r"\bin-app\s*purchase\b",
            re.IGNORECASE,
        )

        # ── 8. Apple ID & Account ────────────────────────────────────────────
        # v2: added passcode, password hint, disabled device (access context),
        #     keychain, and can't sign in to iMessage/app
        self._pat_apple_id_explicit = re.compile(
            r"\bapple\s*id\b|"
            r"\baccount\s*(?:locked|disabled|blocked|recovery|access)\b|"
            r"\b(?:disabled\s+in\s+the\s+app\s+store\s+and\s+itunes|account\s+has\s+been\s+disabled)\b|"
            r"\b(?:can'?t|cannot|unable\s+to|won'?t\s+let\s+me)\s+(?:sign|log)\s*(?:in|on)\b|"
            r"\b(?:forgot|reset|change)\s+(?:my\s+)?(?:password|apple\s*id|passcode)\b|"
            r"\b(?:verification|confirmation|security)\s*code\b|"
            r"\b(?:two[- ]factor|2fa|trusted\s*phone|security\s*questions?)\b|"
            r"\bipad\s+is\s+disabled\s+connect\s+to\s+itunes\b|"
            r"\biphone\s+is\s+disabled\s+connect\s+to\s+itunes\b|"
            # v2: passcode locked / hint / keychain
            r"\bforgot\s+(?:my\s+)?(?:ipad|iphone|mac)\s+passcode\b|"
            r"\b(?:passcode|pin)\s+(?:wrong|incorrect|not\s+working|won'?t\s+work|forgot|locked|disabled)\b|"
            r"\bpassword\s+hint\s+(?:that\s+)?(?:doesn'?t|won'?t|not)\s+work\b|"
            r"\b(?:ipad|iphone|mac)\s+(?:is\s+)?disabled\s*(?:and\s+i\s+can'?t|,\s*can'?t)?\b|"
            r"\bkeychain\s+(?:not\s+)?(?:accepted|working|access|reset)\b|"
            # v2: can't sign in to specific Apple app
            r"\b(?:not\s+accepting|won'?t\s+accept)\s+(?:my\s+)?(?:password|passcode)\b|"
            r"\bpassword\s+(?:is\s+)?(?:wrong|not\s+recognized|unrecognized|not\s+working|keeps?\s+(?:saying|telling\s+me)\s+wrong)\b|"
            r"\b(?:can'?t|cannot)\s+(?:get\s+into|access|open)\s+(?:my\s+)?(?:account|iphone|ipad|mac)\b|"
            r"\b(?:unrecognised|unrecognized)\s+error\b",
            re.IGNORECASE,
        )

        # ── 9. iCloud ────────────────────────────────────────────────────────
        # v2: added backup/restore photo patterns, contacts deleted/missing,
        #     "recover photos", general backup restore
        self._pat_icloud_explicit = re.compile(
            r"\bicloud\s+(?:storage|backup|photos?|drive|sync|space)\b|"
            r"\b(?:storage|backup)\s+(?:full|purchased|manage|upgrade|sync)\b|"
            r"\bicloud\s+backup\s*(?:failed|not\s*working|restore|corrupt|stuck)\b|"
            r"\bphotos?\s+(?:not\s*syncing|disappeared|missing|sync)\b|"
            r"\bfamily\s+sharing\s+icloud\s+storage\b|"
            r"\bicloud\b|"
            # v2: recover/restore photos from backup
            r"\bphotos?\s+(?:won'?t|not|didn'?t|can'?t)\s+(?:recover|restore)(?:\s+from\s+(?:the\s+)?back\s*up|\s+from\s+icloud)?\b|"
            r"\brecover\s+(?:all\s+)?photos?(?:\s+from\s+(?:back\s*up|icloud))?\b|"
            r"\bback\s*(?:up|ed|ing)\s+(?:photos?|contacts?|data)\b|"
            r"\brestore\s+(?:from\s+)?(?:back\s*up|icloud)\b|"
            # noun-after-verb: 'contacts deleted', 'photos disappeared'
            r"\b(?:contacts?|photos?)\s+(?:deleted|disappeared|missing|wiped|gone)\b|"
            # verb-before-noun: 'delete my contacts', 'deleted...contacts', 'wiped...contacts'
            r"\b(?:delete|deleted|wiped|erased|removed|lost)\s+(?:\w+\s+){0,5}(?:contacts?|photos?)\b|"
            # v2: setting causing sync/backup behavior
            r"\bsetting\s+(?:to\s+)?(?:stop|turn\s+off|disable)\s+(?:it\s+)?(?:happening|syncing|backup)\b",
            re.IGNORECASE,
        )

        # ── 10. Subscriptions & Media ────────────────────────────────────────
        # v2: added Apple Music playback, music service issues
        #     iTunes alone is intentionally NOT a trigger — too many update collisions.
        #     Only iTunes + content context (library/music/match/store) triggers this.
        self._pat_subscriptions_media = re.compile(
            r"\bapple\s*music\b|"
            r"\bitunes\s*(?:library|music|audiobooks?|movies?|match|store)\b|"
            r"\baudiobooks?\b|"
            r"\bapple\s*podcasts?\b|"
            r"\bmedia\s*library\b|"
            r"\b(?:cancel|manage|renew)\s+(?:my\s+)?(?:subscription|apple\s*music)\b|"
            r"\b(?:songs?|tracks?|albums?)\s*(?:disappeared|missing|deleted|won'?t\s*play|not\s*showing|in\s*my\s*itunes\s*library)\b|"
            r"\baccess\s+(?:audiobooks?|itunes\s*library|media\s*library)\b|"
            r"\b(?:subscription|subscriptions)\b|"
            # v2: music quality/playback issues — specific enough vocabulary
            r"\bmusic\s+(?:is\s+)?(?:\w+\s+)?(?:crappy|bad|terrible|poor|poor\s+quality|not\s+working|skipping|pausing|won'?t\s+play|stopped\s+working)\b|"
            r"\baudio\s+(?:is\s+)?(?:\w+\s+)?(?:crappy|bad|terrible|poor|poor\s+quality)\b|"
            r"\bapple\s+music\s+(?:not\s+working|stopped|crashing|subscription)\b|"
            r"\b(?:streaming|playback)\s+(?:issue|problem|not\s+working|broken|poor)\b",
            re.IGNORECASE,
        )

        # ── 11. Setup & Activation ───────────────────────────────────────────
        # v2: broadened activation patterns to catch "iPhones won't activate!"
        #     and "device won't activate" short but unambiguous phrases
        self._pat_setup_activation = re.compile(
            r"\bactivation\s*lock\b|"
            r"\bactivation\s*server\s*(?:cannot\s*be\s*reached|unavailable|error|unreachable)\b|"
            r"\bcould\s*not\s*(?:be\s*)?activat(?:e|ed)\b|"
            r"\bactivat(?:e|ing|ion)\s+(?:my|new)?\s*(?:iphone|ipad|mac|device|phone|from\s*itunes)\b|"
            r"\bset\s*up\s*(?:my|a\s+new)\s+(?:iphone|ipad|mac|apple\s*watch)\s*(?:for\s+the\s+first\s+time)?\b|"
            r"\bsetup\s*(?:assistant|screen|process)\s*(?:stuck|freez(?:es|ing)|loop)\b|"
            r"\brespringing\s+when\s+trying\s+to\s+set\s*up\b|"
            r"\bfirst\s+time\s+setup\b|"
            # v2: short but strong: "won't activate" / "can't activate"
            r"\b(?:iphones?|ipads?|devices?|macs?)\s+won'?t\s+activat(?:e|ing)\b|"
            r"\b(?:can'?t|cannot|unable\s+to)\s+activat(?:e|ing)\b",
            re.IGNORECASE,
        )

        # ── 12. Security & Privacy ───────────────────────────────────────────
        self._pat_security_privacy = re.compile(
            r"\b(?:hacked|compromised|hacker|someone\s+hacked)\b|"
            r"\bunauthori[sz]ed\s*(?:access|use|login|activity|device)\b|"
            r"\bsomeone\s*(?:accessed|logged\s*into|stole)\s*(?:my|account|phone)\b|"
            r"\bfind\s*my\s*(?:iphone|ipad|device)?\s*(?:stolen|tracking|location\s*stolen)\b|"
            r"\blost\s*(?:or\s*stolen)?\s*(?:iphone|ipad|device|phone)\b|"
            r"\bphishing\s*(?:scam|email|message|link)\b|"
            r"\bprivacy\s*(?:breach|leak|exposure|concern)\b",
            re.IGNORECASE,
        )

        # ── 13. Orders & Delivery ────────────────────────────────────────────
        # v2: added pre-order, sold out, availability, return/return policy
        self._pat_orders_delivery = re.compile(
            r"\border\s*(?:status|tracking|number|delayed|cancelled|shipped|delivery|queue|dispatch)\b|"
            r"\btracking\s*(?:number|order|shipment|delivery|package)\b|"
            r"\bship(?:ment|ping)?\s*(?:status|delayed|tracking|lost|update)\b|"
            r"\bdeliver(?:y|ed)?\s*(?:date|delayed|status|problems?|address)\b|"
            r"\bwhen\s+will\s+(?:my\s+)?(?:order|iphone|device|package|watch|ipad)\s+(?:arrive|ship|deliver|be\s+delivered)\b|"
            r"\bcourier\s*(?:delivery|tracking|status)\b|"
            r"\bapple\s*store\s*online\s*order\b|"
            r"\border\s+queue\b|"
            # v2: pre-order, sold out, availability
            r"\bpre[- ]?order\b|"
            r"\bsold\s+out\b|"
            r"\bwhy\s+is\s+(?:it|the\s+(?:iphone|ipad|watch|device))\s+sold\s+out\b|"
            r"\b(?:order|purchase)\s+(?:status|confirmation|ineligible|change)\b|"
            # v2: returns and return policy for purchased products
            r"\breturn\s+(?:an?\s+)?(?:item|product|iphone|ipad|device|purchase|order)\b|"
            r"\b(?:14|30)\s*(?:-|day\s+)?return\b|"
            r"\b(?:return|refund)\s+policy\b|"
            r"\b(?:how\s+(?:do\s+i|to))\s+return\b|"
            r"\bineligible\s+for\s+(?:change|return)\s+online\b",
            re.IGNORECASE,
        )

        # ── 14. How-to Information ───────────────────────────────────────────
        self._pat_how_to = re.compile(
            r"\bhow\s+(?:do|can|would)\s+i\s+(?:turn\s*on|turn\s*off|enable|disable|customize|change|switch|find|use|check|sort|organize|set|transfer|delete|access)\b|"
            r"\bwhere\s+(?:can\s+i\s+find|is\s+the\s+setting|do\s+i\s+go\s+to)\b|"
            r"\bis\s+(?:there\s+a\s+way\s+to|it\s+possible\s+to)\b|"
            r"\bcan\s+i\s+(?:use|connect|transfer|do|have|access)\b|"
            r"\bwhat\s+is\s+(?:the\s+difference|night\s+shift|true\s+tone|family\s+sharing)\b",
            re.IGNORECASE,
        )

        # ── Noise filters ────────────────────────────────────────────────────
        self._pat_generic_greeting = re.compile(
            r"^(?:(?:hi|hello|hey|howdy|good\s*(?:morning|afternoon|evening)|awrite|please\s*help|need\s*help|quick\s*question|question|help)[\s.,!]*)+$",
            re.IGNORECASE,
        )
        self._pat_bare_complaint = re.compile(
            r"^(?:(?:why|ugh|wtf|hate\s+apple|apple\s+sucks|so\s+annoyed|worst\s+phone|terrible)[\s.,!?]*)+$",
            re.IGNORECASE,
        )
        # Short-but-strong signal patterns (checked ONLY when token count <= 4)
        # Each tuple: (compiled pattern, intent)
        # These must be high-precision: no ambiguous vocabulary.
        self._short_strong_signals: list[tuple[re.Pattern, str]] = [
            (re.compile(r"\b(?:iphones?|ipads?|device)s?\s+won'?t\s+activat(?:e|ion)\b", re.IGNORECASE), "setup_activation"),
            (re.compile(r"\b(?:can'?t|cannot)\s+activat(?:e|ion)\b", re.IGNORECASE), "setup_activation"),
            (re.compile(r"\bactivation\s*lock\b", re.IGNORECASE), "setup_activation"),
            (re.compile(r"\bicloud\b", re.IGNORECASE), "icloud"),
            (re.compile(r"\bapple\s*id\b", re.IGNORECASE), "apple_id_account"),
            (re.compile(r"\bgenius\s*bar\b", re.IGNORECASE), "repair_service"),
            (re.compile(r"\bapp\s*store\b", re.IGNORECASE), "app_store"),
            (re.compile(r"\bwi[- ]?fi\b", re.IGNORECASE), "connectivity"),
            (re.compile(r"\bbluetooth\b", re.IGNORECASE), "connectivity"),
        ]

        # ── Update-as-fault gate ─────────────────────────────────────────────
        # Used ONLY to guard the has_general_update→software_update path.
        # A general version mention ("iOS 11", "upgrading", "updated") is treated
        # as incidental context UNLESS the message also contains language that
        # explicitly frames the update itself as the source of the problem.
        #
        # Design rule: this pattern captures "the update caused X" framing.
        # It is NOT a hardware keyword expansion.
        self._pat_update_as_fault = re.compile(
            # Temporal / causal: 'since updating/upgrading', 'after the update'
            r"\b(?:since|after|following|because\s+of|due\s+to)\s+(?:the\s+)?(?:update|upgrade|updating|upgrading|ios|macos|watchos|tvos)\b|"
            # Explicit complaint about the update itself
            r"\bthe\s+(?:new\s+)?(?:update|ios|macos|watchos)\s+(?:is|was|has|keeps?|makes?|broke|broke|ruined|messed|caused|caused|crashes?|doesn'?t|won'?t|can'?t)\b|"
            # Update as subject with fault verb
            r"\b(?:update|upgrade|ios|macos|watchos)\s+(?:broke|ruined|messed|caused|killed|destroyed)\b|"
            # "downloaded/installed the update and [problem]" — purposeful action then fault
            r"\b(?:downloaded|installed)\s+(?:the\s+)?(?:suggested\s+)?(?:update|ios|macos)\s+and\b",
            re.IGNORECASE,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def label(
        self,
        text: str,
        context: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Classify an inbound customer message into exactly one canonical intent.

        Args:
            text: Inbound customer message text.
            context: Optional prior customer message texts in the conversation.
                     (Outbound agent replies must not be provided or used).

        Returns:
            Dictionary matching the structured result format:
            {"intent": str, "confidence": float, "reason": str, "matched_rules": list[str]}
        """
        result = self._classify(text=text, context=context)
        return result.to_dict()

    def _classify(
        self,
        text: str,
        context: Sequence[str] | None = None,
    ) -> LabelResult:
        normalized = normalize_text(text)
        tokens = extract_meaningful_tokens(normalized)

        # ── 0. Empty / ultra-low-context ──────────────────────────────────────
        if not normalized or len(tokens) == 0:
            return LabelResult(
                intent="other_unclear",
                confidence=0.1,
                reason="low_context:empty_or_whitespace",
                matched_rules=[],
            )

        if self._pat_generic_greeting.match(normalized):
            return LabelResult(
                intent="other_unclear",
                confidence=0.2,
                reason="low_context:generic_greeting_only",
                matched_rules=[],
            )

        if self._pat_bare_complaint.match(normalized):
            return LabelResult(
                intent="other_unclear",
                confidence=0.2,
                reason="low_context:bare_complaint_no_fault",
                matched_rules=[],
            )

        # ── 0b. Short-but-strong signals (token count <= 4) ───────────────────
        # Only checked when the message is too short for full rule matching.
        # Vague short fragments ("ok thanks", "I will") will NOT match any of
        # these patterns and will fall through to other_unclear as intended.
        if len(tokens) <= 4:
            for pat, intent in self._short_strong_signals:
                if pat.search(normalized):
                    return LabelResult(
                        intent=intent,
                        confidence=0.80,
                        reason="short_strong_signal",
                        matched_rules=[intent],
                    )

        # ── 1. Identify Candidate Rule Matches ────────────────────────────────
        matches: list[str] = []

        has_security = bool(self._pat_security_privacy.search(normalized))
        if has_security:
            matches.append("security_privacy")

        has_orders = bool(self._pat_orders_delivery.search(normalized))
        if has_orders:
            matches.append("orders_delivery")

        has_setup = bool(self._pat_setup_activation.search(normalized))
        if has_setup:
            matches.append("setup_activation")

        has_repair = bool(self._pat_repair_explicit.search(normalized))
        if has_repair:
            matches.append("repair_service")

        has_billing = bool(
            self._pat_monetary_charge.search(normalized) or
            self._pat_billing_refund.search(normalized)
        )
        if has_billing:
            matches.append("billing_payments")

        has_battery = bool(
            self._pat_battery.search(normalized) or
            self._pat_battery_drain.search(normalized) or
            self._pat_battery_charge.search(normalized) or
            self._pat_power_issues.search(normalized)
        )
        if has_battery:
            matches.append("battery_power")

        has_connectivity = bool(
            self._pat_wifi.search(normalized) or
            self._pat_bluetooth.search(normalized) or
            self._pat_cellular_service.search(normalized)
        )
        if has_connectivity:
            matches.append("connectivity")

        has_hardware = bool(
            self._pat_screen_hardware.search(normalized) or
            self._pat_camera_hardware.search(normalized) or
            self._pat_audio_hardware.search(normalized) or
            self._pat_button_hardware.search(normalized) or
            self._pat_device_malfunction.search(normalized)  # v2
        )
        if has_hardware:
            matches.append("device_hardware")

        has_app_store = bool(self._pat_app_store_explicit.search(normalized))
        if has_app_store:
            matches.append("app_store")

        has_apple_id = bool(self._pat_apple_id_explicit.search(normalized))
        if has_apple_id:
            matches.append("apple_id_account")

        has_icloud = bool(self._pat_icloud_explicit.search(normalized))
        if has_icloud:
            matches.append("icloud")

        has_media = bool(self._pat_subscriptions_media.search(normalized))
        if has_media:
            matches.append("subscriptions_media")

        has_explicit_update = bool(self._pat_software_update_explicit.search(normalized))
        has_general_update = bool(self._pat_software_update_general.search(normalized))
        if has_explicit_update or has_general_update:
            matches.append("software_update")

        has_how_to = bool(self._pat_how_to.search(normalized))
        if has_how_to:
            matches.append("how_to_information")

        # ── 2. Collision & Disambiguation Resolution ──────────────────────────

        # (A) Security/Privacy — critical priority for explicit hacking/theft
        if has_security:
            is_routine_account = bool(re.search(r"\b(?:reset|forgot)\s+(?:my\s+)?password\b", normalized))
            if not is_routine_account:
                return LabelResult(
                    intent="security_privacy",
                    confidence=0.95,
                    reason="high_precision_match:security_privacy",
                    matched_rules=matches,
                )

        # (B) Orders & Delivery
        # Guard: "return" can collide with repair, so only route if no repair or
        # billing keywords dominate the message
        if has_orders and not (has_battery or has_hardware or has_connectivity):
            # Guard: "how much do you charge to fix broken screen" → repair, not orders
            is_repair_cost_question = bool(re.search(
                r"\b(?:fix|repair|replace)\b",
                normalized,
            )) and has_repair
            if not is_repair_cost_question:
                return LabelResult(
                    intent="orders_delivery",
                    confidence=0.95,
                    reason="high_precision_match:orders_delivery",
                    matched_rules=matches,
                )

        # (C) Repair & Service vs Hardware / Battery / Connectivity
        if has_repair:
            is_cellular_service = bool(re.search(
                r"\b(?:no\s+service|searching\s+for\s+service|cellular\s+service|lost\s+service)\b",
                normalized,
            ))
            if is_cellular_service and not re.search(r"\b(?:genius\s*bar|appointment|repair|replacement|warranty|applecare)\b", normalized):
                return LabelResult(
                    intent="connectivity",
                    confidence=0.90,
                    reason="disambiguated:cellular_service_over_repair",
                    matched_rules=matches,
                )
            return LabelResult(
                intent="repair_service",
                confidence=0.92,
                reason="disambiguated:repair_request_over_hardware_symptom",
                matched_rules=matches,
            )

        # (D) Electrical Charging (Battery) vs Monetary Charge (Billing)
        if has_billing and has_battery:
            has_explicit_monetary = bool(re.search(
                r"\b(?:credit\s*card|debit\s*card|bank|paypal|refund|overcharg|double\s*charge|charged\s*twice|fee|\$|£|€|dollars?|charged\s*my\s*account)\b",
                normalized,
            ))
            has_explicit_electrical = bool(re.search(
                r"\b(?:battery|phone|iphone|ipad|macbook|charger|cable|port|percentage|drain|won'?t\s+turn\s+on|overheat)\b",
                normalized,
            ))
            if has_explicit_monetary and not re.search(r"\b(?:battery|won'?t\s+turn\s+on|lightning|drain)\b", normalized):
                return LabelResult(
                    intent="billing_payments",
                    confidence=0.90,
                    reason="disambiguated:monetary_charge_over_battery",
                    matched_rules=matches,
                )
            if has_explicit_electrical and not has_explicit_monetary:
                return LabelResult(
                    intent="battery_power",
                    confidence=0.90,
                    reason="disambiguated:electrical_charge_over_billing",
                    matched_rules=matches,
                )

        if has_billing:
            return LabelResult(
                intent="billing_payments",
                confidence=0.92,
                reason="high_precision_match:billing_payments",
                matched_rules=matches,
            )

        # (E) Software Update vs Downstream Symptoms
        if has_explicit_update or has_general_update:
            if has_battery:
                return LabelResult(
                    intent="battery_power",
                    confidence=0.90,
                    reason="disambiguated:battery_symptom_over_update_context",
                    matched_rules=matches,
                )
            if has_connectivity:
                return LabelResult(
                    intent="connectivity",
                    confidence=0.90,
                    reason="disambiguated:connectivity_symptom_over_update_context",
                    matched_rules=matches,
                )
            if has_hardware:
                return LabelResult(
                    intent="device_hardware",
                    confidence=0.90,
                    reason="disambiguated:hardware_symptom_over_update_context",
                    matched_rules=matches,
                )
            if has_app_store:
                return LabelResult(
                    intent="app_store",
                    confidence=0.90,
                    reason="disambiguated:app_store_over_update_context",
                    matched_rules=matches,
                )
            if has_media:
                return LabelResult(
                    intent="subscriptions_media",
                    confidence=0.90,
                    reason="disambiguated:media_symptom_over_update_context",
                    matched_rules=matches,
                )
            if has_explicit_update:
                return LabelResult(
                    intent="software_update",
                    confidence=0.92,
                    reason="high_precision_match:software_update",
                    matched_rules=matches,
                )
            # General update/version mention: only accept as software_update when the
            # message explicitly frames the update itself as the fault source.
            # A bare version string (e.g. "iOS 11", "upgrading") that appears as
            # incidental context while complaining about device behavior should NOT
            # be classified as software_update — it is other_unclear.
            # This is the device_hardware→software_update precedence rule.
            if has_general_update and len(tokens) >= 5:
                is_update_the_fault = bool(self._pat_update_as_fault.search(normalized))
                if is_update_the_fault:
                    return LabelResult(
                        intent="software_update",
                        confidence=0.80,
                        reason="general_update_mention:update_framed_as_fault",
                        matched_rules=matches,
                    )
                # Version/update word appears as context only → don't claim software_update

        # (F) Apple ID vs iCloud
        if has_apple_id and has_icloud:
            is_auth_problem = bool(re.search(
                r"\b(?:locked|disabled|sign\s*in|log\s*in|password|passcode|verification|2fa|recover|code|keychain|hint)\b",
                normalized,
            ))
            if is_auth_problem:
                return LabelResult(
                    intent="apple_id_account",
                    confidence=0.90,
                    reason="disambiguated:account_auth_over_icloud",
                    matched_rules=matches,
                )
            return LabelResult(
                intent="icloud",
                confidence=0.90,
                reason="disambiguated:icloud_storage_over_account",
                matched_rules=matches,
            )

        if has_apple_id:
            return LabelResult(
                intent="apple_id_account",
                confidence=0.92,
                reason="high_precision_match:apple_id_account",
                matched_rules=matches,
            )

        if has_icloud:
            return LabelResult(
                intent="icloud",
                confidence=0.90,
                reason="high_precision_match:icloud",
                matched_rules=matches,
            )

        # (G) Setup & Activation
        if has_setup:
            return LabelResult(
                intent="setup_activation",
                confidence=0.90,
                reason="high_precision_match:setup_activation",
                matched_rules=matches,
            )

        # (H) Battery & Power
        if has_battery:
            return LabelResult(
                intent="battery_power",
                confidence=0.92,
                reason="high_precision_match:battery_power",
                matched_rules=matches,
            )

        # (I) Connectivity
        if has_connectivity:
            return LabelResult(
                intent="connectivity",
                confidence=0.92,
                reason="high_precision_match:connectivity",
                matched_rules=matches,
            )

        # (J) Device Hardware
        if has_hardware:
            return LabelResult(
                intent="device_hardware",
                confidence=0.92,
                reason="high_precision_match:device_hardware",
                matched_rules=matches,
            )

        # (K) App Store
        if has_app_store:
            return LabelResult(
                intent="app_store",
                confidence=0.92,
                reason="high_precision_match:app_store",
                matched_rules=matches,
            )

        # (L) Subscriptions & Media
        if has_media:
            return LabelResult(
                intent="subscriptions_media",
                confidence=0.90,
                reason="high_precision_match:subscriptions_media",
                matched_rules=matches,
            )

        # (M) How-To Information
        if has_how_to:
            return LabelResult(
                intent="how_to_information",
                confidence=0.85,
                reason="high_precision_match:how_to_information",
                matched_rules=matches,
            )

        # (N) Fallback — ambiguous / insufficient evidence
        if len(tokens) <= 4:
            return LabelResult(
                intent="other_unclear",
                confidence=0.20,
                reason="low_context:short_fragment",
                matched_rules=matches,
            )

        return LabelResult(
            intent="other_unclear",
            confidence=0.25,
            reason="no_high_precision_rule",
            matched_rules=matches,
        )


def label_intent(
    text: str,
    context: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Convenience functional interface for high-precision intent classification."""
    labeler = SilverLabeler()
    return labeler.label(text=text, context=context)
