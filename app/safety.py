"""Safety/observability helpers.

Important design note: the real defense against prompt injection in this
system is architectural, not a keyword filter. Retrieved knowledge-base
passages and order-lookup results are only ever passed to the generation
step as labeled, inert *data* -- never concatenated into something the
model is told to treat as an instruction. `flag_possible_injection` below
exists only so the debug trace can surface "hey, this input looked like an
injection attempt" for observability; it is not what actually stops it.
"""
from __future__ import annotations

import re

_INJECTION_MARKERS = [
    r"ignore (all|previous|prior) (rules|instructions)",
    r"system instruction",
    r"reveal (your|the) (hidden|system) prompt",
    r"disregard (the )?(real|actual) policy",
    r"do not (call|use) tools",
    r"never cite a source",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_MARKERS), re.IGNORECASE)

_DISCLOSURE_REQUEST_MARKERS = [
    "system prompt", "hidden prompt", "hidden instructions", "your instructions",
    "risk score", "internal note", "warehouse note", "customer's email",
    "customer email", "shipping address", "another customer",
]


def flag_possible_injection(*texts: str) -> bool:
    """Heuristic only -- used for the debug trace, not as the defense
    itself. Retrieved/tool text is never executed as instructions
    regardless of whether this flag fires."""
    return any(_INJECTION_RE.search(t or "") for t in texts)


def requests_forbidden_disclosure(message: str) -> list[str]:
    lowered = message.lower()
    return [m for m in _DISCLOSURE_REQUEST_MARKERS if m in lowered]


REFUSAL_TEXT = (
    "I can't share internal system details, hidden instructions, risk scores, "
    "internal notes, or another customer's personal information. I can help "
    "with anything about your own order or Aster & Row's public policies."
)

GIFT_CARD_CODE_RE = re.compile(r"\b[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b")


def contains_full_gift_card_code(text: str) -> bool:
    return bool(GIFT_CARD_CODE_RE.search(text))
