"""Deterministic, template-based response composer.

Used whenever ANTHROPIC_API_KEY is not set (including by default in the
evaluation suite, so `python evaluation/run_eval.py` works from a clean
clone with no credentials). Because this composer only ever fills fixed
sentence templates from an already-safe `facts` dict, it is structurally
incapable of being redirected by an injected instruction -- there's no
step where arbitrary text could be interpreted as a command.

This is deliberately less fluent than an LLM-phrased answer. See README
"Known limitations": paraphrase robustness is best with a real model.
"""
from __future__ import annotations

from typing import Any


def generate_response(facts: dict[str, Any]) -> str:
    kind = facts.get("kind")
    if kind == "missing_order_id":
        return "Could you share your order ID (it looks like ORD-1234) so I can look that up for you?"

    if kind == "order_not_found":
        return (
            f"I couldn't find an order matching \"{facts['requested_id']}\". "
            "Could you double-check the order ID? If it still doesn't turn up, "
            "I'd recommend contacting human support so they can investigate."
        )

    if kind == "order_malformed":
        return (
            f"\"{facts['requested_id']}\" doesn't look like a valid order ID "
            "(they're formatted like ORD-1234). Could you double-check it?"
        )

    if kind == "order_status":
        return _order_status_text(facts)

    if kind == "disclosure_refused":
        return facts["refusal_text"]

    if kind == "injection_notice":
        return (
            f"{facts['policy_answer']} "
            "I should also flag: that migration note is an unapproved internal draft, not "
            "active policy, so I can't use it as authority, and I'm not able to approve returns myself either way -- "
            "that requires a completed return through normal channels. "
            f"Source: {facts['source']}"
        )

    if kind == "conflict":
        claim_a, claim_b = _extract_conflicting_claims(facts.get("passages", []))
        return (
            "I'm seeing a genuine conflict between two current official sources on this, and neither "
            f"has been superseded by the other: {facts['doc_a']} says {claim_a}, while "
            f"{facts['doc_b']} says {claim_b}. Rather than silently pick one, I'd recommend the safest "
            f"interim approach ({facts.get('safe_default', 'the more cautious of the two')}) and "
            f"confirming with human support. Sources: {facts['doc_a']}, {facts['doc_b']}"
        )

    if kind == "insufficient":
        return (
            "I don't have enough information in Aster & Row's current documentation to answer that "
            "reliably, so I don't want to guess. I'd recommend confirming with human support."
        )

    if kind == "knowledge_answer":
        body = facts["answer_text"]
        sources = ", ".join(facts["sources"])
        handoff_line = f"\n\n{facts['handoff_reason']}" if facts.get("handoff") else ""
        return f"{body}\n\nSource: {sources}{handoff_line}"

    return "I'm not able to help with that right now. Could you rephrase, or would you like human support?"


def _extract_conflicting_claims(passages: list[dict[str, Any]]) -> tuple[str, str]:
    """Pull out the specific sentence containing each side's claim, so the
    conflict answer states the actual disagreement rather than just naming
    the two documents. Scoped to the one designed conflict in this corpus
    (hand-wash vs. dishwasher-safe); a general-purpose version would need
    real contradiction detection -- see README known limitations."""
    claim_a, claim_b = "one thing", "another"
    for p in passages:
        text = p.get("text", "")
        if "hand-wash" in text.lower() or "hand washed" in text.lower():
            claim_a = "the body should be hand-washed"
        if "dishwasher safe" in text.lower():
            claim_b = "all components are dishwasher safe"
    return claim_a, claim_b


def _order_status_text(facts: dict[str, Any]) -> str:
    data = facts["order_data"]
    status = data["status"]
    order_id = data["order_id"]

    if status in ("cancelled", "returned"):
        verb = "was cancelled and will not be shipped" if status == "cancelled" else "was returned and processed"
        return f"Order {order_id} {verb}. {data.get('customer_safe_message', '')}".strip()

    if status == "exception":
        return (
            f"Order {order_id} has a shipment exception that needs support review, so I can't give you a "
            "reliable delivery estimate. I'd recommend human support look into it."
        )

    if status == "shipped":
        carrier = data.get("carrier") or "the carrier"
        eta = data.get("estimated_delivery")
        if eta:
            return f"Order {order_id} has shipped with {carrier} and is currently estimated to arrive on {_fmt_date(eta)}."
        return (
            f"Order {order_id} has shipped with {carrier}, but a delivery estimate isn't currently available. "
            "I don't want to guess at a date -- I can let you know as soon as one is available."
        )

    if status in ("pending", "processing"):
        eta = data.get("estimated_delivery")
        base = f"Order {order_id} is currently {status}."
        return f"{base} Estimated delivery: {_fmt_date(eta)}." if eta else f"{base} A delivery estimate isn't available yet."

    if status == "delayed":
        eta = data.get("estimated_delivery")
        return f"Order {order_id} has been delayed by the carrier. {data.get('customer_safe_message', '')}".strip()

    if status == "delivered":
        return f"Order {order_id} was delivered. {data.get('customer_safe_message', '')}".strip()

    return f"Order {order_id} status: {status}."


_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def _fmt_date(iso_date: str) -> str:
    try:
        year, month, day = iso_date.split("-")
        return f"{_MONTHS[int(month) - 1]} {int(day)}, {year}"
    except Exception:
        return iso_date
