"""Lightweight unit tests -- no pytest dependency required.

Run with:  python tests/test_core.py
(or `pytest tests/` if pytest happens to be installed; these are plain
functions named test_* so pytest will discover them too.)
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import Agent
from app.orders import OrderStore
from app.retriever import _stem

KB_DIR = ROOT / "knowledge-base"
ORDERS_PATH = ROOT / "data" / "orders.json"


def test_order_redaction_never_leaks_pii():
    store = OrderStore(ORDERS_PATH)
    result = store.lookup("ORD-1007")
    assert result.found
    data = result.data
    for forbidden in ("customer", "internal", "email", "shipping_address", "risk_score"):
        assert forbidden not in data, f"{forbidden!r} leaked into redacted order data"
    assert data["order_id"] == "ORD-1007"


def test_order_id_normalization():
    store = OrderStore(ORDERS_PATH)
    variants = ["ord-1007", "  ORD-1007  ", "Ord-1007.", "ORD-1007"]
    for v in variants:
        result = store.lookup(v)
        assert result.found, f"failed to normalize {v!r}"
        assert result.data["order_id"] == "ORD-1007"


def test_cancelled_order_suppresses_stale_eta():
    store = OrderStore(ORDERS_PATH)
    result = store.lookup("ORD-1004")
    assert result.found
    assert result.data["status"] == "cancelled"
    assert result.data["estimated_delivery"] is None, "stale ETA should be suppressed for cancelled orders"


def test_unknown_order_reports_not_found():
    store = OrderStore(ORDERS_PATH)
    result = store.lookup("ORD-9999")
    assert not result.found
    assert result.error == "not_found"


def test_malformed_order_id_reports_malformed():
    store = OrderStore(ORDERS_PATH)
    result = store.lookup("ORD-12345")  # 5 digits, not the expected 4
    assert not result.found
    assert result.error == "malformed"


def test_stemmer_unifies_ship_variants():
    assert _stem("shipping") == _stem("ships") == _stem("shipped") == "ship"


def test_sessions_are_isolated():
    agent = Agent(kb_dir=str(KB_DIR), orders_path=str(ORDERS_PATH))
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    agent.handle(session_a, "Where is ORD-1007?")
    # Session B never mentioned an order; a bare follow-up here must NOT
    # inherit session A's order focus.
    result_b = agent.handle(session_b, "When will it arrive?")
    assert result_b.tool_called is None, "session B incorrectly inherited session A's order focus"


def test_session_followup_within_same_session_works():
    agent = Agent(kb_dir=str(KB_DIR), orders_path=str(ORDERS_PATH))
    session_id = str(uuid.uuid4())
    agent.handle(session_id, "Where is ORD-1007?")
    result = agent.handle(session_id, "When will it arrive?")
    assert result.tool_called == "order_lookup"
    assert result.tool_arguments == {"order_id": "ORD-1007"}


def _run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} unit tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    _run_all()
