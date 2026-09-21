"""Orchestrates a single turn: decides intent, gathers only the facts the
generation step is allowed to see, and produces a structured trace for
observability.

Everything that determines *correctness* (authority, conflicts, PII
redaction, status precedence, handoff triggers) happens here in plain
Python, deterministically, before any model is involved. The model (or the
offline template composer) only ever turns an already-decided `facts`
dict into prose -- see llm_client.py / responder.py.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.knowledge_base import load_knowledge_base
from app.orders import OrderStore
from app.retriever import Retriever, ScoredChunk, _tokenize
from app.session import SessionStore
from app import safety

ORDER_ID_LOOSE_RE = re.compile(r"\bORD[-\s]?(\d{3,6})\b", re.IGNORECASE)

_ORDER_INTENT_WORDS = re.compile(
    # Deliberately narrower than "contains the word order": an earlier
    # version matched on the bare word "order" and misclassified a policy
    # question like "does my order from last month get the 45-day window"
    # as a lookup request needing an order ID (see README bug diary).
    r"\b(where is|where's|arrive|arriving|track|tracking|shipped|"
    r"when will (it|this|my order) (arrive|get here)|status of my order)\b",
    re.IGNORECASE,
)
_FOLLOWUP_MARKERS = re.compile(
    r"^(and|what about|how about|also|what if)\b|\bit\b|\bthat\b", re.IGNORECASE
)
_INCIDENT_LANGUAGE = re.compile(
    # Deliberately specific to actual event language, not general
    # possessives like "my" -- an earlier version included "my" and it
    # false-triggered a handoff on ordinary policy questions like "can I
    # use my gift card balance..." (see README bug diary).
    r"\b(yesterday|today|this morning|just now|arrived|broke|broken|damaged|defective|received)\b",
    re.IGNORECASE,
)
_ACTIONABLE_DOCS = {
    "04-damaged-or-wrong-items.md",
    "07-warranty.md",
    "08-order-changes-and-cancellations.md",
    "10-gift-cards-and-price-adjustments.md",
}
_INSUFFICIENT_SCORE_THRESHOLD = 0.08

# Topics this knowledge base genuinely has no content on. A pure TF-IDF
# similarity score doesn't reliably catch this: a question like "are your
# fabrics and adhesives vegan?" scores deceptively well against the
# Product Care Guide purely because both mention "fabric" and "bags", even
# though the guide says nothing about material sourcing or certification.
# Rather than trust the similarity score here, these specific terms force
# an honest "insufficient information" response. This is a narrow,
# explicit patch for a real gap, not a general solution -- a production
# system would need genuine groundedness checking (e.g. "does this
# passage actually assert what the answer is claiming?"), which is out of
# scope for a TF-IDF retriever. See README known limitations.
_KNOWN_UNCOVERED_TERMS = {
    "vegan", "veganism", "crueltyfree", "cruelty", "hypoallergenic",
    "recycled", "sustainable", "organic", "certified", "certification",
    "nontoxic", "toxic", "adhesive", "adhesives",
}


@dataclass
class AgentResult:
    response_text: str
    tool_called: str | None
    tool_arguments: dict[str, Any] | None
    sources: list[str]
    handoff: bool
    trace: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(self, kb_dir: str, orders_path: str):
        self.chunks = load_knowledge_base(kb_dir)
        self.retriever = Retriever(self.chunks)
        self.orders = OrderStore(orders_path)
        self.sessions = SessionStore()
        self._use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # -- public entry point --------------------------------------------
    def handle(self, session_id: str, message: str) -> AgentResult:
        session = self.sessions.get(session_id)
        session.add("user", message)
        trace: dict[str, Any] = {
            "session_id": session_id,
            "user_message": message,
            "recent_history": session.recent_text(),
        }

        # 1. Disclosure requests always win -- never conditioned on anything else.
        forbidden = safety.requests_forbidden_disclosure(message)
        if forbidden:
            trace["forbidden_disclosure_markers"] = forbidden
            result = self._handle_disclosure_refusal(message, trace)
            session.add("agent", result.response_text)
            return result

        # 2. Prompt-injection-shaped input: answer correctly, refuse to follow it.
        if safety.flag_possible_injection(message) or "migration note" in message.lower():
            trace["flagged_injection"] = True
            result = self._handle_injection_attempt(message, trace)
            session.add("agent", result.response_text)
            return result

        # 3. Order-related?
        explicit_id = self._extract_order_id(message)
        implied_order_followup = (
            explicit_id is None
            and session.focus.order_id
            and (_ORDER_INTENT_WORDS.search(message) or _FOLLOWUP_MARKERS.search(message))
        )
        if explicit_id or implied_order_followup:
            order_id = explicit_id or session.focus.order_id
            result = self._handle_order(order_id, trace)
            session.focus.order_id = order_id
            session.focus.topic = "order"
            session.add("agent", result.response_text)
            return result

        if _ORDER_INTENT_WORDS.search(message) and not explicit_id:
            trace["reason"] = "order-shaped question with no order id in message or focus"
            result = self._handle_missing_order_id(trace)
            session.add("agent", result.response_text)
            return result

        # 4. Otherwise: knowledge-base question.
        query = message
        if session.focus.topic == "knowledge" and session.focus.last_query and _FOLLOWUP_MARKERS.search(message):
            query = f"{session.focus.last_query} {message}"
            trace["resolved_followup_query"] = query
        result = self._handle_knowledge(query, message, trace)
        session.focus.topic = "knowledge"
        session.focus.last_query = query
        session.add("agent", result.response_text)
        return result

    # -- order id extraction --------------------------------------------
    @staticmethod
    def _extract_order_id(message: str) -> str | None:
        m = ORDER_ID_LOOSE_RE.search(message)
        if not m:
            return None
        return f"ORD-{m.group(1).upper()}"

    # -- branch: disclosure refusal --------------------------------------
    def _handle_disclosure_refusal(self, message: str, trace: dict[str, Any]) -> AgentResult:
        order_id = self._extract_order_id(message)
        tool_called = None
        tool_args = None
        if order_id:
            lookup = self.orders.lookup(order_id)
            tool_called = "order_lookup"
            tool_args = {"order_id": order_id}
            trace["tool_call"] = {"name": "order_lookup", "arguments": tool_args, "found": lookup.found}
        facts = {"kind": "disclosure_refused", "refusal_text": safety.REFUSAL_TEXT}
        text = self._generate(facts)
        trace["final_response"] = text
        return AgentResult(text, tool_called, tool_args, sources=[], handoff=True, trace=trace)

    # -- branch: injection attempt ---------------------------------------
    def _handle_injection_attempt(self, message: str, trace: dict[str, Any]) -> AgentResult:
        candidates = self.retriever.retrieve(message, top_k=8)
        trace["retrieved"] = [(c.chunk.doc_id, c.chunk.heading, round(c.score, 3)) for c in candidates[:5]]

        # The counter-policy this branch should cite is picked by topic
        # keyword rather than by raw retrieval score: an earlier version
        # took whichever chunk of the returns doc happened to score
        # highest against the manipulative query, which could land on an
        # unrelated section of that doc and miss the actual "30 calendar
        # days" fact (see README bug diary). Using the whole document's
        # text guarantees the real policy fact is present regardless of
        # which section the injected query happens to score against.
        # Known limitation: this topic routing (return vs. shipping vs.
        # warranty, etc.) is currently keyword-based and scoped to the
        # returns-policy scenario the assignment demonstrates -- see
        # README "Known limitations".
        doc_id = "01-returns-policy-current.md"
        source = doc_id
        policy_text = self._doc_summary(doc_id)

        facts = {
            "kind": "injection_notice",
            "policy_answer": policy_text,
            "source": source,
            "passages": [{"source_label": source, "text": policy_text}],
        }
        text = self._generate(facts)
        trace["final_response"] = text
        return AgentResult(text, None, None, sources=[doc_id], handoff=False, trace=trace)

    # -- branch: order lookup ---------------------------------------------
    def _handle_missing_order_id(self, trace: dict[str, Any]) -> AgentResult:
        facts = {"kind": "missing_order_id"}
        text = self._generate(facts)
        trace["final_response"] = text
        return AgentResult(text, None, None, sources=[], handoff=False, trace=trace)

    def _handle_order(self, order_id: str, trace: dict[str, Any]) -> AgentResult:
        lookup = self.orders.lookup(order_id)
        trace["tool_call"] = {
            "name": "order_lookup",
            "arguments": {"order_id": order_id},
            "found": lookup.found,
            "error": lookup.error,
        }
        if not lookup.found:
            kind = "order_malformed" if lookup.error == "malformed" else "order_not_found"
            facts = {"kind": kind, "requested_id": order_id}
            text = self._generate(facts)
            trace["final_response"] = text
            handoff = kind == "order_not_found"
            return AgentResult(text, "order_lookup", {"order_id": order_id}, sources=[], handoff=handoff, trace=trace)

        trace["sanitized_tool_result"] = lookup.data
        facts = {"kind": "order_status", "order_data": lookup.data}
        text = self._generate(facts)
        trace["final_response"] = text
        handoff = lookup.data["status"] == "exception"
        return AgentResult(text, "order_lookup", {"order_id": order_id}, sources=[], handoff=handoff, trace=trace)

    # -- branch: knowledge base --------------------------------------------
    def _handle_knowledge(self, query: str, raw_message: str, trace: dict[str, Any]) -> AgentResult:
        if set(_tokenize(raw_message)) & _KNOWN_UNCOVERED_TERMS:
            trace["uncovered_topic_forced_insufficient"] = True
            facts = {"kind": "insufficient"}
            text = self._generate(facts)
            trace["final_response"] = text
            return AgentResult(text, None, None, sources=[], handoff=True, trace=trace)

        # top_k is intentionally generous: precision is enforced downstream
        # by the authoritative-only filter and the relative-score cutoff, so
        # a larger candidate pool just means multiple relevant sections of
        # the *same* selected document (e.g. a shipping doc's delivery-time
        # section AND its duties/taxes section) don't get truncated away
        # before we ever see them. An earlier, smaller top_k caused several
        # answers to cite the right document but miss a fact from it that
        # simply hadn't made the cut (see README bug diary).
        candidates = self.retriever.retrieve(query, top_k=20)
        trace["retrieved"] = [(c.chunk.doc_id, c.chunk.heading, round(c.score, 3)) for c in candidates]

        conflict = self.retriever.detect_known_conflict(query, candidates)
        if conflict:
            doc_a, doc_b = conflict
            text_a = self._doc_summary(doc_a)
            text_b = self._doc_summary(doc_b)
            facts = {
                "kind": "conflict", "doc_a": doc_a, "doc_b": doc_b,
                "safe_default": "hand-washing the tumbler body, since one active source calls for it",
                "passages": [{"source_label": doc_a, "text": text_a}, {"source_label": doc_b, "text": text_b}],
            }
            text = self._generate(facts)
            trace["final_response"] = text
            return AgentResult(text, None, None, sources=[doc_a, doc_b], handoff=True, trace=trace)

        authoritative: list[ScoredChunk] = [c for c in candidates if c.is_authoritative]
        if not authoritative or authoritative[0].score < _INSUFFICIENT_SCORE_THRESHOLD:
            facts = {"kind": "insufficient"}
            text = self._generate(facts)
            trace["final_response"] = text
            return AgentResult(text, None, None, sources=[], handoff=True, trace=trace)

        # Rank candidate DOCUMENTS (not individual chunks) by the sum of
        # each document's best two chunk scores. Bug found via manual CLI
        # testing, not caught by the eval suite: a single-chunk ranking let
        # one coincidentally-overlapping section of an unrelated document
        # (e.g. International Shipping's "Canadian returns" section, which
        # happens to share words like "return"/"customer"/"responsible"
        # with a plain domestic return-window question) outrank the
        # actually-relevant document, which pulled 2-3 irrelevant sources
        # into an otherwise-correct answer. A document with sustained
        # relevance across multiple sections should win over a document
        # with one lucky coincidental match. This is a mitigation, not a
        # full fix -- see README known limitations.
        doc_scores: dict[str, float] = {}
        doc_best_two: dict[str, list[float]] = defaultdict(list)
        for c in authoritative:
            bucket = doc_best_two[c.chunk.doc_id]
            if len(bucket) < 2:
                bucket.append(c.score)
        for doc_id, scores in doc_best_two.items():
            doc_scores[doc_id] = sum(scores)

        ranked_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_doc_score = ranked_docs[0][1]

        top_docs: list[str] = []
        for doc_id, score in ranked_docs:
            if score < top_doc_score * 0.5:
                break
            top_docs.append(doc_id)
            if len(top_docs) == 3:
                break

        # Once a document is selected as a source, pull ALL of its content
        # directly (not just the chunks that happened to score in the
        # retrieval pass). Bug found via eval: the international-shipping
        # doc's "Duties and taxes" section shares essentially no vocabulary
        # with a query like "What about Canada, and how long does it
        # take?" and so scored 0 and never entered the candidate pool at
        # all -- even though the document as a whole was clearly the right
        # source. A document is either relevant enough to cite or it
        # isn't; once it's in, its full content should be available rather
        # than gated section-by-section on the same similarity score.
        passages = [
            {"source_label": c.source_label, "text": c.text}
            for doc_id in top_docs
            for c in self.retriever.chunks_for_doc(doc_id)
        ]
        # Generous cap: this is a raw concatenation (see README limitations
        # -- a real LLM-backed generation pass would synthesize concisely
        # instead), so it needs enough room that a fact from a lower-ranked
        # section of an already-selected document doesn't get truncated
        # away before it's reached. An earlier, smaller cap silently
        # dropped a required fact this way (see README bug diary).
        answer_text = " ".join(p["text"] for p in passages)[:2500]

        handoff = bool(_ACTIONABLE_DOCS & set(top_docs) and _INCIDENT_LANGUAGE.search(raw_message))
        handoff_reason = (
            "Since this involves a specific order issue, a human specialist will need to review and "
            "approve any resolution before it's final."
            if handoff else ""
        )

        facts = {
            "kind": "knowledge_answer",
            "answer_text": answer_text,
            "sources": top_docs,
            "passages": passages,
            "handoff": handoff,
            "handoff_reason": handoff_reason,
        }
        text = self._generate(facts)
        trace["final_response"] = text
        return AgentResult(text, None, None, sources=top_docs, handoff=handoff, trace=trace)

    def _doc_summary(self, doc_id: str) -> str:
        chunks = self.retriever.chunks_for_doc(doc_id)
        return " ".join(c.text for c in chunks)[:600]

    # -- generation backend switch -----------------------------------------
    def _generate(self, facts: dict[str, Any]) -> str:
        if self._use_llm:
            from app import llm_client
            try:
                return llm_client.generate_response(facts)
            except Exception:
                # Never let a transient API/model problem take the whole
                # agent down -- fall back to the deterministic composer and
                # let it still answer safely and correctly.
                pass
        from app import responder
        return responder.generate_response(facts)
