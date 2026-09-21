"""Multi-turn session state.

Each session_id gets its own isolated history and "focus" (the last topic
and/or order id discussed). Follow-ups are resolved by carrying the focus
forward only within the same session -- sessions never share state, so a
follow-up in session B can't accidentally pick up context from session A.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "agent"
    content: str


@dataclass
class Focus:
    """What the conversation is currently 'about', so a bare follow-up like
    'what about Canada?' or 'when will it arrive?' can be resolved."""
    topic: str | None = None       # e.g. "shipping", "returns"
    order_id: str | None = None    # last order id discussed, if any
    last_query: str | None = None  # last knowledge-base query, for follow-up resolution


@dataclass
class Session:
    session_id: str
    history: list[Turn] = field(default_factory=list)
    focus: Focus = field(default_factory=Focus)

    def add(self, role: str, content: str) -> None:
        self.history.append(Turn(role=role, content=content))

    def recent_text(self, n_turns: int = 4) -> str:
        """A short window of recent turns, used only to help resolve
        pronouns/follow-ups -- not sent to the model as free-form history
        the way a naive chatbot would."""
        return "\n".join(t.content for t in self.history[-n_turns:])


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
