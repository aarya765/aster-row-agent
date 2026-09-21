"""Loads knowledge-base/*.md, parses front matter, and splits each document
into heading-level chunks. No external YAML/ML dependency is required —
the front matter here is flat key: value pairs, so a small hand-rolled
parser keeps the system dependency-free and easy to audit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _parse_front_matter(raw: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


@dataclass
class Chunk:
    doc_id: str            # e.g. "01-returns-policy-current.md"
    heading: str            # e.g. "Returns Policy > Return shipping and refunds"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        """Human/citation-facing source reference: filename + heading."""
        return f"{self.doc_id} ({self.heading})" if self.heading else self.doc_id


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into (heading_path, section_text) pairs using
    the top-level title (#) as prefix and each ## as a section boundary."""
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [("", body.strip())]

    title = None
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        level, heading = len(m.group(1)), m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()

        if level == 1:
            title = heading
            if content:
                sections.append((title, content))
            continue

        heading_path = f"{title} > {heading}" if title else heading
        if content:
            sections.append((heading_path, content))
    return sections or [("", body.strip())]


def load_knowledge_base(kb_dir: str | Path) -> list[Chunk]:
    kb_dir = Path(kb_dir)
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = FRONT_MATTER_RE.match(raw)
        if match:
            meta = _parse_front_matter(match.group(1))
            body = match.group(2)
        else:
            meta = {}
            body = raw

        for heading_path, text in _split_into_sections(body):
            chunks.append(Chunk(doc_id=path.name, heading=heading_path, text=text, metadata=meta))
    return chunks
