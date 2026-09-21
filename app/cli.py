"""Minimal CLI for the Aster & Row support agent.

Usage:
    python -m app.cli                  interactive chat
    python -m app.cli --debug          interactive chat with full trace per turn
    python -m app.cli --message "..."  single message, non-interactive
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same reasoning as evaluation/run_eval.py: don't depend on Windows' default
# console codepage for non-ASCII characters that appear in the knowledge
# base (en dashes, curly quotes, the "–" in date ranges, etc.).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent import Agent  # noqa: E402

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"
ORDERS_PATH = Path(__file__).resolve().parent.parent / "data" / "orders.json"


def _print_result(result, debug: bool) -> None:
    print(f"\nAgent: {result.response_text}")
    if result.sources:
        print(f"  [sources: {', '.join(result.sources)}]")
    if result.tool_called:
        print(f"  [tool: {result.tool_called}({result.tool_arguments})]")
    if result.handoff:
        print("  [recommending human support]")
    if debug:
        print("  [trace]", json.dumps(result.trace, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster & Row support agent CLI")
    parser.add_argument("--debug", action="store_true", help="print full observability trace per turn")
    parser.add_argument("--message", help="single message, non-interactive mode")
    parser.add_argument("--session", default=None, help="session id (default: random per run)")
    args = parser.parse_args()

    agent = Agent(kb_dir=str(KB_DIR), orders_path=str(ORDERS_PATH))
    session_id = args.session or str(uuid.uuid4())

    if args.message:
        result = agent.handle(session_id, args.message)
        _print_result(result, args.debug)
        return

    print("Aster & Row support agent. Type 'exit' to quit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break
        result = agent.handle(session_id, message)
        _print_result(result, args.debug)


if __name__ == "__main__":
    main()
