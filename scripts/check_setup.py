"""Check that everything is installed and the Gemini key works.

    python scripts/check_setup.py

Runs one sample email through LangGraph offline, then makes one small Gemini call.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
problems = 0


def report(ok: bool, message: str, hint: str = "") -> None:
    global problems
    print(f"{'[ OK ]' if ok else '[FAIL]'} {message}")
    if not ok:
        problems += 1
        if hint:
            print(f"       -> {hint}")


report(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}", "Install Python 3.10 or newer (3.12 recommended).")
for module, package in [("langgraph", "langgraph"), ("langchain_core", "langchain-core"),
                        ("langchain_google_genai", "langchain-google-genai"), ("streamlit", "streamlit"),
                        ("dotenv", "python-dotenv")]:
    try:
        version = getattr(importlib.import_module(module), "__version__", "")
        report(True, f"{package} {version}".strip())
    except ImportError:
        report(False, f"{package} is not installed", "Run: pip install -r requirements.txt")

if problems == 0:
    from inboxpilot import InboxPilot, OfflineBrain, OrderBook, PolicyIndex, load_emails

    try:
        pilot = InboxPilot(OfflineBrain(), OrderBook.from_file(), PolicyIndex.from_file())
        risky = next(e for e in load_emails() if e["id"] == "E03")
        paused = pilot.process(risky)
        resumed = pilot.resolve(paused.thread_id, "approve")
        report(paused.status == "needs_approval" and resumed.status == "sent",
               "LangGraph workflow pauses for approval and resumes")
    except Exception as error:  # noqa: BLE001
        report(False, f"LangGraph workflow failed: {type(error).__name__}: {error}")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
report(bool(key), f"GOOGLE_API_KEY found (ends in ...{key[-4:]})" if key else "GOOGLE_API_KEY is missing",
       "Copy .env.example to .env and paste your key from https://aistudio.google.com/apikey")

if key and problems == 0:
    from inboxpilot import make_brain

    try:
        brain = make_brain(key)
        triage = brain.triage({"from_name": "Test", "subject": "Where is my order?",
                               "body": "Hi, order ST-10482 hasn't arrived yet."})
        report(triage["category"] in ("order_status", "complaint"),
               f"Gemini works ({brain.model}): triaged a test email as '{triage['category']}'")
    except Exception as error:  # noqa: BLE001
        report(False, f"Gemini call failed: {type(error).__name__}: {str(error)[:300]}",
               "Check the key. If the model is not found, set GEMINI_CHAT_MODEL in .env (e.g. gemini-3.1-flash-lite).")

print()
print("All good. Next: streamlit run app.py" if problems == 0 else f"{problems} problem(s) to fix above.")
sys.exit(1 if problems else 0)
