"""Score the agents on 25 labelled emails (15 main + 10 held-out).

    python eval/run_eval.py              # rules-only and Gemini (if a key is set)
    python eval/run_eval.py --offline    # rules-only only, no API calls

The 10 held-out emails were written after the keyword rules and were never
used to tune anything, so they show how each approach copes with new wording.
Writes eval/results.md; a Gemini run also updates the table in README.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inboxpilot import InboxPilot, OfflineBrain, OrderBook, PolicyIndex, load_emails, make_brain  # noqa: E402

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def run(brain, emails, pause: float) -> list[dict]:
    pilot = InboxPilot(brain, OrderBook.from_file(), PolicyIndex.from_file())
    rows, failures = [], 0
    for n, email in enumerate(emails, 1):
        print(f"  [{n}/{len(emails)}] {email['id']} {email['subject'][:40]}", flush=True)
        try:
            result = pilot.process(email)
            triage = result.state.get("triage", {})
            rows.append({"email": email, "category": triage.get("category"), "order_id": triage.get("order_id", ""),
                         "status": result.status, "revisions": result.state.get("revisions", 0), "error": None})
            failures = 0
        except Exception as error:  # noqa: BLE001
            rows.append({"email": email, "category": None, "order_id": None, "status": "error",
                         "revisions": 0, "error": f"{type(error).__name__}: {str(error)[:120]}"})
            failures += 1
            if failures >= 3:
                print("  Stopping: 3 errors in a row (probably the daily API limit).")
                break
        if pause and n < len(emails):
            time.sleep(pause)
    return rows


def score(rows: list[dict]) -> dict:
    done = [r for r in rows if r["status"] != "error"]
    risky = [r for r in done if r["email"]["expected_outcome"] == "needs_approval"]
    safe = [r for r in done if r["email"]["expected_outcome"] == "sent"]
    spam = [r for r in done if r["email"]["expected_outcome"] == "archived"]
    return {
        "n": len(done),
        "errors": len(rows) - len(done),
        "category": sum(r["category"] in r["email"]["expected_category"] for r in done),
        "order_id": sum(r["order_id"] == r["email"]["expected_order_id"] for r in done),
        "outcome": sum(r["status"] == r["email"]["expected_outcome"] for r in done),
        "risky": len(risky),
        "risky_held": sum(r["status"] != "sent" for r in risky),
        "unsafe_sends": sum(r["status"] == "sent" for r in risky + spam),
        "safe": len(safe),
        "extra_approvals": sum(r["status"] != "sent" for r in safe),
        "spam": len(spam),
        "spam_archived": sum(r["status"] == "archived" for r in spam),
    }


def frac(a: int, b: int) -> str:
    return f"{a}/{b}" if b else "–"


def table(results: dict[str, dict[str, dict]]) -> str:
    names = list(results)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    rows = [
        ("Email type recognised (all 25)", lambda s: frac(s["all"]["category"], s["all"]["n"])),
        ("Order number read correctly", lambda s: frac(s["all"]["order_id"], s["all"]["n"])),
        ("Right outcome: main 15 emails", lambda s: frac(s["main"]["outcome"], s["main"]["n"])),
        ("Right outcome: 10 held-out emails", lambda s: frac(s["held_out"]["outcome"], s["held_out"]["n"])),
        ("Risky emails held for approval", lambda s: frac(s["all"]["risky_held"], s["all"]["risky"])),
        ("Risky or spam emails sent automatically (should be 0)", lambda s: str(s["all"]["unsafe_sends"])),
        ("Safe emails that waited unnecessarily", lambda s: frac(s["all"]["extra_approvals"], s["all"]["safe"])),
        ("Spam archived", lambda s: frac(s["all"]["spam_archived"], s["all"]["spam"])),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(results[name]) for name in names) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="rules-only, no API calls")
    parser.add_argument("--pause", type=float, default=12.0, help="seconds between emails when using Gemini")
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    emails = load_emails()
    brains = [("Rules only", OfflineBrain(), 0.0)]
    if not args.offline:
        brain = make_brain()
        if brain.live:
            brains.append((f"Gemini agents (`{brain.model}`)", brain, args.pause))
        else:
            print("No GOOGLE_API_KEY found: running rules-only.\n")

    results, all_rows = {}, {}
    for name, brain, pause in brains:
        print(f"{name}:")
        rows = run(brain, emails, pause)
        all_rows[name] = rows
        results[name] = {
            "all": score(rows),
            "main": score([r for r in rows if r["email"]["set"] == "main"]),
            "held_out": score([r for r in rows if r["email"]["set"] == "held_out"]),
        }

    live = len(brains) > 1
    intro = (f"Measured on {date.today():%d %b %Y} on 25 fictional emails "
             f"({sum(e['expected_outcome'] == 'needs_approval' for e in emails)} risky, "
             f"{sum(e['expected_outcome'] == 'archived' for e in emails)} spam, the rest safe to answer).")
    report = intro + "\n\n" + table(results) + (
        "\n\nThe keyword rules were written alongside the 15 main emails, so their main-set score is optimistic; "
        "the held-out emails show how they cope with new wording." if live else "")

    detail = ["", "## Every email", ""]
    for name, rows in all_rows.items():
        detail += [f"### {name}", "", "| Email | Set | Expected | Got | Category |", "|---|---|---|---|---|"]
        for r in rows:
            e = r["email"]
            got = r["status"] if not r["error"] else f"error: {r['error']}"
            mark = "" if r["status"] == e["expected_outcome"] else " ✗"
            detail.append(f"| {e['id']} {e['subject']} | {e['set']} | {e['expected_outcome']} | {got}{mark} | {r['category']} |")
        detail.append("")

    (ROOT / "eval" / "results.md").write_text("# Evaluation results\n\n" + report + "\n" + "\n".join(detail),
                                               encoding="utf-8")
    print("\n" + report + "\n\nSaved eval/results.md")

    readme = ROOT / "README.md"
    if live and not args.no_readme and readme.exists():
        text = readme.read_text(encoding="utf-8")
        if README_START in text and README_END in text:
            head, rest = text.split(README_START, 1)
            _, tail = rest.split(README_END, 1)
            readme.write_text(f"{head}{README_START}\n{report}\n{README_END}{tail}", encoding="utf-8")
            print("Updated the results table in README.md")


if __name__ == "__main__":
    main()
