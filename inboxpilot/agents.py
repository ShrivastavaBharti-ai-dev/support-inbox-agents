"""The agent team as plain functions: each node takes the ticket state and
returns the fields it changes. graph.py wires them together with LangGraph."""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Callable, TypedDict

from . import rules
from .brains import Brain
from .knowledge import OrderBook, PolicyIndex
from .outbox import Outbox

MAX_REVISIONS = 1  # the writer gets one chance to fix what the checker found

# Extra search words per category, so the right policy sections come first.
CATEGORY_HINTS = {
    "order_status": "tracking shipping delivery late lost parcel",
    "return_or_exchange": "returns full-price final sale exchange pickup fee refund timelines",
    "refund_or_damage": "damaged defective refund timelines pickup",
    "cancellation": "order changes cancellations dispatch",
    "shipping_question": "shipping india international delivery cash on delivery",
    "product_question": "care sizes warranty gift wrapping",
    "wholesale": "bulk corporate orders discount",
    "complaint": "late lost parcels damaged defective",
}


class TicketState(TypedDict, total=False):
    email: dict
    triage: dict
    order_found: bool
    facts: dict
    policies: list[dict]
    draft: str
    revisions: int
    issues: list[str]
    review: dict
    approval_reasons: list[str]
    owner_alert: str
    decision: dict
    final_reply: str
    status: str
    trace: Annotated[list[dict], operator.add]  # every agent appends what it did


def step(agent: str, text: str) -> list[dict]:
    return [{"agent": agent, "text": text}]


def make_nodes(brain: Brain, orders: OrderBook, policies: PolicyIndex, outbox: Outbox,
               interrupt: Callable[[dict], dict] | None = None,
               now: Callable[[], datetime] | None = None) -> dict[str, Callable[[TicketState], dict]]:
    """Build the node functions. `interrupt` defaults to LangGraph's."""
    clock = now or (lambda: orders.as_of)

    def triage_agent(state: TicketState) -> dict:
        t = brain.triage(state["email"])
        text = (f"{t['category'].replace('_', ' ')} · {t['sentiment']} · urgency {t['urgency']}"
                f"{' · order ' + t['order_id'] if t['order_id'] else ''} · {t['language']} "
                f"(confidence {t['confidence']:.0%})")
        return {"triage": t, "trace": step("Triage agent", text)}

    def research_agent(state: TicketState) -> dict:
        t, email = state["triage"], state["email"]
        order = orders.get(t["order_id"]) if t["order_id"] else None
        facts = rules.compute_facts(order, t["category"], clock()) if order else {}
        hints = CATEGORY_HINTS.get(t["category"], "")
        query = f"{email.get('subject', '')} {email.get('body', '')} {t.get('summary', '')} {hints} {hints}"
        found = policies.search(query, k=3)
        if order:
            order_note = f"order {order['id']}: {order['status']}, total {facts['total']}"
        elif t["order_id"]:
            order_note = f"order {t['order_id']} NOT found"
        else:
            order_note = "no order number in the email"
        return {
            "order_found": bool(order),
            "facts": facts,
            "policies": [{"title": p.title, "text": p.text} for p in found],
            "trace": step("Research agent", f"{order_note}; policies: {', '.join(p.title for p in found) or 'none'}"),
        }

    def _sections(state: TicketState):
        from .knowledge import PolicySection
        return [PolicySection(p["title"], p["text"]) for p in state.get("policies", [])]

    def writer_agent(state: TicketState) -> dict:
        is_rewrite = bool(state.get("draft"))
        feedback = state.get("issues", []) if is_rewrite else []
        draft = brain.write(state["email"], state["triage"], state.get("facts") or None, _sections(state), feedback)
        revisions = state.get("revisions", 0) + (1 if is_rewrite else 0)
        note = f"rewrote the draft to fix {len(feedback)} issue(s)" if is_rewrite else f"drafted a {len(draft.split())}-word reply"
        return {"draft": draft, "revisions": revisions, "trace": step("Writer agent", note)}

    def checker_agent(state: TicketState) -> dict:
        facts = state.get("facts") or None
        known = policies.full_text + "\n" + rules.facts_text(facts)
        issues = rules.check_draft(state["draft"], known, state["triage"]["order_id"], facts)
        review = brain.review(state["email"], state["draft"], facts, _sections(state))
        issues += [i for i in review["issues"] if i not in issues]
        text = "no problems found" if not issues else f"{len(issues)} issue(s): " + "; ".join(issues[:3])
        return {"issues": issues, "review": review,
                "trace": step("Checker agent", f"{text} (risk {review['risk']})")}

    def approval_rules(state: TicketState) -> dict:
        email = state["email"]
        reasons = rules.approval_reasons(state["triage"], state.get("facts") or None, state.get("order_found", False),
                                         state.get("issues", []), f"{email.get('subject', '')} {email.get('body', '')}")
        if state.get("review", {}).get("risk") == "high" and not any(r.startswith("Quality") for r in reasons):
            reasons.append("The checker rated this reply high-risk")
        alert = rules.owner_alert(state["triage"], email)
        text = "needs your approval: " + "; ".join(reasons) if reasons else "safe to send automatically"
        update = {"approval_reasons": reasons, "trace": step("Approval rules", text)}
        if alert:
            update["owner_alert"] = alert
        return update

    def human_approval(state: TicketState) -> dict:
        ask = interrupt
        if ask is None:
            from langgraph.types import interrupt as ask  # noqa: N813
        # Nothing before this line has side effects: LangGraph re-runs the node on resume.
        decision = ask({"email_id": state["email"].get("id"), "draft": state["draft"],
                        "reasons": state.get("approval_reasons", [])})
        decision = decision if isinstance(decision, dict) else {"action": "approve"}
        action = decision.get("action", "approve")
        if action == "reject":
            return {"decision": decision, "trace": step("You", "rejected the draft")}
        text = (decision.get("text") or state["draft"]).strip()
        verb = "edited and approved" if text != state["draft"].strip() else "approved"
        return {"decision": decision, "final_reply": text, "trace": step("You", f"{verb} the reply")}

    def send_reply(state: TicketState) -> dict:
        email = state["email"]
        reply = state.get("final_reply") or state["draft"]
        outbox.add("reply", to=email.get("from_email", ""), subject=f"Re: {email.get('subject', '')}",
                   body=reply, email_id=email.get("id"))
        notes = ["reply sent"]
        if state.get("owner_alert"):
            outbox.add("owner_alert", to="owner", subject="Inbox alert", body=state["owner_alert"], email_id=email.get("id"))
            notes.append("owner alerted")
        return {"final_reply": reply, "status": "sent", "trace": step("Outbox", ", ".join(notes))}

    def archive_email(state: TicketState) -> dict:
        rejected = state.get("decision", {}).get("action") == "reject"
        status = "rejected" if rejected else "archived"
        why = "draft rejected, left for a manual reply" if rejected else "spam: archived, no reply"
        return {"status": status, "trace": step("Outbox", why)}

    return {
        "triage_agent": triage_agent,
        "research_agent": research_agent,
        "writer_agent": writer_agent,
        "checker_agent": checker_agent,
        "approval_rules": approval_rules,
        "human_approval": human_approval,
        "send_reply": send_reply,
        "archive_email": archive_email,
    }


# ------------------------------------------------------------------ routing
def route_after_triage(state: TicketState) -> str:
    return "archive_email" if state["triage"]["category"] == "spam" else "research_agent"


def route_after_check(state: TicketState) -> str:
    if state.get("issues") and state.get("revisions", 0) < MAX_REVISIONS:
        return "writer_agent"
    return "approval_rules"


def route_after_rules(state: TicketState) -> str:
    return "human_approval" if state.get("approval_reasons") else "send_reply"


def route_after_human(state: TicketState) -> str:
    return "archive_email" if state.get("decision", {}).get("action") == "reject" else "send_reply"
