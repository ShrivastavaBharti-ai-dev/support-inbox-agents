"""Agent nodes and triage cleanup, tested without LangGraph or an API key."""
from inboxpilot import OfflineBrain, OrderBook, Outbox, PolicyIndex
from inboxpilot.agents import make_nodes, route_after_check, route_after_rules, route_after_triage
from inboxpilot.brains import normalise_triage

EMAIL = {"id": "T1", "from_name": "Priya Sharma", "from_email": "priya@example.com",
         "subject": "Where is my order?", "body": "Order ST-10482 has not arrived yet."}


def nodes(brain=None, outbox=None, decision=None):
    return make_nodes(brain or OfflineBrain(), OrderBook.from_file(), PolicyIndex.from_file(),
                      outbox or Outbox(), interrupt=lambda payload: decision or {"action": "approve"})


def run(state, *names, **kw):
    ns = nodes(**kw)
    for name in names:
        update = ns[name](state)
        trace = state.get("trace", []) + update.pop("trace", [])
        state = {**state, **update, "trace": trace}
    return state


def test_normalise_triage_rejects_bad_values():
    t = normalise_triage({"category": "refunds!!", "sentiment": "furious", "confidence": 7}, EMAIL)
    assert t["category"] == "other" and t["sentiment"] == "neutral" and t["confidence"] == 1.0


def test_order_number_comes_from_the_email_not_the_model():
    t = normalise_triage({"category": "order_status", "order_id": "ST-11111"}, EMAIL)
    assert t["order_id"] == "ST-10482"


def test_safe_flow_sends_reply_and_records_trace():
    box = Outbox()
    state = run({"email": EMAIL, "revisions": 0, "trace": []}, "triage_agent", "research_agent", "writer_agent",
                "checker_agent", "approval_rules", outbox=box)
    assert route_after_triage(state) == "research_agent"
    assert route_after_check(state) == "approval_rules" and route_after_rules(state) == "send_reply"
    state = run(state, "send_reply", outbox=box)
    assert state["status"] == "sent" and box.replies()[0]["to"] == "priya@example.com"
    assert [t["agent"] for t in state["trace"]] == ["Triage agent", "Research agent", "Writer agent",
                                                   "Checker agent", "Approval rules", "Outbox"]


class SloppyWriter(OfflineBrain):
    """First draft invents a discount; the rewrite drops it."""

    def write(self, email, triage, facts, policies, feedback):
        draft = super().write(email, triage, facts, policies, feedback)
        return draft if feedback else draft + " As an apology, here is ₹500 off your next order."


def test_checker_sends_bad_draft_back_once():
    state = run({"email": EMAIL, "revisions": 0, "trace": []}, "triage_agent", "research_agent",
                "writer_agent", "checker_agent", brain=SloppyWriter())
    assert any("₹500" in i for i in state["issues"]) and route_after_check(state) == "writer_agent"
    state = run(state, "writer_agent", "checker_agent", brain=SloppyWriter())
    assert state["revisions"] == 1 and state["issues"] == [] and route_after_check(state) == "approval_rules"


def test_human_edit_is_used():
    state = run({"email": EMAIL, "draft": "old", "approval_reasons": ["x"], "trace": []}, "human_approval",
                decision={"action": "edit", "text": "new text"})
    assert state["final_reply"] == "new text" and state["trace"][-1]["text"] == "edited and approved the reply"


def test_spam_is_archived_without_reply():
    box = Outbox()
    spam = {**EMAIL, "subject": "Boost your SEO ranking", "body": "Cheap backlinks"}
    state = run({"email": spam, "trace": []}, "triage_agent", outbox=box)
    assert route_after_triage(state) == "archive_email"
    state = run(state, "archive_email", outbox=box)
    assert state["status"] == "archived" and box.items == []
