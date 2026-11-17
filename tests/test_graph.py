"""The full LangGraph workflow, including pausing for approval and resuming.
Needs `langgraph` installed (pip install -r requirements.txt)."""
import pytest

pytest.importorskip("langgraph")

from inboxpilot import InboxPilot, OfflineBrain, OrderBook, PolicyIndex, load_emails  # noqa: E402

EMAILS = {e["id"]: e for e in load_emails()}


@pytest.fixture()
def pilot():
    return InboxPilot(OfflineBrain(), OrderBook.from_file(), PolicyIndex.from_file())


@pytest.mark.parametrize("email_id", [i for i, e in EMAILS.items() if e["set"] == "main"])
def test_each_sample_email_takes_the_expected_path(pilot, email_id):
    result = pilot.process(EMAILS[email_id])
    assert result.status == EMAILS[email_id]["expected_outcome"]


def test_risky_email_waits_then_sends_after_approval(pilot):
    result = pilot.process(EMAILS["E03"])
    assert result.status == "needs_approval" and pilot.outbox.items == []
    assert "₹4,800" in result.pending["reasons"][0]
    done = pilot.resolve(result.thread_id, "approve")
    assert done.status == "sent" and len(pilot.outbox.replies()) == 1


def test_edited_reply_is_what_gets_sent(pilot):
    result = pilot.process(EMAILS["E11"])
    done = pilot.resolve(result.thread_id, "edit", "Hi Meera, your order is cancelled.")
    assert pilot.outbox.replies()[0]["body"] == "Hi Meera, your order is cancelled."
    assert done.state["trace"][-2]["text"] == "edited and approved the reply"


def test_rejected_reply_is_not_sent(pilot):
    result = pilot.process(EMAILS["E07"])
    done = pilot.resolve(result.thread_id, "reject")
    assert done.status == "rejected" and pilot.outbox.replies() == []


def test_wholesale_lead_alerts_owner_after_approval(pilot):
    result = pilot.process(EMAILS["E07"])
    pilot.resolve(result.thread_id, "approve")
    assert len(pilot.outbox.alerts()) == 1


def test_tickets_are_independent(pilot):
    first = pilot.process(EMAILS["E03"])
    second = pilot.process(EMAILS["E11"])
    pilot.resolve(second.thread_id, "reject")
    assert pilot.resolve(first.thread_id, "approve").status == "sent"


def test_held_out_emails_run_without_errors(pilot):
    for email in EMAILS.values():
        assert pilot.process(email).status in {"sent", "needs_approval", "archived"}
