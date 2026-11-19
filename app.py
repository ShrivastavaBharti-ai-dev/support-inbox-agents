"""InboxPilot review screen.

    streamlit run app.py

Runs without an API key (rules-only mode). With GOOGLE_API_KEY set, the
agents use Gemini.
"""
from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path

import streamlit as st

from inboxpilot import InboxPilot, OrderBook, PolicyIndex, load_emails, make_brain

ROOT = Path(__file__).resolve().parent
MAX_RUNS_PER_SESSION = 8   # keeps the public demo inside Gemini's free tier
# A global safety cap shared by every visitor (on top of the per-session limit),
# so a public demo can't run the owner's API key past a daily budget. Override
# with the MAX_LLM_CALLS_PER_DAY environment variable.
MAX_LLM_CALLS_PER_DAY = int(os.getenv("MAX_LLM_CALLS_PER_DAY", "200"))
CONTRA_URL = "https://contra.com/bharti_shrivastava_zu3hypce"
STATUS_ICON = {"sent": "✅", "needs_approval": "⏳", "archived": "🗑️", "rejected": "✋", None: "✉️"}
STATUS_TEXT = {"sent": "Sent automatically", "needs_approval": "Waiting for your approval",
               "archived": "Archived as spam", "rejected": "Rejected: reply by hand"}


def setting(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    if any(p.exists() for p in [ROOT / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml"]):
        try:
            return st.secrets.get(name)
        except Exception:  # noqa: BLE001
            return None
    return None


@st.cache_resource(show_spinner=False)
def get_brain(api_key: str | None):
    return make_brain(api_key)


@st.cache_resource(show_spinner=False)
def _daily_usage() -> dict:
    """One counter shared by all sessions in this server process."""
    return {"date": datetime.date.today(), "calls": 0}


def _roll_day(usage: dict) -> None:
    today = datetime.date.today()
    if usage["date"] != today:
        usage["date"], usage["calls"] = today, 0


def day_calls_left() -> int:
    usage = _daily_usage()
    _roll_day(usage)
    return MAX_LLM_CALLS_PER_DAY - usage["calls"]


def record_call() -> None:
    usage = _daily_usage()
    _roll_day(usage)
    usage["calls"] += 1


@st.cache_resource(show_spinner=False)
def get_data():
    return OrderBook.from_file(), PolicyIndex.from_file(), [e for e in load_emails() if e.get("set") == "main"]


def is_rate_limit(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(s in text for s in ("429", "resource_exhausted", "resourceexhausted", "quota", "rate limit"))


def show_trace(state: dict) -> None:
    for item in state.get("trace", []):
        st.markdown(f"**{item['agent']}**: {item['text']}")


def main() -> None:
    st.set_page_config(page_title="InboxPilot: AI agents for a support inbox", page_icon="📬", layout="wide")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    api_key = setting("GOOGLE_API_KEY") or setting("GEMINI_API_KEY")
    model = setting("GEMINI_CHAT_MODEL")
    if model:
        os.environ["GEMINI_CHAT_MODEL"] = model

    try:
        brain = get_brain(api_key)
    except Exception as error:  # noqa: BLE001
        st.error(f"Couldn't start Gemini ({type(error).__name__}). Using rules-only mode.")
        brain = get_brain(None)
    orders, policies, inbox = get_data()

    state = st.session_state
    if "pilot" not in state or state.get("brain_name") != brain.name:
        state.pilot = InboxPilot(brain, orders, policies)
        state.brain_name = brain.name
        state.tickets = {}
        state.runs = 0
        state.custom = []

    # ---------------------------------------------------------------- header
    st.title("InboxPilot")
    st.markdown("**A team of AI agents that answers a small shop's support emails.** Safe replies go out "
                "automatically. Anything involving money, an upset customer or legal risk waits for your approval.")
    st.caption("Demo shop: *Saffron & Thread*, a fictional handloom store. All customers, orders and emails are made up. "
               "Nothing is really emailed.")
    st.caption("⏳ On the free demo server, the very first run can take up to a minute while it wakes up — "
               "after that it's quick.")

    statuses = [t["status"] for t in state.tickets.values()]
    auto_sent = sum(1 for t in state.tickets.values() if t["status"] == "sent" and not t["state"].get("decision"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed", len(statuses))
    c2.metric("Sent automatically", auto_sent)
    c3.metric("Waiting for you", statuses.count("needs_approval"))
    c4.metric("Spam archived", statuses.count("archived"))

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        if brain.live:
            st.success(f"Live: {brain.name.replace('gemini:', 'Gemini ')}")
            st.caption(f"Emails left this session: {max(0, MAX_RUNS_PER_SESSION - state.runs)}")
        else:
            st.info("Rules-only mode: no API key, so keyword rules and reply templates stand in for Gemini.")
        st.header("Inbox")
        emails = inbox + state.custom
        by_id = {e["id"]: e for e in emails}
        options = [e["id"] for e in emails] + ["__new__"]

        def label(email_id: str) -> str:
            if email_id == "__new__":
                return "✏️ Write your own email"
            email = by_id[email_id]
            icon = STATUS_ICON.get(state.tickets.get(email_id, {}).get("status"))
            return f"{icon} {email['from_name']}: {email['subject']}"

        selected = st.radio("Emails", options, format_func=label, label_visibility="collapsed")
        st.divider()
        st.caption(f"Built by Bharti Shrivastava · [Hire me on Contra]({CONTRA_URL})")

    inbox_tab, outbox_tab, how_tab = st.tabs(["Inbox", "Outbox and alerts", "How it works"])

    # ---------------------------------------------------------------- inbox
    with inbox_tab:
        if selected == "__new__":
            with st.form("new_email"):
                name = st.text_input("Customer name", "Riya Kapoor", max_chars=60)
                subject = st.text_input("Subject", "Question about my order", max_chars=120)
                body = st.text_area("Email", "Hi, my order ST-10482 hasn't arrived. When will it come?",
                                    max_chars=1500, height=150)
                if st.form_submit_button("Add to inbox"):
                    email_id = f"C{len(state.custom) + 1}"
                    state.custom.append({"id": email_id, "from_name": name.strip() or "Customer",
                                         "from_email": "customer@example.com", "subject": subject.strip(),
                                         "body": body.strip()})
                    st.success("Added. Pick it from the inbox on the left.")
            st.caption("Order numbers you can use: " + ", ".join(orders.ids()))
            return

        email = by_id[selected]
        ticket = state.tickets.get(selected)
        with st.container(border=True):
            st.markdown(f"**From:** {email['from_name']} &lt;{email['from_email']}&gt;  \n**Subject:** {email['subject']}")
            st.write(email["body"])

        if ticket is None:
            if brain.live and state.runs >= MAX_RUNS_PER_SESSION:
                st.warning(f"This demo processes {MAX_RUNS_PER_SESSION} emails per visit to stay within its free "
                           "API limit. Refresh the page later, or run it yourself with your own key.")
                return
            if brain.live and day_calls_left() <= 0:
                st.warning("This public demo has paused for today after reaching its daily limit, which keeps "
                           "it inside its free API budget. Please try again tomorrow, or run it yourself with "
                           "your own key.")
                return
            if st.button("Run the agent team", type="primary", key=f"run-{selected}"):
                try:
                    with st.spinner("The agents are working on it…"):
                        result = state.pilot.process(email)
                except Exception as error:  # noqa: BLE001
                    st.error("The demo has hit its free API limit. Please try again in a few minutes."
                             if is_rate_limit(error) else f"Something went wrong ({type(error).__name__}).")
                    return
                if brain.live:
                    record_call()
                    state.runs += 1
                state.tickets[selected] = {"thread_id": result.thread_id, "status": result.status,
                                           "state": result.state, "pending": result.pending}
                st.rerun()
            return

        left, right = st.columns([2, 3])
        with left:
            st.subheader("What the agents did")
            show_trace(ticket["state"])
        with right:
            status = ticket["status"]
            heading = STATUS_TEXT.get(status, status)
            if status == "sent" and ticket["state"].get("decision"):
                heading = "Sent after your approval"
            st.subheader(f"{STATUS_ICON.get(status)} {heading}")
            if status == "needs_approval":
                st.warning("**Why it needs you:**\n\n" + "\n".join(f"- {r}" for r in ticket["pending"]["reasons"]))
                text = st.text_area("Draft reply (you can edit it)", ticket["pending"]["draft"], height=260,
                                    key=f"draft-{selected}")
                a, b = st.columns(2)
                action = None
                if a.button("Approve and send", type="primary", key=f"approve-{selected}"):
                    action = "edit" if text.strip() != ticket["pending"]["draft"].strip() else "approve"
                if b.button("Reject", key=f"reject-{selected}"):
                    action = "reject"
                if action:
                    result = state.pilot.resolve(ticket["thread_id"], action, text if action == "edit" else None)
                    state.tickets[selected] = {"thread_id": result.thread_id, "status": result.status,
                                               "state": result.state, "pending": result.pending}
                    st.rerun()
            elif status == "sent":
                st.text_area("Reply sent", ticket["state"].get("final_reply", ""), height=260, disabled=True,
                             key=f"sent-{selected}")
            elif status == "rejected":
                st.info("The draft was rejected, so nothing was sent. The email stays with you for a manual reply.")
            else:
                st.info("Recognised as spam or phishing. No reply was written.")
            facts = ticket["state"].get("facts")
            if facts:
                with st.expander("Order facts the agents used"):
                    st.json(facts)

    # ---------------------------------------------------------------- outbox
    with outbox_tab:
        outbox = state.pilot.outbox
        if not outbox.items:
            st.caption("Nothing yet. Replies and owner alerts appear here once emails are processed.")
        for item in reversed(outbox.items):
            title = "🔔 Alert for the owner" if item["type"] == "owner_alert" else f"✉️ To {item['to']}: {item['subject']}"
            with st.expander(title):
                st.text(item["body"])

    # ---------------------------------------------------------------- how
    with how_tab:
        st.markdown(
            "1. **Triage agent** reads the email: what the customer wants, their mood, urgency and language.\n"
            "2. **Research agent** looks up the order and finds the relevant policy sections.\n"
            "3. **Writer agent** drafts a reply using only those facts.\n"
            "4. **Checker agent** checks the draft against the order and policies, and sends it back for one rewrite "
            "if something is wrong.\n"
            "5. **Approval rules** (plain code, not AI) decide whether it can go out automatically. A human approves "
            "anything with a refund over ₹2,000, a cancellation, a sales lead, an upset customer, legal threats, "
            "an unknown order number or a failed check.\n"
            "6. Spam and phishing are archived without a reply.\n\n"
            "Built with LangGraph (agent workflow with a pause for approval), LangChain and Gemini."
        )


main()
