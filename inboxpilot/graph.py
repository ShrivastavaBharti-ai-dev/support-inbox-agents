"""Wire the agents into a LangGraph state machine with a human approval step.

    triage ──spam──────────────────────────────────────────────► archive
       │
       ▼
    research ─► writer ─► checker ──issues (1 retry)──► writer
                             │
                             ▼
                       approval rules ──risky──► human approval ──reject──► archive
                             │                         │
                             └──────safe──────► send ◄─┘ approve / edit
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .agents import (
    TicketState,
    make_nodes,
    route_after_check,
    route_after_human,
    route_after_rules,
    route_after_triage,
)
from .brains import Brain
from .knowledge import OrderBook, PolicyIndex
from .outbox import Outbox


def build_graph(brain: Brain, orders: OrderBook, policies: PolicyIndex, outbox: Outbox, checkpointer=None):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    nodes = make_nodes(brain, orders, policies, outbox)
    graph = StateGraph(TicketState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "triage_agent")
    graph.add_conditional_edges("triage_agent", route_after_triage,
                                {"archive_email": "archive_email", "research_agent": "research_agent"})
    graph.add_edge("research_agent", "writer_agent")
    graph.add_edge("writer_agent", "checker_agent")
    graph.add_conditional_edges("checker_agent", route_after_check,
                                {"writer_agent": "writer_agent", "approval_rules": "approval_rules"})
    graph.add_conditional_edges("approval_rules", route_after_rules,
                                {"human_approval": "human_approval", "send_reply": "send_reply"})
    graph.add_conditional_edges("human_approval", route_after_human,
                                {"send_reply": "send_reply", "archive_email": "archive_email"})
    graph.add_edge("send_reply", END)
    graph.add_edge("archive_email", END)
    # The checkpointer saves each ticket's state, so a run can pause for approval and resume later.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


@dataclass
class TicketResult:
    thread_id: str
    state: dict
    pending: dict | None  # what the approver needs to see, if waiting

    @property
    def status(self) -> str:
        return "needs_approval" if self.pending else self.state.get("status", "unknown")


class InboxPilot:
    """Small facade used by the app and the evaluation."""

    def __init__(self, brain: Brain, orders: OrderBook, policies: PolicyIndex, outbox: Outbox | None = None):
        self.outbox = outbox or Outbox()
        self.graph = build_graph(brain, orders, policies, self.outbox)

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _result(self, thread_id: str, output: dict) -> TicketResult:
        interrupts = output.get("__interrupt__") or []
        state = {k: v for k, v in output.items() if k != "__interrupt__"}
        return TicketResult(thread_id, state, interrupts[0].value if interrupts else None)

    def process(self, email: dict, thread_id: str | None = None) -> TicketResult:
        thread_id = thread_id or f"{email.get('id', 'email')}-{uuid.uuid4().hex[:8]}"
        output = self.graph.invoke({"email": email, "revisions": 0, "trace": []}, self._config(thread_id))
        return self._result(thread_id, output)

    def resolve(self, thread_id: str, action: str, text: str | None = None) -> TicketResult:
        """action: "approve", "edit" (with text) or "reject"."""
        from langgraph.types import Command

        decision = {"action": action}
        if text is not None:
            decision["text"] = text
        output = self.graph.invoke(Command(resume=decision), self._config(thread_id))
        return self._result(thread_id, output)
