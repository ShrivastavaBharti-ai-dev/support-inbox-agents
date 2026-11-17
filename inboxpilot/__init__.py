"""InboxPilot: a team of AI agents that works through a small shop's support inbox."""
from .brains import GeminiBrain, OfflineBrain, make_brain
from .graph import InboxPilot, TicketResult, build_graph
from .knowledge import OrderBook, PolicyIndex, load_emails
from .outbox import Outbox

__all__ = ["GeminiBrain", "InboxPilot", "OfflineBrain", "OrderBook", "Outbox", "PolicyIndex", "TicketResult",
           "build_graph", "load_emails", "make_brain"]
