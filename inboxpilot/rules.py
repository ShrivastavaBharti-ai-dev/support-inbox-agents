"""Deterministic business rules.

The language model reads and writes; these rules decide. Anything involving
money, legal risk or an action in the store system is checked here in plain
code, so the AI never makes those calls alone.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta

CATEGORIES = [
    "order_status", "return_or_exchange", "refund_or_damage", "cancellation",
    "product_question", "shipping_question", "wholesale", "complaint", "spam", "other",
]
SENTIMENTS = ["positive", "neutral", "negative", "angry"]
URGENCIES = ["low", "normal", "high"]

REFUND_APPROVAL_LIMIT = 2000          # rupees; bigger refunds need a human
RETURN_WINDOW_DAYS = 30
FINAL_SALE_EXCHANGE_DAYS = 7
CANCEL_WINDOW_HOURS = 12
LATE_CLAIM_AFTER_BUSINESS_DAYS = 5
LOST_CLAIM_WINDOW_DAYS = 15
METRO_CITIES = {"delhi", "mumbai", "bengaluru", "chennai", "kolkata", "hyderabad", "pune",
                "ahmedabad", "jaipur", "chandigarh", "lucknow", "kochi"}

ORDER_ID = re.compile(r"\bST-\d{4,6}\b", re.IGNORECASE)
LEGAL_TERMS = re.compile(
    r"consumer (court|forum)|legal (action|notice)|lawyer|advocate|\bsue\b|chargeback|police complaint|cyber ?cell",
    re.IGNORECASE)
_AMOUNT = re.compile(r"(?:₹|\brs\.?|\binr)\s?(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_PROMISES_REFUND = re.compile(
    r"\b(?:we(?:'ll| will| can| shall)|you(?:'ll| will) (?:get|receive)|(?:are|is) eligible for)\s+"
    r"(?:(?:process|issue|give you|offer)\s+)?(?:a\s+|your\s+)?(?:full\s+)?refund", re.IGNORECASE)
_GUARANTEE = re.compile(r"\bguarantee[sd]?\b|100\s?%", re.IGNORECASE)


# ------------------------------------------------------------------ formatting
def inr(amount: int | float) -> str:
    """₹ with Indian digit grouping: ₹1,23,450."""
    whole = int(round(amount))
    s = str(abs(whole))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return ("-" if whole < 0 else "") + "₹" + s


def nice_date(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def find_order_id(text: str) -> str:
    match = ORDER_ID.search(text or "")
    return match.group(0).upper() if match else ""


def mentions_legal_action(text: str) -> bool:
    return bool(LEGAL_TERMS.search(text or ""))


def amounts_in(text: str) -> set[int]:
    found = set()
    for raw in _AMOUNT.findall(text or ""):
        try:
            found.add(int(round(float(raw.replace(",", "")))))
        except ValueError:
            continue
    return found


# ------------------------------------------------------------------ dates
def add_business_days(start: date, days: int) -> date:
    current = start
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday-Friday
            days -= 1
    return current


def business_days_after(start: date, end: date) -> int:
    """Business days strictly after `start`, up to and including `end`."""
    count, current = 0, start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return count


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


# ------------------------------------------------------------------ order facts
def compute_facts(order: dict, category: str, now: datetime) -> dict:
    """Everything the writer may say about an order, worked out in code."""
    today = now.date()
    placed = datetime.fromisoformat(order["placed_at"])
    shipped = _parse_date(order.get("shipped_on"))
    delivered = _parse_date(order.get("delivered_on"))
    facts: dict = {
        "order_id": order["id"],
        "customer": order["customer"],
        "city": order["city"],
        "status": order["status"],
        "placed_on": nice_date(placed.date()),
        "payment": order.get("payment"),
        "items": [f"{i['qty']} x {i['name']} ({inr(i['price'])} each)" + (" [Final Sale]" if i["final_sale"] else "")
                  for i in order["items"]],
        "total": inr(order["total"]),
    }
    if shipped:
        facts.update(shipped_on=nice_date(shipped), courier=order.get("courier"), tracking=order.get("tracking"))
    if shipped and not delivered:
        metro = order["city"].lower() in METRO_CITIES
        expected_by = add_business_days(shipped, 6 if metro else 9)
        late_days = business_days_after(expected_by, today) if today > expected_by else 0
        facts.update(expected_delivery_by=nice_date(expected_by), is_late=late_days > 0,
                     business_days_late=late_days)
        if late_days >= LATE_CLAIM_AFTER_BUSINESS_DAYS:
            facts["lost_parcel_claim_deadline"] = nice_date(expected_by + timedelta(days=LOST_CLAIM_WINDOW_DAYS))
    if delivered:
        days = (today - delivered).days
        facts.update(delivered_on=nice_date(delivered), days_since_delivery=days)

    full_price = [i for i in order["items"] if not i["final_sale"]]
    final_sale = [i for i in order["items"] if i["final_sale"]]
    return_open = bool(delivered and full_price and (today - delivered).days <= RETURN_WINDOW_DAYS)
    exchange_open = bool(delivered and final_sale and (today - delivered).days <= FINAL_SALE_EXCHANGE_DAYS)
    if delivered and full_price:
        facts["return_window"] = "open" if return_open else "closed"
        facts["return_until"] = nice_date(delivered + timedelta(days=RETURN_WINDOW_DAYS))
    if final_sale:
        facts["final_sale_items"] = [i["name"] for i in final_sale]
        if delivered:
            facts["final_sale_exchange"] = "open (exchange only, no refund)" if exchange_open else "closed"
            facts["exchange_until"] = nice_date(delivered + timedelta(days=FINAL_SALE_EXCHANGE_DAYS))

    if category == "refund_or_damage":
        refund_possible = order["status"] == "delivered"
        at_stake = order["total"] if refund_possible else 0
    elif category == "return_or_exchange":
        refund_possible = return_open
        at_stake = sum(i["qty"] * i["price"] for i in full_price) if return_open else 0
    else:
        refund_possible, at_stake = False, 0
    facts["refund_possible"] = refund_possible
    facts["money_at_stake"] = at_stake

    hours_since = (now - placed).total_seconds() / 3600
    facts["cancellable"] = order["status"] == "processing" and hours_since <= CANCEL_WINDOW_HOURS
    return facts


# ------------------------------------------------------------------ checks
def check_draft(draft: str, known_text: str, order_id: str, facts: dict | None) -> list[str]:
    """Fact checks that need no AI: amounts, order numbers, refund promises."""
    issues: list[str] = []
    text = draft.strip()
    if not text:
        return ["The reply is empty"]
    if len(text.split()) > 220:
        issues.append("The reply is too long (over 220 words)")
    allowed = amounts_in(known_text)
    for amount in sorted(amounts_in(text) - allowed):
        issues.append(f"Mentions {inr(amount)}, which is not in the order or the policies")
    for mentioned in {m.upper() for m in ORDER_ID.findall(text)}:
        if mentioned != order_id:
            issues.append(f"Mentions order {mentioned}, but the email is about {order_id or 'no order'}")
    if not (facts or {}).get("refund_possible") and _PROMISES_REFUND.search(text):
        issues.append("Promises a refund that the policy doesn't allow here")
    if _GUARANTEE.search(text):
        issues.append("Uses a guarantee, which the shop doesn't offer")
    return issues


def approval_reasons(triage: dict, facts: dict | None, order_found: bool, issues: list[str],
                     email_text: str) -> list[str]:
    """Why a human must approve this reply. Empty list = safe to send."""
    reasons: list[str] = []
    category = triage.get("category", "other")
    at_stake = (facts or {}).get("money_at_stake", 0)
    if at_stake > REFUND_APPROVAL_LIMIT:
        reasons.append(f"A refund of up to {inr(at_stake)} is possible (limit for automatic replies: "
                       f"{inr(REFUND_APPROVAL_LIMIT)})")
    if category == "cancellation":
        reasons.append("Cancelling needs an action in the order system")
    if category == "wholesale":
        reasons.append("Sales lead: the owner replies to these personally")
    if category == "complaint" or triage.get("sentiment") == "angry":
        reasons.append("The customer is upset")
    if mentions_legal_action(email_text):
        reasons.append("Mentions legal action or a chargeback")
    if triage.get("order_id") and not order_found:
        reasons.append(f"Order {triage['order_id']} was not found")
    if category == "refund_or_damage" and not triage.get("order_id"):
        reasons.append("Refund request without an order number")
    if float(triage.get("confidence", 0)) < 0.6:
        reasons.append("Not sure how to classify this email")
    if issues:
        reasons.append("Quality check: " + "; ".join(issues[:3]))
    return reasons


def owner_alert(triage: dict, email: dict) -> str | None:
    """A short note for the owner about emails they should know about."""
    category = triage.get("category")
    who = f"{email.get('from_name', '')} <{email.get('from_email', '')}>"
    if category == "wholesale":
        return f"New wholesale lead from {who}: {triage.get('summary', email.get('subject', ''))}"
    if mentions_legal_action(email.get("body", "")) or triage.get("sentiment") == "angry":
        return f"Upset customer {who} ({triage.get('order_id') or 'no order'}): {triage.get('summary', '')}"
    return None


def facts_text(facts: dict | None) -> str:
    return json.dumps(facts or {}, ensure_ascii=False)
