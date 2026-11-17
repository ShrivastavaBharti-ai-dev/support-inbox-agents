"""The deterministic rules: dates, order facts, draft checks, approval reasons."""
from datetime import date

import pytest

from inboxpilot import rules
from inboxpilot.knowledge import OrderBook, PolicyIndex

BOOK = OrderBook.from_file()
NOW = BOOK.as_of


def facts(order_id, category="order_status"):
    return rules.compute_facts(BOOK.get(order_id), category, NOW)


@pytest.mark.parametrize("amount, text", [(999, "₹999"), (3450, "₹3,450"), (123456, "₹1,23,456"),
                                          (12345678, "₹1,23,45,678")])
def test_indian_rupee_format(amount, text):
    assert rules.inr(amount) == text


def test_business_days_skip_weekends():
    assert rules.add_business_days(date(2026, 11, 13), 1) == date(2026, 11, 16)  # Fri -> Mon
    assert rules.business_days_after(date(2026, 11, 13), date(2026, 11, 16)) == 1


def test_order_id_and_legal_detection():
    assert rules.find_order_id("my order st-10482 please") == "ST-10482"
    assert rules.find_order_id("no order here") == ""
    assert rules.mentions_legal_action("I will go to Consumer Court")
    assert not rules.mentions_legal_action("I love the court yard print")


def test_shipped_order_expected_date():
    f = facts("ST-10482")
    assert f["expected_delivery_by"] == "18 Nov 2026" and f["is_late"] is False


def test_late_order_gets_claim_deadline():
    f = facts("ST-10155")
    assert f["is_late"] and f["business_days_late"] >= 5 and "lost_parcel_claim_deadline" in f


def test_full_price_return_inside_window():
    f = facts("ST-10377", "return_or_exchange")
    assert f["return_window"] == "open" and f["refund_possible"] and f["money_at_stake"] == 1200


def test_final_sale_is_exchange_only():
    f = facts("ST-10511", "return_or_exchange")
    assert f["refund_possible"] is False and f["money_at_stake"] == 0
    assert f["final_sale_exchange"].startswith("open") and f["exchange_until"] == "16 Nov 2026"


def test_damage_puts_whole_order_at_stake():
    assert facts("ST-10290", "refund_or_damage")["money_at_stake"] == 4800


def test_cancellation_window():
    assert facts("ST-10560", "cancellation")["cancellable"] is True
    assert facts("ST-10482", "cancellation")["cancellable"] is False


KNOWN = PolicyIndex.from_file().full_text


def test_clean_draft_passes():
    f = facts("ST-10482")
    draft = "Hi Priya, your order ST-10482 should reach you by 18 Nov 2026. Shipping was free above ₹1,999."
    assert rules.check_draft(draft, KNOWN + rules.facts_text(f), "ST-10482", f) == []


def test_invented_amount_and_wrong_order_are_flagged():
    f = facts("ST-10482")
    issues = rules.check_draft("We'll give you ₹500 off on ST-10377.", KNOWN + rules.facts_text(f), "ST-10482", f)
    assert any("₹500" in i for i in issues) and any("ST-10377" in i for i in issues)


def test_refund_promise_only_when_allowed():
    f = facts("ST-10511", "return_or_exchange")
    known = KNOWN + rules.facts_text(f)
    assert rules.check_draft("Don't worry, we will refund you.", known, "ST-10511", f)
    assert rules.check_draft("Final Sale items are not eligible for a refund.", known, "ST-10511", f) == []
    ok = facts("ST-10377", "return_or_exchange")
    assert rules.check_draft("You will get a full refund.", KNOWN + rules.facts_text(ok), "ST-10377", ok) == []


def test_guarantee_is_flagged_and_empty_draft():
    assert rules.check_draft("We guarantee delivery tomorrow.", KNOWN, "", None)
    assert rules.check_draft("   ", KNOWN, "", None) == ["The reply is empty"]


def triage(**kw):
    base = {"category": "order_status", "sentiment": "neutral", "order_id": "", "confidence": 0.9, "summary": ""}
    return {**base, **kw}


def test_safe_email_needs_no_approval():
    assert rules.approval_reasons(triage(), facts("ST-10482"), True, [], "where is my order") == []


@pytest.mark.parametrize("kwargs, expected", [
    (dict(category="cancellation"), "order system"),
    (dict(category="wholesale"), "Sales lead"),
    (dict(sentiment="angry"), "upset"),
    (dict(confidence=0.4), "Not sure"),
    (dict(order_id="ST-1"), "was not found"),
])
def test_each_approval_rule(kwargs, expected):
    reasons = rules.approval_reasons(triage(**kwargs), None, False, [], "hello")
    assert any(expected in r for r in reasons)


def test_big_refund_and_legal_threat_need_approval():
    f = facts("ST-10290", "refund_or_damage")
    reasons = rules.approval_reasons(triage(category="refund_or_damage", order_id="ST-10290"), f, True, [],
                                     "refund or I file a chargeback")
    assert any("₹4,800" in r for r in reasons) and any("legal" in r for r in reasons)


def test_quality_issues_need_approval():
    reasons = rules.approval_reasons(triage(), None, True, ["Mentions ₹500"], "x")
    assert reasons and reasons[-1].startswith("Quality check")


def test_owner_alert_for_leads_and_angry_customers():
    email = {"from_name": "A", "from_email": "a@example.com", "body": "bulk order", "subject": "Bulk"}
    assert "wholesale lead" in rules.owner_alert(triage(category="wholesale", summary="60 sheets"), email)
    assert rules.owner_alert(triage(), email) is None
