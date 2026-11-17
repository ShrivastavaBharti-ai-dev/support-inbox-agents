"""The language-model side of the agents.

* GeminiBrain: Gemini through LangChain (structured output for triage and review).
* OfflineBrain: keyword rules and reply templates, so the demo and the tests run
  without an API key. It is also the "rules only" baseline in the evaluation.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Protocol, TypeVar

from . import rules
from .knowledge import PolicySection

DEFAULT_CHAT_MODEL = "gemini-3.5-flash-lite"
SIGN_OFF = "Warm regards,\nTeam Saffron & Thread"
T = TypeVar("T")


class Brain(Protocol):
    name: str
    live: bool

    def triage(self, email: dict) -> dict: ...

    def write(self, email: dict, triage: dict, facts: dict | None, policies: list[PolicySection],
              feedback: list[str]) -> str: ...

    def review(self, email: dict, draft: str, facts: dict | None, policies: list[PolicySection]) -> dict: ...


# ------------------------------------------------------------------ helpers
def email_text(email: dict) -> str:
    return f"From: {email.get('from_name', '')} <{email.get('from_email', '')}>\nSubject: {email.get('subject', '')}\n\n{email.get('body', '')}"


def first_name(email: dict) -> str:
    name = (email.get("from_name") or "").strip()
    return name.split()[0] if name else "there"


def normalise_triage(raw: dict, email: dict) -> dict:
    """Clean model output: valid enums, an order number that really appears in
    the email, confidence between 0 and 1."""
    raw = raw or {}
    category = raw.get("category") if raw.get("category") in rules.CATEGORIES else "other"
    sentiment = raw.get("sentiment") if raw.get("sentiment") in rules.SENTIMENTS else "neutral"
    urgency = raw.get("urgency") if raw.get("urgency") in rules.URGENCIES else "normal"
    in_email = rules.find_order_id(f"{email.get('subject', '')} {email.get('body', '')}")
    try:
        confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "category": category,
        "sentiment": sentiment,
        "urgency": urgency,
        "order_id": in_email,  # taken from the text itself, never guessed
        "language": str(raw.get("language") or "English"),
        "summary": str(raw.get("summary") or email.get("subject", ""))[:300],
        "confidence": round(confidence, 2),
    }


def format_policies(policies: list[PolicySection]) -> str:
    return "\n".join(f"[{n}] {p.title}: {p.text}" for n, p in enumerate(policies, 1)) or "(none found)"


def with_backoff(call: Callable[[], T], retries: int = 4, base_delay: float = 2.0) -> T:
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as error:  # noqa: BLE001
            text = f"{type(error).__name__} {error}".lower()
            limited = any(s in text for s in ("429", "resource_exhausted", "resourceexhausted", "rate limit", "quota"))
            if attempt == retries or not limited:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError("unreachable")


def message_text(message) -> str:
    text = getattr(message, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str) and text:
        return text
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ------------------------------------------------------------------ offline
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("spam", ("seo", "backlink", "ranking", "crypto", "lottery", "verify your password", "account will be suspended",
              "claim your prize", "70% off")),
    ("complaint", ("worst", "pathetic", "third time", "fed up", "ridiculous", "cheated", "consumer court")),
    ("cancellation", ("cancel",)),
    ("refund_or_damage", ("torn", "damaged", "defect", "broken", "stain", "hole in", "wrong item")),
    ("return_or_exchange", ("return", "exchange", "replace", "too small", "too big")),
    ("refund_or_damage", ("refund", "money back")),
    ("wholesale", ("wholesale", "bulk", "corporate", "units", "hotel")),
    ("order_status", ("where is my order", "status", "track", "hasn't arrived", "not arrived", "not received",
                      "still not", "when will", "kab", "dispatched", "aayega")),
    ("shipping_question", ("ship to", "shipping", "deliver to", "international", "customs", "delivery take",
                           "cash on delivery", "cod")),
    ("product_question", ("wash", "size", "fit", "material", "fabric", "colour", "color", "care", "gift wrap",
                          "dry clean")),
    ("other", ("thank", "loved", "beautiful")),
]
_ANGRY = ("worst", "pathetic", "third time", "fed up", "ridiculous", "cheated", "!!")
_NEGATIVE = ("torn", "damaged", "disappointed", "unhappy", "doesn't match", "mistake", "too small", "not come")
_POSITIVE = ("thank", "love", "beautiful", "lovely", "great")
_HINGLISH = ("mera", "kab", "hai", "bhai", "chahiye", "aayega", "nahi", "kya")


class OfflineBrain:
    """Keyword triage and template replies. No API calls."""

    name = "rules-only"
    live = False

    def triage(self, email: dict) -> dict:
        text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
        category, confidence = "other", 0.5
        for cat, words in _KEYWORDS:
            if any(w in text for w in words):
                category, confidence = cat, 0.85
                break
        if rules.mentions_legal_action(text) or any(w in text for w in _ANGRY):
            sentiment = "angry"
        elif any(w in text for w in _NEGATIVE):
            sentiment = "negative"
        elif any(w in text for w in _POSITIVE):
            sentiment = "positive"
        else:
            sentiment = "neutral"
        words = set(re.findall(r"[a-z]+", text))
        language = "Hinglish" if len(words & set(_HINGLISH)) >= 2 else "English"
        urgency = ("high" if sentiment == "angry" or category == "cancellation"
                   else "low" if category in ("spam", "other") else "normal")
        return normalise_triage({"category": category, "sentiment": sentiment, "urgency": urgency,
                                 "language": language, "summary": email.get("subject", ""),
                                 "confidence": confidence}, email)

    def write(self, email, triage, facts, policies, feedback) -> str:
        name = first_name(email)
        cat = triage["category"]
        lines: list[str] = []
        if triage["order_id"] and not facts:
            lines.append(f"Thanks for writing. We couldn't find an order with the number {triage['order_id']}. "
                         "Could you check the order confirmation email and send us the number that starts with ST-?")
        elif cat == "order_status" and facts:
            if facts["status"] == "processing":
                lines.append(f"Your order {facts['order_id']} is being packed and will be dispatched within 2 business days.")
            elif facts["status"] == "delivered":
                lines.append(f"Our records show that order {facts['order_id']} was delivered on {facts['delivered_on']}.")
            else:
                lines.append(f"Your order {facts['order_id']} was shipped on {facts['shipped_on']} with "
                             f"{facts['courier']}, tracking number {facts['tracking']}.")
                if facts.get("is_late"):
                    lines.append("It is running later than expected, so we have asked the courier to investigate.")
                else:
                    lines.append(f"It should reach you by {facts['expected_delivery_by']}.")
        elif cat == "return_or_exchange" and facts:
            if facts.get("refund_possible"):
                lines.append(f"Of course. Items from order {facts['order_id']} can be returned until {facts['return_until']}. "
                             "Please email returns@saffronandthread.example with two photos and we will reply within "
                             "1 business day with a pickup date. A pickup fee of ₹149 is deducted from the refund, and "
                             "refunds are processed within 5–7 business days after the item reaches us.")
            elif facts.get("final_sale_items"):
                item = ", ".join(facts["final_sale_items"]).lower()
                if facts.get("final_sale_exchange", "").startswith("open"):
                    lines.append(f"The {item} in order {facts['order_id']} was a Final Sale item, so it can't be "
                                 "refunded, but you can exchange it for a different size or colour until "
                                 f"{facts['exchange_until']}. Just reply and tell us which size you'd like.")
                else:
                    lines.append(f"The {item} in order {facts['order_id']} was a Final Sale item. Final Sale items "
                                 "can't be refunded, and the 7-day exchange window has closed.")
            else:
                lines.append(f"Unfortunately the return window for order {facts['order_id']} has closed.")
        elif cat == "refund_or_damage" and facts:
            lines.append(f"We're very sorry that your order {facts['order_id']} arrived damaged. Please send us photos "
                         "of the damage. We will arrange a free pickup and offer you a replacement or a full refund.")
        elif cat == "cancellation" and facts:
            if facts.get("cancellable"):
                lines.append(f"We've received your request to cancel order {facts['order_id']}. Our team will confirm "
                             "the cancellation by email shortly.")
            else:
                lines.append(f"Order {facts['order_id']} has already been dispatched, so it can't be cancelled, but "
                             "you can return it once it arrives.")
        elif cat == "wholesale":
            lines.append("Thank you for thinking of us. Bulk orders of 25 units or more get a 12% discount, and our "
                         "founder will get in touch personally to discuss quantities and delivery dates.")
        elif cat == "complaint":
            lines.append("We're truly sorry for the trouble and for the slow reply.")
            if facts and facts.get("is_late"):
                lines.append(f"Your order {facts['order_id']} was shipped on {facts['shipped_on']} and is late, so we "
                             "have opened an investigation with the courier. If it is confirmed lost, we will send a "
                             "free replacement or a full refund.")
            lines.append("Our founder has been informed and will contact you personally.")
        elif cat == "other" and triage["sentiment"] == "positive":
            lines.append("Thank you so much for your kind words! It means a lot to our weavers and our small team.")
        elif policies:
            lines.append(f"Here is what our {policies[0].title.lower()} policy says: {policies[0].text}")
        else:
            lines.append("Thanks for your message. A member of our team will get back to you shortly.")
        return f"Hi {name},\n\n" + " ".join(lines) + f"\n\n{SIGN_OFF}"

    def review(self, email, draft, facts, policies) -> dict:
        return {"ok": True, "risk": "low", "issues": []}


# ------------------------------------------------------------------ Gemini
TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": rules.CATEGORIES},
        "sentiment": {"type": "string", "enum": rules.SENTIMENTS},
        "urgency": {"type": "string", "enum": rules.URGENCIES},
        "language": {"type": "string", "description": "Language the customer wrote in, e.g. English or Hinglish"},
        "summary": {"type": "string", "description": "One sentence: what the customer wants"},
        "confidence": {"type": "number", "description": "0 to 1: how sure you are about the category"},
    },
    "required": ["category", "sentiment", "urgency", "language", "summary", "confidence"],
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ok", "risk", "issues"],
}

TRIAGE_PROMPT = """You triage the customer-support inbox of Saffron & Thread, a small online shop selling handloom
bedsheets, curtains, cushion covers, rugs and table linen.
Categories:
- order_status: where is my order / delivery date
- return_or_exchange: wants to return or exchange an item (no damage)
- refund_or_damage: item damaged, defective, wrong, or asks for a refund
- cancellation: wants to cancel or change an order
- product_question: sizes, care, materials, gift wrap
- shipping_question: shipping costs, countries, delivery times before ordering
- wholesale: bulk, corporate or business purchase enquiries
- complaint: upset about service and wants it escalated
- spam: marketing, phishing or anything not from a real customer
- other: thanks, feedback, anything else
Sentiment "angry" means insulting, threatening or shouting; "negative" means unhappy but calm."""

WRITER_PROMPT = """You write customer-support replies for Saffron & Thread, a small handloom home-textiles shop.
Rules:
- Use ONLY the ORDER FACTS and POLICY passages provided. Never invent dates, prices, tracking numbers, discounts or timelines.
- If information you need is missing (for example no order number), ask the customer for it.
- Only promise a refund if ORDER FACTS say refund_possible is true. Final Sale items can only be exchanged.
- Never use the words "guarantee" or "100%".
- Reply in the same language and style as the customer (for example Hinglish if they wrote Hinglish).
- Warm, clear and short: 50-140 words. Start with "Hi <first name>," and end with:
Warm regards,
Team Saffron & Thread
- Plain text only, no markdown."""

CHECKER_PROMPT = """You are a strict quality checker for customer-support replies at Saffron & Thread.
Compare the DRAFT with the CUSTOMER EMAIL, ORDER FACTS and POLICY. List concrete issues:
- a fact, date, price or promise that is not supported by the ORDER FACTS or POLICY
- a refund or exchange the policy does not allow
- not answering what the customer asked
- rude, defensive or robotic tone
risk is "high" if the draft could cost the shop money or cause legal trouble, "medium" if unclear or incomplete,
"low" otherwise. ok is true only when there are no issues. Do not invent issues about style preferences."""


class GeminiBrain:
    """Gemini via LangChain. Reads GOOGLE_API_KEY from the environment."""

    live = True

    def __init__(self, model: str | None = None):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model = model or os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.name = f"gemini:{self.model}"
        self.llm = ChatGoogleGenerativeAI(model=self.model, max_retries=2)
        self._triage = self.llm.with_structured_output(TRIAGE_SCHEMA, method="json_schema")
        self._review = self.llm.with_structured_output(REVIEW_SCHEMA, method="json_schema")

    def triage(self, email: dict) -> dict:
        raw = with_backoff(lambda: self._triage.invoke([("system", TRIAGE_PROMPT), ("human", email_text(email))]))
        return normalise_triage(raw if isinstance(raw, dict) else {}, email)

    def write(self, email, triage, facts, policies, feedback) -> str:
        parts = [
            f"CUSTOMER EMAIL:\n{email_text(email)}",
            f"TRIAGE: {json.dumps(triage, ensure_ascii=False)}",
            "ORDER FACTS: " + (json.dumps(facts, ensure_ascii=False) if facts else
                               ("No order with that number was found." if triage.get("order_id") else "No order number given.")),
            f"POLICY:\n{format_policies(policies)}",
        ]
        if feedback:
            parts.append("A reviewer found these problems in your previous draft. Fix all of them:\n- " + "\n- ".join(feedback))
        message = with_backoff(lambda: self.llm.invoke([("system", WRITER_PROMPT), ("human", "\n\n".join(parts))]))
        return message_text(message).strip()

    def review(self, email, draft, facts, policies) -> dict:
        human = "\n\n".join([
            f"CUSTOMER EMAIL:\n{email_text(email)}",
            "ORDER FACTS: " + (json.dumps(facts, ensure_ascii=False) if facts else "none"),
            f"POLICY:\n{format_policies(policies)}",
            f"DRAFT:\n{draft}",
        ])
        raw = with_backoff(lambda: self._review.invoke([("system", CHECKER_PROMPT), ("human", human)]))
        raw = raw if isinstance(raw, dict) else {}
        issues = [str(i) for i in raw.get("issues") or [] if str(i).strip()][:5]
        risk = raw.get("risk") if raw.get("risk") in ("low", "medium", "high") else "medium"
        return {"ok": bool(raw.get("ok")) and not issues, "risk": risk, "issues": issues}


def make_brain(api_key: str | None = None) -> Brain:
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        return GeminiBrain()
    return OfflineBrain()
