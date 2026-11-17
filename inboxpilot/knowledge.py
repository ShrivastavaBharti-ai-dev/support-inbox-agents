"""The shop's data the agents can look up: orders and the policy knowledge base."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .bm25 import BM25Index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class OrderBook:
    """Read-only order lookup. In a real shop this would call Shopify,
    WooCommerce or the shop's own database."""

    def __init__(self, orders: list[dict], as_of: datetime):
        self._orders = {o["id"].upper(): o for o in orders}
        self.as_of = as_of

    @classmethod
    def from_file(cls, path: str | Path = DATA_DIR / "orders.json") -> "OrderBook":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["orders"], datetime.fromisoformat(data["as_of"]))

    def get(self, order_id: str) -> dict | None:
        return self._orders.get((order_id or "").upper())

    def ids(self) -> list[str]:
        return sorted(self._orders)

    def __len__(self) -> int:
        return len(self._orders)


@dataclass(frozen=True)
class PolicySection:
    title: str
    text: str


class PolicyIndex:
    """Keyword search over the policy sections (## headings in a Markdown file)."""

    def __init__(self, sections: list[PolicySection]):
        self.sections = sections
        self._index = BM25Index([f"{s.title}\n{s.text}" for s in sections])

    @classmethod
    def from_markdown(cls, text: str) -> "PolicyIndex":
        sections = []
        for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
            title, _, body = block.partition("\n")
            sections.append(PolicySection(title.strip(), " ".join(body.split())))
        return cls(sections)

    @classmethod
    def from_file(cls, path: str | Path = DATA_DIR / "policies.md") -> "PolicyIndex":
        return cls.from_markdown(Path(path).read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 3) -> list[PolicySection]:
        return [self.sections[i] for i, _ in self._index.top(query, k)]

    @property
    def full_text(self) -> str:
        return "\n".join(f"{s.title}: {s.text}" for s in self.sections)


def load_emails(path: str | Path = DATA_DIR / "emails.jsonl") -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
