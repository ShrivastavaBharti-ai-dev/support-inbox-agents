"""Where finished replies go. In the demo nothing is emailed: replies are kept
here (and optionally in a JSON-lines file). A real setup would send them
through Gmail, Outlook or a helpdesk API."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Outbox:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.items: list[dict] = []

    def add(self, kind: str, to: str, subject: str, body: str, email_id: str | None = None) -> dict:
        item = {"type": kind, "to": to, "subject": subject, "body": body, "email_id": email_id,
                "at": datetime.now().isoformat(timespec="seconds")}
        self.items.append(item)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    def replies(self) -> list[dict]:
        return [i for i in self.items if i["type"] == "reply"]

    def alerts(self) -> list[dict]:
        return [i for i in self.items if i["type"] == "owner_alert"]
