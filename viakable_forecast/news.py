"""news.py — stub (news monitor not available on this machine)."""
from __future__ import annotations

TTL_MINUTES = 30

TOPICS = [
    "Arancel / T-MEC", "Normativa México", "Precios de metales",
    "Construcción y vivienda", "Industria y nearshoring",
    "Competencia", "Otros",
]

STATUSES = ["nuevo", "relevante", "revisado", "ignorado"]

# {topic_key: base_description} — used for news cards in signal section
WHY: dict[str, str] = {}


def meta_get(_key: str):
    """Return None for any metadata key (no database on this machine)."""
    return None


def refresh(force: bool = False) -> None:
    pass


def load(limit: int = 100, days: int = 30, topics=None,
         relevance=None, statuses=None, text: str = "") -> list:
    return []


def set_status(_item_id, _status: str) -> None:
    pass


def log(_action: str, _item_id=None, _detail: str = "") -> None:
    pass


def changelog(_limit: int = 50) -> list:
    return []
