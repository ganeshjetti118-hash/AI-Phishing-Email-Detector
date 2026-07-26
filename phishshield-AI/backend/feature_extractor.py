from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from html import unescape
from typing import Any


URL_RE = re.compile(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)

PHISHING_KEYWORDS = {
    "account",
    "bank",
    "click",
    "confirm",
    "credentials",
    "limited",
    "login",
    "password",
    "payment",
    "refund",
    "reset",
    "security",
    "suspended",
    "update",
    "urgent",
    "verify",
    "winner",
}


@dataclass(frozen=True)
class EmailFeatures:
    character_count: int
    word_count: int
    sentence_count: int
    url_count: int
    email_count: int
    exclamation_count: int
    question_count: int
    uppercase_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    phishing_keyword_count: int
    has_html: bool
    has_money_symbol: bool


def strip_html(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"<(script|style).*?>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    clean = strip_html(text)
    clean = clean.replace("\x00", " ")
    return re.sub(r"\s+", " ", clean).strip()


def extract_urls(text: str) -> list[str]:
    return [match.rstrip(").,;!?'\"") for match in URL_RE.findall(text or "")]


def combine_email_text(subject: str = "", body: str = "", sender: str = "") -> str:
    parts = [sender or "", subject or "", body or ""]
    return normalize_text(" ".join(part for part in parts if part))


def extract_email_features(subject: str = "", body: str = "", sender: str = "") -> EmailFeatures:
    text = combine_email_text(subject=subject, body=body, sender=sender)
    lowered = text.lower()
    char_count = len(text)
    words = re.findall(r"\b[\w'-]+\b", lowered)
    sentences = [chunk for chunk in re.split(r"[.!?]+", text) if chunk.strip()]
    uppercase_chars = sum(1 for char in text if char.isupper())
    digit_chars = sum(1 for char in text if char.isdigit())
    punctuation_chars = sum(1 for char in text if char in string.punctuation)

    return EmailFeatures(
        character_count=char_count,
        word_count=len(words),
        sentence_count=len(sentences),
        url_count=len(extract_urls(text)),
        email_count=len(EMAIL_RE.findall(text)),
        exclamation_count=text.count("!"),
        question_count=text.count("?"),
        uppercase_ratio=round(uppercase_chars / char_count, 4) if char_count else 0.0,
        digit_ratio=round(digit_chars / char_count, 4) if char_count else 0.0,
        punctuation_ratio=round(punctuation_chars / char_count, 4) if char_count else 0.0,
        phishing_keyword_count=sum(1 for word in words if word in PHISHING_KEYWORDS),
        has_html=bool(re.search(r"<[^>]+>", body or "")),
        has_money_symbol=bool(re.search(r"[$£€₹]|usd|dollars?", lowered)),
    )


def extract_feature_dict(subject: str = "", body: str = "", sender: str = "") -> dict[str, Any]:
    return asdict(extract_email_features(subject=subject, body=body, sender=sender))


def summarize_risk_features(features: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    if features.get("url_count", 0) > 0:
        signals.append(f"Contains {features['url_count']} link(s)")
    if features.get("phishing_keyword_count", 0) >= 3:
        signals.append("Uses multiple phishing-related urgency or account terms")
    if features.get("uppercase_ratio", 0) > 0.2:
        signals.append("Uses unusually high uppercase text")
    if features.get("has_money_symbol"):
        signals.append("Mentions money or financial value")
    if features.get("has_html"):
        signals.append("Contains HTML content")
    return signals
