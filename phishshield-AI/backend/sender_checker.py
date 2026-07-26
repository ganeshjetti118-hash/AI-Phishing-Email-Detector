from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from email.utils import parseaddr


FREE_EMAIL_DOMAINS = {
    "aol.com",
    "gmail.com",
    "hotmail.com",
    "icloud.com",
    "outlook.com",
    "proton.me",
    "protonmail.com",
    "yahoo.com",
}

BRAND_DOMAINS = {
    "amazon": "amazon.com",
    "apple": "apple.com",
    "facebook": "facebook.com",
    "google": "google.com",
    "microsoft": "microsoft.com",
    "netflix": "netflix.com",
    "paypal": "paypal.com",
}


@dataclass(frozen=True)
class SenderAnalysis:
    sender: str
    display_name: str
    email_address: str
    domain: str
    risk_score: int
    risk_level: str
    flags: list[str]


def _risk_level(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _domain_from_email(email: str) -> str:
    if "@" not in email:
        return ""
    return email.rsplit("@", 1)[1].lower().strip()


def analyze_sender(sender: str, reply_to: str | None = None) -> dict:
    display_name, email_address = parseaddr(sender or "")
    email_address = email_address.lower()
    domain = _domain_from_email(email_address)
    flags: list[str] = []
    score = 0

    if not email_address or "@" not in email_address:
        flags.append("Sender is missing a valid email address")
        score += 35
    if domain in FREE_EMAIL_DOMAINS:
        flags.append("Uses a free email provider")
        score += 8
    if domain.count(".") >= 3:
        flags.append("Sender domain has many subdomains")
        score += 8
    if re.search(r"\d{3,}", domain):
        flags.append("Sender domain contains a long number sequence")
        score += 10
    if re.search(r"[^\x00-\x7f]", sender or ""):
        flags.append("Sender contains non-ASCII characters")
        score += 10
    if display_name and email_address and "@" in display_name:
        flags.append("Display name contains an email address")
        score += 8

    lowered_sender = (sender or "").lower()
    for brand, official_domain in BRAND_DOMAINS.items():
        if brand in lowered_sender and domain and not domain.endswith(official_domain):
            flags.append(f"Mentions {brand.title()} but does not use {official_domain}")
            score += 30

    if reply_to:
        _, reply_address = parseaddr(reply_to)
        reply_domain = _domain_from_email(reply_address.lower())
        if reply_domain and domain and reply_domain != domain:
            flags.append("Reply-To domain differs from sender domain")
            score += 18

    score = min(score, 100)
    if not flags:
        flags.append("No obvious sender red flags found")

    return asdict(
        SenderAnalysis(
            sender=sender or "",
            display_name=display_name,
            email_address=email_address,
            domain=domain,
            risk_score=score,
            risk_level=_risk_level(score),
            flags=flags,
        )
    )
