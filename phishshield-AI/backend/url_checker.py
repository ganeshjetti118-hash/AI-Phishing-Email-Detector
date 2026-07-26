from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlparse


SHORTENER_DOMAINS = {
    "bit.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "s.id",
    "t.co",
    "tinyurl.com",
}

SUSPICIOUS_TLDS = {"click", "country", "download", "gq", "kim", "link", "ml", "party", "ru", "tk", "top", "work", "zip"}
BRAND_WORDS = {"amazon", "apple", "facebook", "google", "microsoft", "netflix", "paypal", "wellsfargo"}


@dataclass(frozen=True)
class UrlAnalysis:
    url: str
    normalized_url: str
    domain: str
    risk_score: int
    risk_level: str
    flags: list[str]


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    frequencies = {char: value.count(char) / len(value) for char in set(value)}
    return -sum(probability * math.log2(probability) for probability in frequencies.values())


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url and not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        return f"http://{url}"
    return url


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _risk_level(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def analyze_url(url: str) -> dict:
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower().strip(".")
    path_and_query = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    flags: list[str] = []
    score = 0

    if not hostname:
        flags.append("URL is missing a valid domain")
        score += 35
    if parsed.scheme and parsed.scheme != "https":
        flags.append("Does not use HTTPS")
        score += 10
    if "@" in normalized:
        flags.append("Contains @ symbol, which can hide the real destination")
        score += 20
    if hostname in SHORTENER_DOMAINS:
        flags.append("Uses a known link shortener")
        score += 20
    if hostname and _is_ip_address(hostname):
        flags.append("Uses an IP address instead of a domain")
        score += 25
    if hostname.count("-") >= 2:
        flags.append("Domain contains many hyphens")
        score += 10
    if len(hostname) > 45:
        flags.append("Domain is unusually long")
        score += 10
    if any(char.isdigit() for char in hostname) and any(char.isalpha() for char in hostname):
        flags.append("Domain mixes letters and numbers")
        score += 8
    if hostname.split(".")[-1:] and hostname.split(".")[-1] in SUSPICIOUS_TLDS:
        flags.append("Uses a high-risk top-level domain")
        score += 15
    if "xn--" in hostname:
        flags.append("Uses punycode, which can impersonate lookalike domains")
        score += 20
    if _entropy(hostname.replace(".", "")) > 3.8 and len(hostname) > 18:
        flags.append("Domain appears random or machine-generated")
        score += 12
    if len(path_and_query) > 90:
        flags.append("URL path or query is unusually long")
        score += 8
    if any(word in hostname for word in BRAND_WORDS) and not any(hostname.endswith(f"{word}.com") for word in BRAND_WORDS):
        flags.append("Domain contains a brand word but is not the official brand domain")
        score += 15

    score = min(score, 100)
    if not flags:
        flags.append("No obvious URL red flags found")

    return asdict(
        UrlAnalysis(
            url=url,
            normalized_url=normalized,
            domain=hostname,
            risk_score=score,
            risk_level=_risk_level(score),
            flags=flags,
        )
    )


def analyze_urls(urls: list[str]) -> list[dict]:
    return [analyze_url(url) for url in urls]
