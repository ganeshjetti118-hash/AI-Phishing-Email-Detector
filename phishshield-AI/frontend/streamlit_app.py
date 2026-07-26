from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from feature_extractor import extract_urls  # noqa: E402
from model import service  # noqa: E402
from sender_checker import analyze_sender  # noqa: E402
from url_checker import analyze_urls  # noqa: E402


API_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="PhishShield AI", page_icon="PS", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1180px;}
    [data-testid="stMetricValue"] {font-size: 1.85rem;}
    .risk-low {color: #177245; font-weight: 700;}
    .risk-medium {color: #9a6700; font-weight: 700;}
    .risk-high {color: #b42318; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def call_api(subject: str, body: str, sender: str, reply_to: str) -> dict | None:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"subject": subject, "body": body, "sender": sender, "reply_to": reply_to or None},
            timeout=4,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def local_prediction(subject: str, body: str, sender: str, reply_to: str) -> dict:
    prediction = service.predict(subject=subject, body=body, sender=sender)
    urls = extract_urls(" ".join([subject, body]))
    url_results = analyze_urls(urls)
    sender_result = analyze_sender(sender, reply_to=reply_to or None)
    highest_url_score = max((result["risk_score"] for result in url_results), default=0)
    combined_score = min(
        100,
        round(prediction.phishing_probability * 70 + sender_result["risk_score"] * 0.15 + highest_url_score * 0.15),
    )
    return {
        **prediction.__dict__,
        "combined_risk_score": combined_score,
        "sender_analysis": sender_result,
        "url_analysis": url_results,
    }


def risk_gauge(score: int) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#344054"},
                "steps": [
                    {"range": [0, 35], "color": "#d1fadf"},
                    {"range": [35, 70], "color": "#fef0c7"},
                    {"range": [70, 100], "color": "#fee4e2"},
                ],
            },
        )
    )
    figure.update_layout(height=230, margin={"l": 20, "r": 20, "t": 20, "b": 10})
    return figure


st.title("PhishShield AI")

left, right = st.columns([0.58, 0.42], gap="large")

with left:
    st.subheader("Email analysis")
    sender = st.text_input("Sender", placeholder="Security Team <alerts@example.com>")
    reply_to = st.text_input("Reply-To", placeholder="Optional")
    subject = st.text_input("Subject", placeholder="Your account needs verification")
    body = st.text_area("Email body", height=280, placeholder="Paste the email body here...")
    use_api = st.toggle("Use FastAPI backend", value=True)
    analyze = st.button("Analyze email", type="primary", use_container_width=True)

with right:
    st.subheader("Quick URL check")
    url = st.text_input("URL", placeholder="https://example.com/login")
    if st.button("Analyze URL", use_container_width=True):
        if url.strip():
            result = analyze_urls([url.strip()])[0]
            st.metric("URL risk", f"{result['risk_score']}/100", result["risk_level"].title())
            for flag in result["flags"]:
                st.write(f"- {flag}")
        else:
            st.warning("Enter a URL first.")

if analyze:
    if not body.strip():
        st.warning("Paste an email body before analyzing.")
        st.stop()

    with st.spinner("Analyzing email..."):
        result = call_api(subject, body, sender, reply_to) if use_api else None
        if result is None:
            result = local_prediction(subject, body, sender, reply_to)
            if use_api:
                pass

    score = int(result["combined_risk_score"])
    label = result["label"].title()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Verdict", label)
    metric_cols[1].metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    metric_cols[2].metric("Phishing probability", f"{result['phishing_probability'] * 100:.1f}%")
    metric_cols[3].metric("Combined risk", f"{score}/100")

    gauge_col, detail_col = st.columns([0.38, 0.62], gap="large")
    with gauge_col:
        st.plotly_chart(risk_gauge(score), use_container_width=True)
    with detail_col:
        st.subheader("Signals")
        signals = result.get("signals") or ["No strong text-based warning signs found."]
        for signal in signals:
            st.write(f"- {signal}")

        st.subheader("Sender")
        sender_analysis = result["sender_analysis"]
        st.write(f"Domain: `{sender_analysis.get('domain') or 'unknown'}`")
        st.write(f"Risk: {sender_analysis['risk_score']}/100 ({sender_analysis['risk_level']})")
        for flag in sender_analysis["flags"]:
            st.write(f"- {flag}")

    st.subheader("Detected URLs")
    url_analysis = result.get("url_analysis", [])
    if url_analysis:
        st.dataframe(
            pd.DataFrame(url_analysis)[["url", "domain", "risk_score", "risk_level", "flags"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No URLs found in the email.")

    st.subheader("Extracted features")
    features = pd.DataFrame([result["features"]]).T.rename(columns={0: "value"})
    st.dataframe(features, use_container_width=True)
