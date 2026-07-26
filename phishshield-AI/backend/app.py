from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from config import API_VERSION, APP_NAME
    from feature_extractor import extract_urls
    from model import service
    from sender_checker import analyze_sender
    from url_checker import analyze_url, analyze_urls
except ImportError:
    from .config import API_VERSION, APP_NAME
    from .feature_extractor import extract_urls
    from .model import service
    from .sender_checker import analyze_sender
    from .url_checker import analyze_url, analyze_urls


app = FastAPI(title=APP_NAME, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmailRequest(BaseModel):
    subject: str = ""
    body: str = Field(default="", min_length=1)
    sender: str = ""
    reply_to: str | None = None


class UrlRequest(BaseModel):
    url: str = Field(min_length=1)


class TrainRequest(BaseModel):
    max_rows_per_dataset: int | None = Field(default=None, ge=100, le=100000)


@app.get("/")
def root() -> dict:
    return {
        "name": APP_NAME,
        "version": API_VERSION,
        "message": "Use /predict to analyze an email and /docs for API documentation.",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": service.is_ready() or service.load(),
        "metrics": service.metrics,
    }


@app.post("/predict")
def predict_email(request: EmailRequest) -> dict:
    try:
        prediction = service.predict(
            subject=request.subject,
            body=request.body,
            sender=request.sender,
        )
        urls = extract_urls(" ".join([request.subject, request.body]))
        url_results = analyze_urls(urls)
        sender_result = analyze_sender(request.sender, reply_to=request.reply_to)
        highest_url_score = max((result["risk_score"] for result in url_results), default=0)
        combined_score = min(
            100,
            round(
                prediction.phishing_probability * 70
                + sender_result["risk_score"] * 0.15
                + highest_url_score * 0.15
            ),
        )

        return {
            **prediction.__dict__,
            "combined_risk_score": combined_score,
            "sender_analysis": sender_result,
            "url_analysis": url_results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze-url")
def check_url(request: UrlRequest) -> dict:
    return analyze_url(request.url)


@app.post("/train")
def train_model(request: TrainRequest) -> dict:
    try:
        metrics = service.train(max_rows_per_dataset=request.max_rows_per_dataset)
        return {"status": "trained", "metrics": metrics}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
