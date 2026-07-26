from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from config import DATASET_FILES, MAX_ROWS_PER_DATASET, MODEL_PATH, RANDOM_STATE, VECTORIZER_PATH
    from feature_extractor import combine_email_text, extract_feature_dict, summarize_risk_features
    from utils import load_email_datasets, load_pickle, save_pickle
except ImportError:
    from .config import DATASET_FILES, MAX_ROWS_PER_DATASET, MODEL_PATH, RANDOM_STATE, VECTORIZER_PATH
    from .feature_extractor import combine_email_text, extract_feature_dict, summarize_risk_features
    from .utils import load_email_datasets, load_pickle, save_pickle


@dataclass
class PredictionResult:
    label: str
    is_phishing: bool
    confidence: float
    phishing_probability: float
    safe_probability: float
    features: dict
    signals: list[str]


class PhishingModelService:
    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.metrics: dict[str, float | int] = {}
        self._lock = Lock()

    def load(self) -> bool:
        try:
            self.model = load_pickle(MODEL_PATH)
            self.vectorizer = load_pickle(VECTORIZER_PATH)
            return True
        except Exception:
            self.model = None
            self.vectorizer = None
            return False

    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def ensure_ready(self) -> None:
        if self.is_ready():
            return
        with self._lock:
            if self.is_ready():
                return
            if not self.load():
                self.train()

    def train(self, max_rows_per_dataset: int | None = MAX_ROWS_PER_DATASET) -> dict[str, float | int]:
        data = load_email_datasets(DATASET_FILES, max_rows_per_dataset=max_rows_per_dataset)
        if data["label"].nunique() < 2:
            raise RuntimeError("Training data must contain both phishing and safe examples.")

        x_train, x_test, y_train, y_test = train_test_split(
            data["text"],
            data["label"],
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=data["label"],
        )
        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=25000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        model = LogisticRegression(max_iter=1200, class_weight="balanced")
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)

        self.model = model
        self.vectorizer = vectorizer
        self.metrics = {
            "training_rows": int(len(data)),
            "test_rows": int(len(y_test)),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        }
        save_pickle(model, MODEL_PATH)
        save_pickle(vectorizer, VECTORIZER_PATH)
        return self.metrics

    def predict(self, subject: str = "", body: str = "", sender: str = "") -> PredictionResult:
        self.ensure_ready()
        assert self.model is not None
        assert self.vectorizer is not None

        text = combine_email_text(subject=subject, body=body, sender=sender)
        features = extract_feature_dict(subject=subject, body=body, sender=sender)
        vector = self.vectorizer.transform([text])

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            classes = list(self.model.classes_)
            phishing_index = classes.index(1) if 1 in classes else int(probabilities.argmax())
            phishing_probability = float(probabilities[phishing_index])
        else:
            phishing_probability = float(self.model.predict(vector)[0])

        is_phishing = phishing_probability >= 0.5
        confidence = phishing_probability if is_phishing else 1 - phishing_probability

        return PredictionResult(
            label="phishing" if is_phishing else "safe",
            is_phishing=is_phishing,
            confidence=round(confidence, 4),
            phishing_probability=round(phishing_probability, 4),
            safe_probability=round(1 - phishing_probability, 4),
            features=features,
            signals=summarize_risk_features(features),
        )


service = PhishingModelService()
