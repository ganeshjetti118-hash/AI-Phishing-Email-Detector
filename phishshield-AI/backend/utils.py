from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

try:
    from config import DATASET_DIR, LABEL_COLUMN, MODEL_DIR, TEXT_COLUMNS
except ImportError:
    from .config import DATASET_DIR, LABEL_COLUMN, MODEL_DIR, TEXT_COLUMNS


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


logger = get_logger(__name__)


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def pickle_is_usable(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_pickle(path: Path):
    if not pickle_is_usable(path):
        raise FileNotFoundError(f"Model artifact is missing or empty: {path}")
    return joblib.load(path)


def save_pickle(obj, path: Path) -> None:
    ensure_model_dir()
    joblib.dump(obj, path)


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match
    return None


def read_dataset(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=max_rows)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=max_rows, encoding="latin-1")


def load_email_dataset(filename: str, max_rows: int | None = None) -> pd.DataFrame:
    path = DATASET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = read_dataset(path, max_rows=max_rows)
    label_column = first_existing_column(raw.columns, (LABEL_COLUMN,))
    if label_column is None:
        raise ValueError(f"{filename} does not contain a '{LABEL_COLUMN}' column")

    text_column = first_existing_column(raw.columns, TEXT_COLUMNS)
    if text_column == "text_combined":
        text = raw[text_column].fillna("").astype(str)
    else:
        subject = raw["subject"].fillna("").astype(str) if "subject" in raw else ""
        body = raw["body"].fillna("").astype(str) if "body" in raw else ""
        text = (subject + " " + body).astype(str)

    frame = pd.DataFrame(
        {
            "text": text,
            "label": pd.to_numeric(raw[label_column], errors="coerce").fillna(0).astype(int),
        }
    )
    frame = frame[frame["text"].str.strip().astype(bool)]
    frame["label"] = frame["label"].clip(lower=0, upper=1)
    return frame


def load_email_datasets(files: Iterable[str], max_rows_per_dataset: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for filename in files:
        try:
            frame = load_email_dataset(filename, max_rows=max_rows_per_dataset)
            frames.append(frame)
            logger.info("Loaded %s rows from %s", len(frame), filename)
        except Exception as exc:
            logger.warning("Skipping %s: %s", filename, exc)

    if not frames:
        raise RuntimeError("No usable training datasets were found.")

    data = pd.concat(frames, ignore_index=True).dropna()
    data = data.drop_duplicates(subset=["text"])
    return data.sample(frac=1, random_state=42).reset_index(drop=True)
