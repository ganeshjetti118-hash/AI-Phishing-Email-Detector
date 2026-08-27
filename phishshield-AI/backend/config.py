from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "datasets"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "phishing_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

DATASET_FILES = (
    "phishing_email.csv",
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
)

TEXT_COLUMNS = ("text_combined", "subject", "body")
SENDER_COLUMNS = ("sender", "from")
LABEL_COLUMN = "label"

MAX_ROWS_PER_DATASET = 12000
RANDOM_STATE = 42

APP_NAME = "PhishShield AI"
API_VERSION = "1.0.0"
