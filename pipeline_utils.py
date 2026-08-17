from __future__ import annotations

from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

SPAMBASE_SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
)

FEATURE_COLUMNS = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]
TARGET_COLUMN = "is_spam"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
}


def ensure_data_source(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(SPAMBASE_SOURCE_URL, destination)
    return destination


def load_spambase_frame(source_path: Path) -> pd.DataFrame:
    raw_frame = pd.read_csv(source_path, header=None, names=ALL_COLUMNS)
    raw_frame[TARGET_COLUMN] = raw_frame[TARGET_COLUMN].astype(int)
    return raw_frame


def build_preprocessing_block() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_blueprints() -> Dict[str, object]:
    shared_prep = build_preprocessing_block()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("prep", shared_prep),
                ("learner", LogisticRegression(max_iter=1500, random_state=42)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("prep", shared_prep),
                ("learner", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("prep", shared_prep),
                ("learner", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "naive_bayes": Pipeline(
            steps=[
                ("prep", shared_prep),
                ("learner", GaussianNB()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", shared_prep),
                (
                    "learner",
                    RandomForestClassifier(
                        n_estimators=250,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def score_binary_model(model, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
    predicted_labels = model.predict(features)
    if hasattr(model, "predict_proba"):
        positive_scores = model.predict_proba(features)[:, 1]
    else:
        positive_scores = model.decision_function(features)

    return {
        "accuracy": accuracy_score(labels, predicted_labels),
        "auc": roc_auc_score(labels, positive_scores),
        "precision": precision_score(labels, predicted_labels, zero_division=0),
        "recall": recall_score(labels, predicted_labels, zero_division=0),
        "f1": f1_score(labels, predicted_labels, zero_division=0),
        "mcc": matthews_corrcoef(labels, predicted_labels),
    }


def build_confusion_and_report(model, features: pd.DataFrame, labels: pd.Series):
    predicted_labels = model.predict(features)
    matrix = confusion_matrix(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, zero_division=0, output_dict=True)
    return matrix, report, predicted_labels


def model_file_name(model_key: str) -> str:
    return f"{model_key}.joblib"


def save_model(model, model_key: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_path = MODEL_DIR / model_file_name(model_key)
    joblib.dump(model, target_path)
    return target_path


def load_model(model_key: str):
    return joblib.load(MODEL_DIR / model_file_name(model_key))


def coerce_uploaded_frame(uploaded_frame: pd.DataFrame) -> pd.DataFrame:
    if uploaded_frame.empty:
        return uploaded_frame

    cleaned = uploaded_frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    if TARGET_COLUMN in cleaned.columns and set(FEATURE_COLUMNS).issubset(cleaned.columns):
        return cleaned[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    if set(FEATURE_COLUMNS).issubset(cleaned.columns):
        return cleaned[FEATURE_COLUMNS].copy()
    if cleaned.shape[1] == len(ALL_COLUMNS):
        cleaned.columns = ALL_COLUMNS
        return cleaned
    if cleaned.shape[1] == len(FEATURE_COLUMNS):
        cleaned.columns = FEATURE_COLUMNS
        return cleaned
    return cleaned
