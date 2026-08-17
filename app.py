from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, auc, matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score

from pipeline_utils import (
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_LABELS,
    TARGET_COLUMN,
    build_confusion_and_report,
    coerce_uploaded_frame,
    load_model,
)


st.set_page_config(
    page_title="SpamGuard Lab",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #15314b 0%, #0b1020 45%, #050814 100%);
        color: #f4f7fb;
    }
    .hero-card {
        padding: 1.5rem 1.4rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.24);
    }
    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.09);
        min-height: 92px;
    }
    .small-note {
        color: rgba(244, 247, 251, 0.75);
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_benchmark_table() -> pd.DataFrame:
    metrics_path = Path(__file__).resolve().parent / "model" / "benchmark_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


@st.cache_resource
def cached_model(model_key: str):
    return load_model(model_key)


def render_score_card(title: str, value: float) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-note">{title}</div>
            <div style="font-size: 1.7rem; font-weight: 700; margin-top: 0.25rem;">{value:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confusion_matrix(matrix: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axis,
        linewidths=0.4,
        linecolor="white",
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion Matrix")
    st.pyplot(fig, use_container_width=True)


def ensure_uploaded_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = coerce_uploaded_frame(frame)
    if TARGET_COLUMN not in normalized.columns and set(FEATURE_COLUMNS).issubset(normalized.columns):
        return normalized[FEATURE_COLUMNS].copy()
    if TARGET_COLUMN in normalized.columns:
        ordered_columns = [column for column in FEATURE_COLUMNS if column in normalized.columns]
        if len(ordered_columns) == len(FEATURE_COLUMNS):
            return normalized[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    return normalized


def main() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size: 0.95rem; letter-spacing: 0.12em; text-transform: uppercase; color: #8ad3ff;">BITS WILP Assignment 2</div>
            <h1 style="margin: 0.3rem 0 0.4rem 0; font-size: 2.4rem;">SpamGuard Lab</h1>
            <p style="margin: 0; max-width: 920px; line-height: 1.6; color: rgba(244,247,251,0.82);">
                Upload the provided test CSV, compare five classification models on the same dataset, and inspect the evaluation metrics, classification report, and confusion matrix in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    benchmark_table = load_benchmark_table()

    with st.sidebar:
        st.header("Controls")
        available_models = list(MODEL_LABELS.keys())
        chosen_model_key = st.selectbox(
            "Select model",
            available_models,
            format_func=lambda key: MODEL_LABELS[key],
        )
        uploaded_csv = st.file_uploader("Upload test_data.csv", type=["csv"])
        st.caption("The app works best with the included test_data.csv that contains the label column.")

        if not benchmark_table.empty:
            st.divider()
            st.subheader("Holdout leaderboard")
            st.dataframe(
                benchmark_table[["model_name", "accuracy", "auc", "precision", "recall", "f1", "mcc"]],
                use_container_width=True,
                hide_index=True,
            )

    if uploaded_csv is None:
        default_path = DATA_DIR / "test_data.csv"
        if default_path.exists():
            st.info("No file uploaded yet. Showing the bundled test split from the project folder.")
            working_frame = pd.read_csv(default_path)
        else:
            st.warning("Upload a CSV or run train_models.py first to create the bundled test split.")
            return
    else:
        working_frame = pd.read_csv(uploaded_csv)

    cleaned_frame = ensure_uploaded_columns(working_frame)

    st.subheader("Uploaded data preview")
    st.dataframe(cleaned_frame.head(12), use_container_width=True)

    model = cached_model(chosen_model_key)

    if TARGET_COLUMN in cleaned_frame.columns:
        feature_block = cleaned_frame[FEATURE_COLUMNS]
        label_block = cleaned_frame[TARGET_COLUMN]
        confusion_matrix_values, classification_bundle, predicted_labels = build_confusion_and_report(
            model, feature_block, label_block
        )
        if hasattr(model, "predict_proba"):
            auc_value = roc_auc_score(label_block, model.predict_proba(feature_block)[:, 1])
        else:
            auc_value = roc_auc_score(label_block, model.decision_function(feature_block))

        metric_cols = st.columns(6)
        metric_values = {
            "Accuracy": accuracy_score(label_block, predicted_labels),
            "AUC": auc_value,
            "Precision": precision_score(label_block, predicted_labels, zero_division=0),
            "Recall": recall_score(label_block, predicted_labels, zero_division=0),
            "F1": f1_score(label_block, predicted_labels, zero_division=0),
            "MCC": matthews_corrcoef(label_block, predicted_labels),
        }

        for index, (metric_name, metric_value) in enumerate(metric_values.items()):
            with metric_cols[index]:
                render_score_card(metric_name, float(metric_value))

        left_panel, right_panel = st.columns([1.05, 1.0])
        with left_panel:
            render_confusion_matrix(pd.DataFrame(confusion_matrix_values))
        with right_panel:
            st.markdown("### Classification report")
            st.dataframe(pd.DataFrame(classification_bundle).T, use_container_width=True)

        scored_output = cleaned_frame.copy()
        scored_output["predicted_label"] = predicted_labels
        st.download_button(
            "Download scored predictions",
            data=scored_output.to_csv(index=False).encode("utf-8"),
            file_name=f"{chosen_model_key}_predictions.csv",
            mime="text/csv",
        )
    else:
        st.warning("The uploaded CSV does not contain the target column, so only predictions can be generated.")
        predictions = model.predict(cleaned_frame[FEATURE_COLUMNS])
        preview = cleaned_frame.copy()
        preview["predicted_label"] = predictions
        st.dataframe(preview.head(20), use_container_width=True)

    st.divider()
    st.markdown("### About the dataset")
    st.write(
        "UCI Spambase has 57 numeric features and 4,601 examples, which satisfies the assignment minimums while keeping the preprocessing pipeline straightforward for deployment."
    )


if __name__ == "__main__":
    main()
