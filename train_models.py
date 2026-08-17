from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline_utils import (
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_LABELS,
    TARGET_COLUMN,
    ensure_data_source,
    load_spambase_frame,
    model_blueprints,
    save_model,
    score_binary_model,
)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = ensure_data_source(DATA_DIR / "spambase.data")
    full_frame = load_spambase_frame(dataset_path)
    full_frame.to_csv(DATA_DIR / "spambase_full.csv", index=False)

    train_frame, test_frame = train_test_split(
        full_frame,
        test_size=0.2,
        random_state=42,
        stratify=full_frame[TARGET_COLUMN],
    )

    test_frame.to_csv(DATA_DIR / "test_data.csv", index=False)
    train_frame.to_csv(DATA_DIR / "train_data.csv", index=False)

    train_features = train_frame[FEATURE_COLUMNS]
    train_labels = train_frame[TARGET_COLUMN]
    test_features = test_frame[FEATURE_COLUMNS]
    test_labels = test_frame[TARGET_COLUMN]

    summary_rows = []
    for model_key, blueprint in model_blueprints().items():
        fitted_model = blueprint.fit(train_features, train_labels)
        save_model(fitted_model, model_key)
        metrics = score_binary_model(fitted_model, test_features, test_labels)
        summary_rows.append(
            {
                "model_key": model_key,
                "model_name": MODEL_LABELS[model_key],
                **metrics,
            }
        )

    metrics_frame = pd.DataFrame(summary_rows).sort_values(by="auc", ascending=False)
    metrics_frame.to_csv(MODEL_DIR / "benchmark_metrics.csv", index=False)
    (MODEL_DIR / "benchmark_metrics.json").write_text(
        json.dumps(metrics_frame.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
