"""Dataset loading helpers for the NIDS ML core.

The provided train/test split is respected as-is: Train_data.csv is the only
labelled dataset and is used for fitting and validation; Test_data.csv is
features-only (verified: no 'class' column) and is never merged into training.
"""

import pandas as pd

from . import config


def load_training_data(path=config.TRAIN_CSV) -> tuple[pd.DataFrame, pd.Series]:
    """Load the labelled training dataset as (X, y).

    y is binary: 1 = anomaly, 0 = normal. Raises on missing target or
    unexpected label values so schema changes fail loudly.
    """
    df = pd.read_csv(path)

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"training data must contain a '{config.TARGET_COLUMN}' column: {path}"
        )

    labels = set(df[config.TARGET_COLUMN].unique())
    unexpected = labels - {config.POSITIVE_CLASS, config.NEGATIVE_CLASS}
    if unexpected:
        raise ValueError(f"unexpected labels in training data: {sorted(unexpected)}")

    y = (df[config.TARGET_COLUMN] == config.POSITIVE_CLASS).astype(int)
    X = df.drop(columns=[config.TARGET_COLUMN])
    return X, y


def load_inference_data(path=config.TEST_CSV) -> pd.DataFrame:
    """Load the unlabelled test/inference dataset (features only)."""
    df = pd.read_csv(path)

    if config.TARGET_COLUMN in df.columns:
        raise ValueError(
            f"unexpected '{config.TARGET_COLUMN}' column in inference data: {path}"
        )
    return df


def load_category_values() -> dict[str, list[str]]:
    """Sorted unique values of each categorical column from the training data.

    Used by the dashboard to render categorical fields as dropdowns. Returns
    {} if the training CSV cannot be read (the dashboard then falls back to
    free-text inputs, which the model's handle_unknown='ignore' tolerates).
    """
    try:
        df = pd.read_csv(config.TRAIN_CSV, usecols=config.CATEGORICAL_COLUMNS)
    except Exception:
        return {}
    return {
        column: sorted(df[column].dropna().astype(str).unique().tolist())
        for column in config.CATEGORICAL_COLUMNS
    }
