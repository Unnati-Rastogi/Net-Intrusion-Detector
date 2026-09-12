"""Train the NIDS model on the full labelled training dataset.

Usage (from the project root):
    python -m src.nids.train
"""

import joblib

from . import config
from .data import load_training_data
from .pipeline import build_pipeline


def main() -> None:
    X, y = load_training_data()
    print(f"loaded training data: {X.shape[0]} rows x {X.shape[1]} feature columns")
    print(
        f"class balance: normal={int((y == 0).sum())}, "
        f"anomaly={int((y == 1).sum())}"
    )

    model = build_pipeline()
    model.fit(X, y)
    print("training complete on the full training dataset")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"saved fitted pipeline to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
