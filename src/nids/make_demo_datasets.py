"""Create five small, real CIC-IDS2017 datasets for NetGuard demonstrations.

Each output keeps the original CIC flow columns and source ``Label`` values,
so it can be uploaded directly to the one-file scanner. The data is sampled
from Friday, which was not used to train or choose the default detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cic_detector, compare_cic

SOURCE_PATH = compare_cic.DATA_DIR / "friday.csv"
OUTPUT_DIR = compare_cic.PROJECT_ROOT / "data" / "test_datasets"
RANDOM_STATE = 20261103

DATASETS = (
    ("01_normal_traffic.csv", "Normal traffic baseline", {"BENIGN": 5_000}),
    ("02_ddos_heavy.csv", "Major DDoS incident", {"BENIGN": 500, "DDoS": 4_500}),
    ("03_low_prevalence_attack.csv", "Mostly normal traffic with rare DDoS", {"BENIGN": 4_950, "DDoS": 50}),
    ("04_portscan_reconnaissance.csv", "Port-scan reconnaissance", {"BENIGN": 2_500, "Portscan": 2_500}),
    ("05_rare_botnet_challenge.csv", "Rare Botnet activity challenge", {"BENIGN": 4_264, "Botnet": 736}),
)


def _load_label_pools(required: dict[str, int]) -> dict[str, pd.DataFrame]:
    """Uniformly sample each required label without retaining the full capture."""
    rng = np.random.default_rng(RANDOM_STATE)
    pools: dict[str, pd.DataFrame] = {
        label: pd.DataFrame() for label in required
    }
    for chunk in pd.read_csv(SOURCE_PATH, chunksize=50_000, low_memory=False):
        labels = chunk[compare_cic.LABEL_COLUMN].astype("string").str.strip()
        for label in required:
            found = chunk.loc[labels.eq(label)]
            if not found.empty:
                found = found.assign(_sample_key=rng.random(len(found)))
                candidates = pd.concat([pools[label], found], ignore_index=True)
                pools[label] = candidates.nsmallest(required[label], "_sample_key")
    unavailable = [label for label, frame in pools.items() if len(frame) < required[label]]
    if unavailable:
        raise ValueError("source data did not contain enough records for: " + ", ".join(unavailable))
    return {label: frame.drop(columns="_sample_key").reset_index(drop=True) for label, frame in pools.items()}


def create_datasets() -> list[dict[str, object]]:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"CIC source file is missing: {SOURCE_PATH}")
    required: dict[str, int] = {}
    for _, _, labels in DATASETS:
        for label, count in labels.items():
            required[label] = max(required.get(label, 0), count)
    pools = _load_label_pools(required)
    rng = np.random.default_rng(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename, scenario, labels in DATASETS:
        parts = [
            pools[label].sample(n=count, random_state=int(rng.integers(0, 2**31 - 1)))
            for label, count in labels.items()
        ]
        frame = pd.concat(parts, ignore_index=True).sample(
            frac=1, random_state=int(rng.integers(0, 2**31 - 1))
        ).reset_index(drop=True)
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        scan_output = OUTPUT_DIR / f".{filename}.scan.csv"
        scan = cic_detector.scan_csv(path, scan_output)
        scan_output.unlink(missing_ok=True)
        manifest.append({
            "file": filename,
            "scenario": scenario,
            "records": int(len(frame)),
            "source_label_counts": {label: int(count) for label, count in labels.items()},
            "default_detector_metrics": scan.get("labeled_evaluation"),
        })
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    for dataset in create_datasets():
        print(f"{dataset['file']}: {dataset['records']:,} rows")


if __name__ == "__main__":
    main()
