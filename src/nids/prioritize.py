"""Deterministic alert prioritization for the NIDS (offline layer).

Consumes the risk-assessment records produced by src/nids/risk.py and
orders them for analyst review. It introduces NO new score, NO risk logic,
and NO model/SHAP calls: prioritization is a pure, deterministic ordering
of existing structured risk output.

Purpose
-------
Given many model-generated alerts, answer "what does an analyst review
first?". Detection (deciding that a record is worth attention) is already
done by the model; prioritization only ORDERS that attention.

Ordering rules (all four, in order)
-----------------------------------
1. risk_score   descending   — the primary signal
2. anomaly_score descending — same risk, higher model score first
3. risk_level by explicit priority rank (CRITICAL > HIGH > MEDIUM > LOW)
   — included for explicitness and as an integrity guard; because levels
   are a deterministic function of risk_score (risk.classify_risk), this
   never reorders records whose level/score are consistent (validated).
4. sample_id ascending      — total-order tiebreaker, fully deterministic

Priority mapping (review priority, NOT attack severity)
-------------------------------------------------------
    CRITICAL -> P1    HIGH -> P2    MEDIUM -> P3    LOW -> P4

P1 means "highest analyst review priority", never "most dangerous
attack". Impact, scope, and asset value are not modelled here.

True labels
-----------
The risk records carry true_label for offline evaluation only. This module
DROPS that field from its output: operational ordering is computed purely
from model-derived fields, so ground truth structurally cannot influence
priority.

Usage:
    from src.nids import prioritize
    alerts = prioritize.prioritize_alerts(risk_records)

    python -m src.nids.prioritize   # writes reports/alert_prioritization.json
                                    # from the existing risk-assessment report
"""

import json
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from . import config, evaluate, risk

# --- Identity ----------------------------------------------------------------
REPORT_NAME = "nids_alert_prioritization"
SCHEMA_VERSION = "1.0"
ORDERING_VERSION = "1.0.0"

# Review-priority mapping (documented in docs/alert_prioritization.md).
PRIORITY_MAP = {"CRITICAL": "P1", "HIGH": "P2", "MEDIUM": "P3", "LOW": "P4"}
# Tie-breaker 3: explicit priority rank per risk level.
LEVEL_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

TOP_N_DEFAULT = 10
RISK_REPORT_PATH = evaluate.REPORTS_DIR / "risk_assessment.json"
ALERT_REPORT_PATH = evaluate.REPORTS_DIR / "alert_prioritization.json"

REPORT_NOTE = (
    "An offline deterministic prioritization of model-generated alerts."
)
CAVEAT = (
    "P1..P4 are analyst REVIEW priorities derived from model risk scores "
    "on the labelled training distribution. They are not attack severity, "
    "not probabilities of compromise, and not impact estimates. True "
    "labels are excluded from the operational ordering."
)

REQUIRED_FIELDS = (
    "sample_id",
    "prediction",
    "anomaly_score",
    "evidence_score",
    "risk_score",
    "risk_level",
    "top_evidence",
)


# --- Input validation ----------------------------------------------------------
def validate_alert_record(record: dict[str, Any]) -> None:
    """Reject malformed, non-finite, or internally inconsistent records."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"alert record missing field(s): {missing}")

    for field in ("anomaly_score", "evidence_score", "risk_score"):
        value = float(record[field])
        if not np.isfinite(value):
            raise ValueError(
                f"alert record {record.get('sample_id')!r} has non-finite "
                f"{field}: {record[field]!r}"
            )
    s = float(record["anomaly_score"])
    if not 0.0 <= s <= 1.0:
        raise ValueError(
            f"alert record {record.get('sample_id')!r} has anomaly_score "
            f"outside [0, 1]: {s}"
        )

    level = record["risk_level"]
    if level not in LEVEL_RANK:
        raise ValueError(f"unknown risk_level: {level!r}")
    # Integrity guard: the level must be the deterministic function of the
    # score (risk.classify_risk), so tie-breaker 3 is always well-defined.
    expected_level = risk.classify_risk(float(record["risk_score"]))
    if level != expected_level:
        raise ValueError(
            f"alert record {record.get('sample_id')!r} is inconsistent: "
            f"risk_level {level!r} does not match risk_score "
            f"{record['risk_score']!r} (expected {expected_level!r})"
        )

    if not isinstance(record["top_evidence"], list):
        raise ValueError(
            f"alert record {record.get('sample_id')!r}: top_evidence must "
            "be a list"
        )


# --- Prioritization --------------------------------------------------------------
def prioritize_alerts(
    alerts: Iterable[dict[str, Any]],
    top_n: int | None = TOP_N_DEFAULT,
) -> dict[str, Any]:
    """Order risk records for analyst review; returns the structured output.

    Pure ordering over the given records: no SHAP recomputation, no model
    inference, no risk recomputation, no randomness, no true labels. Input
    records are validated (finite scores, consistent level/score, unique
    sample_ids). Output alert records drop true_label entirely.
    """
    records = list(alerts)
    seen_ids: set[int] = set()
    for record in records:
        validate_alert_record(record)
        sample_id = int(record["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id in alert records: {sample_id}")
        seen_ids.add(sample_id)

    def sort_key(record: dict[str, Any]) -> tuple:
        return (
            -float(record["risk_score"]),        # 1. risk descending
            -float(record["anomaly_score"]),     # 2. model score descending
            -LEVEL_RANK[record["risk_level"]],   # 3. explicit level rank
            int(record["sample_id"]),            # 4. id ascending (total order)
        )

    ordered = sorted(records, key=sort_key)

    alert_records = [
        {
            "rank": rank,
            "sample_id": int(record["sample_id"]),
            "prediction": record["prediction"],
            "anomaly_score": float(record["anomaly_score"]),
            "evidence_score": float(record["evidence_score"]),
            "risk_score": float(record["risk_score"]),
            "risk_level": record["risk_level"],
            "priority": PRIORITY_MAP[record["risk_level"]],
            "top_evidence": record["top_evidence"],
        }
        for rank, record in enumerate(ordered, start=1)
    ]

    counts = {priority: 0 for priority in ("P1", "P2", "P3", "P4")}
    for alert in alert_records:
        counts[alert["priority"]] += 1

    return {
        "total_alerts": len(alert_records),
        "counts_by_priority": counts,
        "ordering": {
            "rules": [
                "risk_score descending",
                "anomaly_score descending",
                "risk_level by explicit priority rank (CRITICAL > HIGH > MEDIUM > LOW)",
                "sample_id ascending",
            ],
            "deterministic": True,
            "uses_true_labels": False,
        },
        "priority_mapping": dict(PRIORITY_MAP),
        "ordered_alerts": alert_records,
        "top_n": alert_records[:top_n] if top_n else [],
    }


# --- Offline report ---------------------------------------------------------------
def build_prioritization_report(
    risk_report: dict[str, Any],
    top_n: int = TOP_N_DEFAULT,
) -> dict[str, Any]:
    """Prioritize the records of an existing risk-assessment report."""
    result = prioritize_alerts(risk_report["samples"], top_n=top_n)
    return {
        "report": REPORT_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": REPORT_NOTE,
        "caveat": CAVEAT,
        "evaluation_population": risk_report["evaluation_population"],
        "methodology": {
            "ordering_version": ORDERING_VERSION,
            "ordering_rules": result["ordering"]["rules"],
            "priority_mapping": {
                "CRITICAL": "P1 — highest analyst review priority",
                "HIGH": "P2",
                "MEDIUM": "P3",
                "LOW": "P4 — review backlog",
            },
            "mapping_rationale": (
                "The mapping exists so an analyst queue can be organized "
                "by how firmly the model's evidence supports each alert. "
                "P1 means 'highest analyst review priority', NOT 'most "
                "dangerous attack': priorities order attention, they do "
                "not grade severity or impact."
            ),
            "unflagged_records_policy": (
                "All records of the risk population are prioritized, "
                "including LOW (P4) records the model did not flag: the "
                "queue is complete and P4 is the explicit review backlog. "
                "Excluding them would hide the false-negative-prone tail "
                "of the model output; a deployment may instead drop P4 "
                "from the working queue — that choice belongs to the "
                "operator, not to this artifact."
            ),
            "true_labels": (
                "excluded from the operational ordering (fields dropped "
                "from alert records); used nowhere in this artifact"
            ),
            "no_second_score": (
                "prioritization reuses risk_score, risk_level, "
                "anomaly_score and the evidence consensus from "
                "src/nids/risk.py verbatim; no new score is computed"
            ),
            "upstream": {
                "risk_report": risk_report["report"],
                "risk_report_timestamp_utc": risk_report["timestamp_utc"],
                "risk_methodology_version":
                    risk_report["risk"]["methodology"]["version"],
                "risk_formula": risk_report["risk"]["methodology"]["formula"],
            },
        },
        "total_alerts": result["total_alerts"],
        "counts_by_priority": result["counts_by_priority"],
        "counts_by_risk_level": {
            level: result["counts_by_priority"][priority]
            for level, priority in PRIORITY_MAP.items()
        },
        "dataset": risk_report["dataset"],
        "excluded_unlabelled_dataset": risk_report["excluded_unlabelled_dataset"],
        "model": risk_report["model"],
        "environment": {**evaluate.environment_info()},
        "provenance": {
            "git_revision": evaluate.git_revision(),
            "git_dirty": evaluate.git_dirty(),
            "command": "python -m src.nids.prioritize",
        },
        "alerts": result["ordered_alerts"],
        "top_n": result["top_n"],
    }


def main(top_n: int = TOP_N_DEFAULT) -> dict[str, Any]:
    """Prioritize the existing reports/risk_assessment.json (no recompute)."""
    import time

    if not RISK_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"{RISK_REPORT_PATH} not found. Run 'python -m src.nids.risk' "
            "first; prioritization consumes its records."
        )
    risk_report = json.loads(RISK_REPORT_PATH.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    report = build_prioritization_report(risk_report, top_n=top_n)
    elapsed = time.perf_counter() - t0

    evaluate.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ALERT_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(REPORT_NOTE)
    print(f"prioritized {report['total_alerts']} alerts in {elapsed*1000:.1f} ms")
    print("counts by priority: " + ", ".join(
        f"{p}={c}" for p, c in report["counts_by_priority"].items()
    ))
    print("\ntop alerts:")
    for alert in report["top_n"]:
        print(f"  #{alert['rank']:>4} {alert['priority']} {alert['risk_level']:<8} "
              f"risk={alert['risk_score']:6.1f} s={alert['anomaly_score']:.4f} "
              f"id={alert['sample_id']}")
    print(f"\nsaved prioritization report to {ALERT_REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
