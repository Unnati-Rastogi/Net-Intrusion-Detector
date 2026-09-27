"""Transparent risk assessment for the NIDS (offline layer).

Turns prediction + SHAP evidence into a MODEL-DERIVED PRIORITIZATION SCORE:

    risk = 100 * anomaly_score * (1 + consensus_ratio) / 2

where
    anomaly_score   = the production model's P(anomaly) for the record
                      (exactly the score predict.py already exposes)
    consensus_ratio = (sum of SHAP contributions) / (sum of |SHAP
                      contributions|)  in [-1, 1]
                    = how unanimously the per-feature attribution evidence
                      pushes in ONE direction. +1: every feature pushes
                      toward the net effect; 0: evidence fully contested;
                      -1: unanimous push the other way.

WHY THIS FORMULA (see docs/risk_scoring.md for the full argument):
SHAP values explain the model output, and P(anomaly) IS the model output
(base + sum(phi) = P(anomaly) exactly), so ADDING attribution magnitude to
the score would double-count the same information. Verified empirically: an
additive-mass candidate is rank-identical to the score alone (Spearman 1.000
on a 1,000-row sample) because attribution mass is coupled to the net via
the additivity identity. The information the attributions add BEYOND the
score is their directional agreement, so the evidence enters as a bounded
multiplicative consensus factor, not as an additive term:

    risk=100 requires P(anomaly)=1 AND unanimous evidence (consensus_ratio=1)
    a flagged record with divided evidence is discounted toward review
    priority, not treated as an emergency.

WHAT THE SCORE IS NOT: it is NOT a probability of an attack, of compromise,
of financial/security loss, and NOT a causal risk. It is a model-derived
prioritization score computed on the model's own output and attributions,
evaluated on the labelled training distribution.

THRESHOLDS are fixed policy values anchored to the model's operating point
(0.5, the threshold predict.py uses), NOT fitted to data: see
docs/risk_scoring.md. Quantile-based thresholds degenerate on the strongly
bimodal score distribution.

Inputs are the outputs of src/nids/explain.py::explain_samples -- SHAP is
NEVER recomputed here. This module is offline-only: nothing in it is wired
into the API, dashboard, or prediction path.

Usage:
    from src.nids import risk
    result = risk.calculate_risk_for_explanation(explanation_row)

    python -m src.nids.risk      # writes reports/risk_assessment.json
"""

import json
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from . import config, evaluate, explain

# --- Identity ----------------------------------------------------------------
METHOD_NAME = "consensus_weighted_model_risk"
METHOD_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
REPORT_NAME = "nids_risk_assessment"

RISK_FORMULA = "risk = 100 * anomaly_score * (1 + consensus_ratio) / 2"

# --- Policy thresholds (fixed, documented in docs/risk_scoring.md) ------------
# Anchors: 50 = a fully-consensual model-flagged record (s=0.5, consensus=1);
# 20 = elevated-but-sub-threshold evidence (e.g. s=0.2 with full consensus);
# 80 = flagged with strong consensus (e.g. s=1.0 with consensus>=0.6).
THRESHOLDS = {"CRITICAL": 80.0, "HIGH": 50.0, "MEDIUM": 20.0}  # else LOW
LEVEL_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

CAVEAT = (
    "Risk scores are model-derived prioritization scores evaluated on the "
    "labelled training distribution. They are not probabilities of attack "
    "or compromise, not estimates of real-world impact, and not causal "
    "risk. The risk level prioritizes model alerts; it does not measure "
    "actual attack severity."
)

# Tolerance for the optional additivity consistency check (same scale as
# explain.py's checks; one-vote threshold ties are tolerated there and may
# propagate here, so the band matches explain.ADDITIVITY_TOLERANCE plus a
# vote quantum of slack).
CONSISTENCY_TOLERANCE = 2.0 * explain.ADDITIVITY_TOLERANCE

# The note explain.py attaches to samples whose feature values sit exactly on
# a trained split threshold (their reconstruction differs by one tree vote).
TIE_NOTE = "one_vote_split_threshold_tie"


def additivity_tolerance(n_estimators: int | None, tie: bool) -> float:
    """Consistency tolerance for a record's contribution list.

    Base tolerance is CONSISTENCY_TOLERANCE; tie-flagged records (see
    explain.py) get exactly one tree-vote quantum (1/n_estimators) of extra
    slack, matching how explain.verify_additivity treats them. A tie-flagged
    record without n_estimators is rejected rather than silently relaxed.
    """
    if not tie:
        return CONSISTENCY_TOLERANCE
    if n_estimators is None or n_estimators <= 0:
        raise ValueError(
            "explanation is tie-flagged ('one_vote_split_threshold_tie'); "
            "pass n_estimators to allow exactly one tree-vote of slack"
        )
    return CONSISTENCY_TOLERANCE + 1.0 / n_estimators


# --- Core engine ---------------------------------------------------------------
def consensus_ratio(net_contribution: float, abs_mass: float) -> float:
    """Directional agreement of the attribution evidence, in [-1, 1].

    net_contribution = sum of signed SHAP contributions for the record
    abs_mass         = sum of their absolute values
    A zero mass (no attribution evidence at all) yields 0.0: no evidence,
    no consensus either way. Values are clipped into [-1, 1] to absorb
    floating-point residue only.
    """
    if abs_mass <= 0.0:
        return 0.0
    ratio = net_contribution / abs_mass
    return float(min(1.0, max(-1.0, ratio)))


def evidence_score(
    anomaly_score: float,
    shap_contributions: Iterable[float],
    base_value: float | None = None,
    tolerance: float | None = None,
) -> dict[str, float]:
    """Aggregate SHAP contributions into the evidence block.

    Returns net_contribution (signed sum), abs_mass (sum of absolute
    values), and consensus_ratio in [-1, 1]. If base_value is supplied, the
    contributions are consistency-checked against the additivity identity
    sum(phi) = anomaly_score - base_value, so partially-reported
    contribution lists cannot silently distort the consensus. `tolerance`
    overrides the default band (risk.additivity_tolerance computes the
    correct value for tie-flagged records).
    """
    s = float(anomaly_score)
    contribs = [float(c) for c in shap_contributions]
    for value in [s, *contribs]:
        if not np.isfinite(value):
            raise ValueError(f"non-finite input to risk engine: {value!r}")
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"anomaly_score must be in [0, 1], got {s}")

    net = float(sum(contribs))
    mass = float(sum(abs(c) for c in contribs))

    if base_value is not None and contribs:
        expected_net = s - float(base_value)
        band = CONSISTENCY_TOLERANCE if tolerance is None else float(tolerance)
        if abs(net - expected_net) > band:
            raise ValueError(
                "SHAP contributions are inconsistent with the additivity "
                f"identity (sum={net:.6f}, anomaly_score - base_value="
                f"{expected_net:.6f}); pass the full per-feature "
                "contribution list (top_k=None) or omit base_value"
            )
    return {
        "net_contribution": net,
        "abs_mass": mass,
        "consensus_ratio": consensus_ratio(net, mass),
    }


def calculate_risk(
    anomaly_score: float,
    shap_contributions: Iterable[float],
    base_value: float | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Risk score + level from the model score and SHAP evidence.

    Deterministic: pure floating-point arithmetic on the given inputs, no
    randomness, no model calls, no SHAP recomputation.
    """
    evidence = evidence_score(anomaly_score, shap_contributions, base_value,
                              tolerance=tolerance)
    s = float(anomaly_score)
    alpha = evidence["consensus_ratio"]
    risk_score = 100.0 * s * (1.0 + alpha) / 2.0
    return {
        "anomaly_score": s,
        "evidence": evidence,
        "evidence_score": alpha,
        "risk_score": risk_score,
        "risk_level": classify_risk(risk_score),
        "methodology": {
            "name": METHOD_NAME,
            "version": METHOD_VERSION,
            "formula": RISK_FORMULA,
            "thresholds": dict(THRESHOLDS),
            "interpretation": (
                "model-derived prioritization score in [0, 100]; NOT a "
                "probability of attack, compromise, or impact, and not "
                "causal risk"
            ),
        },
    }


def classify_risk(risk_score: float) -> str:
    """Map a risk score onto the fixed policy levels (LOW..CRITICAL)."""
    score = float(risk_score)
    if not np.isfinite(score):
        raise ValueError(f"risk_score must be finite, got {risk_score!r}")
    if score >= THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if score >= THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def calculate_risk_for_explanation(
    explanation: dict[str, Any],
    n_estimators: int | None = None,
) -> dict[str, Any]:
    """Risk for one explain_samples() row (top_k=None output).

    Uses the row's anomaly_score, base_value, and full per-feature
    contribution list; re-checks the additivity identity via
    evidence_score. Tie-flagged rows (split-threshold ties, see
    explain.py) get exactly one tree-vote of extra slack when
    n_estimators is supplied. Returns the risk block plus the
    explanation's prediction for convenience.
    """
    contribs = [c["shap_contribution"] for c in explanation["top_features"]]
    tie = explanation.get("additivity_note") == TIE_NOTE
    result = calculate_risk(
        explanation["anomaly_score"],
        contribs,
        base_value=explanation.get("base_value"),
        tolerance=additivity_tolerance(n_estimators, tie),
    )
    result["prediction"] = explanation.get("prediction")
    result["additivity_note"] = explanation.get("additivity_note")
    return result


# --- Report ---------------------------------------------------------------------
REPORT_SAMPLE_SIZE = 1000
REPORT_SEED = explain.BACKGROUND_SEED
TOP_EVIDENCE_IN_REPORT = 5


def build_risk_report(
    context: explain.ExplanationContext,
    X,
    y,
    sample_size: int = REPORT_SAMPLE_SIZE,
    sample_seed: int = REPORT_SEED,
    explanations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Explain a deterministic training sample once, then score it.

    SHAP values are computed HERE ONCE (upstream) and handed to the risk
    engine per sample; risk.py never recomputes explanations. Precomputed
    explanations may be passed in (used by main() and tests to avoid
    explaining twice).
    """
    sample_X, sample_idx = explain._report_sample_frame(X, sample_size, sample_seed)
    if explanations is None:
        explanations = explain.explain_samples(context, sample_X, top_k=None)

    per_sample: list[dict[str, Any]] = []
    level_counts = {level: 0 for level in LEVEL_ORDER}
    scores: list[float] = []
    for explanation, row_pos in zip(explanations, sample_idx):
        result = calculate_risk_for_explanation(
            explanation, n_estimators=context.classifier.n_estimators
        )
        level_counts[result["risk_level"]] += 1
        scores.append(result["risk_score"])
        true_label = config.POSITIVE_CLASS if int(y.iloc[row_pos]) == 1 \
            else config.NEGATIVE_CLASS
        per_sample.append(
            {
                "sample_id": result_index(explanation),
                "true_label": true_label,
                "prediction": result["prediction"],
                "anomaly_score": result["anomaly_score"],
                "evidence_score": result["evidence_score"],
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "top_evidence": explanation["top_features"][:TOP_EVIDENCE_IN_REPORT],
            }
        )

    score_arr = np.asarray(scores, dtype=float)
    histogram = {
        f"{lo}-{hi}": int(((score_arr >= lo) & (score_arr < hi)).sum())
        for lo, hi in zip(range(0, 100, 10), range(10, 101, 10))
    }
    histogram["100"] = int((score_arr == 100.0).sum())

    return {
        "report": REPORT_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_population": (
            f"{len(per_sample)} deterministically sampled labelled training "
            f"rows (seed {sample_seed}) from data/Train_data.csv "
            "(25,192 rows); risk scores are model-derived prioritization "
            "scores evaluated on the labelled training distribution"
        ),
        "risk": {
            "methodology": {
                "name": METHOD_NAME,
                "version": METHOD_VERSION,
                "formula": RISK_FORMULA,
                "components": {
                    "anomaly_score": (
                        "production model's P(anomaly) for the record (the "
                        "same score predict.py exposes)"
                    ),
                    "evidence_score": (
                        "consensus_ratio = sum(shap) / sum(|shap|) over the "
                        "record's full per-feature attribution list; "
                        "directional agreement of the evidence in [-1, 1]"
                    ),
                },
                "thresholds": dict(THRESHOLDS),
                "threshold_methodology": (
                    "fixed policy thresholds anchored to the model's 0.5 "
                    "operating point (predict.py's decision threshold): "
                    "HIGH begins where a fully-consensual model-flagged "
                    "record lands (s=0.5, consensus=1 -> 50); CRITICAL "
                    "requires a flagged record with strong consensus "
                    "(s=1.0, consensus>=0.6 -> 80); MEDIUM marks elevated "
                    "but sub-threshold evidence. NOT fitted to data; "
                    "quantile thresholds degenerate on the bimodal score "
                    "distribution (75th percentile of the consensus-"
                    "weighted score is ~95 on the sample below)."
                ),
                "double_counting_analysis": (
                    "SHAP explains the model output and P(anomaly) IS the "
                    "model output (base + sum(shap) = P(anomaly)), so "
                    "attribution magnitude is already inside the score: an "
                    "additive-mass candidate was verified rank-identical to "
                    "the score alone (Spearman 1.000 on a 1,000-row "
                    "sample). The evidence therefore enters only through "
                    "its directional consensus (multiplicative, bounded), "
                    "which measurably reorders flagged records by evidence "
                    "agreement (Spearman 0.879 vs the score-only order)."
                ),
                "caveat": CAVEAT,
            },
            "levels": {
                "order": list(LEVEL_ORDER),
                "counts": level_counts,
                "definition": (
                    "LOW/MEDIUM/HIGH/CRITICAL prioritize model alerts; "
                    "they are NOT claims about actual attack severity"
                ),
            },
            "score_distribution": {
                "min": float(score_arr.min()),
                "max": float(score_arr.max()),
                "mean": float(score_arr.mean()),
                "median": float(np.median(score_arr)),
                "histogram_binwidth_10": histogram,
            },
        },
        "dataset": {
            **evaluate.dataset_provenance(config.TRAIN_CSV),
            "target_column": config.TARGET_COLUMN,
            "class_counts": {
                "normal": int((y == 0).sum()),
                "anomaly": int((y == 1).sum()),
            },
        },
        "excluded_unlabelled_dataset":
            evaluate.excluded_unlabelled_dataset_provenance(),
        "model": {
            "path": "models/rf_ids_pipeline.joblib",
            "sha256": context.model_hash,
            "classifier": type(context.classifier).__name__,
            "n_estimators": context.classifier.n_estimators,
            "retrained": False,
        },
        "shap_inputs": {
            "explainer": "shap.TreeExplainer (interventional, 100-row "
                         "training background, seed "
                         f"{context.metadata['background_seed']})",
            "shap_version": context.shap_version,
            "explanation_population_matches_report_sample": True,
        },
        "environment": {**evaluate.environment_info()},
        "provenance": {
            "git_revision": evaluate.git_revision(),
            "git_dirty": evaluate.git_dirty(),
            "command": "python -m src.nids.risk",
        },
        "samples": per_sample,
    }


def result_index(explanation: dict[str, Any]) -> int:
    """The sample identifier recorded by explain_samples()."""
    return int(explanation.get("index", 0))


def main(
    sample_size: int = REPORT_SAMPLE_SIZE,
    sample_seed: int = REPORT_SEED,
) -> dict[str, Any]:
    """Generate reports/risk_assessment.json from the saved model."""
    import time

    t0 = time.perf_counter()
    context = explain.build_context()
    t_context = time.perf_counter() - t0

    X, y = __import__("src.nids.data", fromlist=["load_training_data"]) \
        .load_training_data(config.TRAIN_CSV)

    t0 = time.perf_counter()
    sample_X, _ = explain._report_sample_frame(X, sample_size, sample_seed)
    explanations = explain.explain_samples(context, sample_X, top_k=None)
    t_shap = time.perf_counter() - t0

    t0 = time.perf_counter()
    report = build_risk_report(
        context, X, y, sample_size, sample_seed, explanations=explanations
    )
    t_risk = time.perf_counter() - t0

    REPORTS_DIR = evaluate.REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "risk_assessment.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    counts = report["risk"]["levels"]["counts"]
    print(CAVEAT)
    print(f"\ncontext build (background only): {t_context:.2f}s")
    print(f"SHAP explanations for {len(explanations)} rows: {t_shap:.1f}s "
          f"(upstream input, computed once)")
    print(f"risk engine over {len(explanations)} rows: {t_risk*1000:.1f} ms "
          f"(~{t_risk/max(len(explanations),1)*1e6:.0f} us/sample)")
    print(f"\nlevel counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    dist = report["risk"]["score_distribution"]
    print(f"score range [{dist['min']:.1f}, {dist['max']:.1f}], "
          f"mean {dist['mean']:.2f}, median {dist['median']:.2f}")
    print(f"saved risk report to {out}")
    return report


if __name__ == "__main__":
    main()
