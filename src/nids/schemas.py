"""Pydantic request/response schemas for the NIDS API.

FlowFeatures lists one typed field per dataset column so the API contract is
readable and self-documenting. extra="forbid" makes unknown fields a
validation error; missing fields and wrong types are rejected automatically.
A test pins these fields to the real dataset schema via config.FEATURE_COLUMNS.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlowFeatures(BaseModel):
    """One raw network-flow record (the 41 dataset feature columns)."""

    model_config = ConfigDict(extra="forbid")

    duration: int
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float


class PredictRequest(BaseModel):
    features: FlowFeatures


class BatchPredictRequest(BaseModel):
    features: list[FlowFeatures] = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    prediction: str
    label: int
    probability_normal: float
    probability_anomaly: float
    confidence: float


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    feature_count: int
    classes: list[int]


# --- Analyst analysis (/analyze) -------------------------------------------


class EvidenceItem(BaseModel):
    """One per-feature SHAP attribution (the model's output, not causality)."""

    feature: str
    value: float | int | str | None
    shap_value: float
    direction: str          # 'increases_anomaly_score' | 'decreases_anomaly_score'
    abs_contribution: float


class AnalyzeRequest(BaseModel):
    """One analyst analysis: an ad-hoc feature vector XOR a known report sample.

    Exactly one of `features` / `sample_id` must be provided. `sample_id`
    refers to a record of the committed risk-assessment report; its lineage
    (sample_id + source_row_id) is then preserved in the response. For an
    ad-hoc vector both ids are null unless the exact feature vector uniquely
    matches a known report sample.
    """

    features: FlowFeatures | None = None
    sample_id: int | None = Field(default=None, ge=0)


class AnalyzeResponse(BaseModel):
    """Full analyst workflow for ONE record: detection -> risk -> priority.

    Terminology contract: anomaly_score is the model's P(anomaly) output
    (a model score, not an attack probability); risk_score is the
    model-derived prioritization score (not attack severity or impact);
    priority is the analyst review priority (P1 = highest review priority).
    """

    prediction: str
    label: int
    anomaly_score: float
    probability_normal: float
    probability_anomaly: float
    confidence: float
    risk_score: float
    risk_level: str
    priority: str
    sample_id: int | None = None
    source_row_id: int | None = None
    explanation: list[EvidenceItem]
    methodology: dict[str, str]


class AlertItem(BaseModel):
    """One record of the committed prioritization report, served verbatim.

    top_evidence keeps the report's own field names (shap_contribution, ...)
    so the served payload is byte-faithful to the committed artifact.
    """

    rank: int
    sample_id: int
    source_row_id: int | None = None
    prediction: str
    anomaly_score: float
    risk_score: float
    risk_level: str
    priority: str
    top_evidence: list[dict[str, Any]]


class AlertsResponse(BaseModel):
    """The precomputed prioritization report (the backend stays authoritative;
    the dashboard must not re-rank anything in JavaScript)."""

    report: str
    timestamp_utc: str
    total_alerts: int
    counts_by_priority: dict[str, int]
    note: str
    alerts: list[AlertItem]


# --- Dashboard support schemas -------------------------------------------


class FeaturesResponse(BaseModel):
    feature_columns: list[str]
    categorical_columns: list[str]
    category_values: dict[str, list[str]]


class SampleResponse(BaseModel):
    features: FlowFeatures


class DemoRowsResponse(BaseModel):
    count: int
    rows: list[FlowFeatures]
