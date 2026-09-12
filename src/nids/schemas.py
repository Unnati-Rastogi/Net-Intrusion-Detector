"""Pydantic request/response schemas for the NIDS API.

FlowFeatures lists one typed field per dataset column so the API contract is
readable and self-documenting. extra="forbid" makes unknown fields a
validation error; missing fields and wrong types are rejected automatically.
A test pins these fields to the real dataset schema via config.FEATURE_COLUMNS.
"""

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
