"""FastAPI layer around the NIDS prediction module.

Endpoints:
    GET  /                -> dashboard (static/index.html)
    GET  /health          -> model status, feature count, class list
    POST /predict         -> one flow record -> PredictionResponse (fast, no SHAP)
    POST /predict/batch   -> list of flow records -> BatchPredictionResponse
    POST /analyze         -> full analyst workflow for ONE record:
                             prediction -> SHAP explanation -> risk ->
                             review priority (orchestrates the existing
                             domain modules; runs no new ML)
    GET  /alerts          -> the committed prioritization report, verbatim
                             (no re-ranking happens here or in the browser)
    GET  /features        -> dashboard support: feature column metadata
    GET  /sample          -> dashboard support: canonical sample flow record
    GET  /demo/test-rows  -> dashboard support: real rows from Test_data.csv

Malformed predict/analyze requests return HTTP 422 with a flattened
"feature_errors" list alongside the standard detail, e.g.
{"feature_errors": ["features -> src_bytes: Field required", ...]}.
Unknown categorical values (e.g. an unseen 'service') are NOT errors - the
model's OneHotEncoder (handle_unknown='ignore') handles them by design.
"""

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import analyze as analyze_module
from . import config, prioritize
from .data import load_category_values, load_inference_data
from .predict import load_model, predict_connection, predict_connections
from .sample_data import SAMPLE_CONNECTION
from .schemas import (
    AlertItem,
    AlertsResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchPredictionResponse,
    BatchPredictRequest,
    DemoRowsResponse,
    FeaturesResponse,
    FlowFeatures,
    HealthResponse,
    PredictRequest,
    PredictionResponse,
    SampleResponse,
)


def _feature_errors(exc: RequestValidationError) -> list[str]:
    """Flatten pydantic validation errors into deduped human-readable strings."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err.get("loc", []) if p != "body")
        msg = err.get("msg", "invalid input")
        errors.append(f"{loc}: {msg}" if loc else msg)
    return list(dict.fromkeys(errors))  # dedupe, preserve order


def create_app() -> FastAPI:
    """Create the FastAPI app, loading the model once at startup (fail fast)."""
    app = FastAPI(
        title="ML Network Intrusion Detection API",
        description=(
            "Classifies network-flow records as normal or anomaly. "
            "A minimal dashboard is served at '/'."
        ),
        version="0.1.0",
    )
    model = load_model()  # ModelNotTrainedError at startup if untrained
    app.state.model = model

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = _feature_errors(exc)
        return JSONResponse(
            status_code=422,
            content={"detail": errors, "feature_errors": errors},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Last-resort handler: JSON 500 without leaking internals.

        Validation and known artifact problems are handled by their own
        paths; anything unexpected lands here so the client gets a stable,
        readable error instead of a bare 'Internal Server Error' page.
        """
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error; the request was not processed"},
        )

    @app.get("/", include_in_schema=False)
    def dashboard() -> HTMLResponse:
        """Serve the minimal dashboard (no template engine needed)."""
        index_path = config.STATIC_DIR / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=app.state.model is not None,
            model_path=str(config.MODEL_PATH),
            feature_count=len(model.feature_names_in_),
            classes=sorted(int(c) for c in model.classes_),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictRequest) -> PredictionResponse:
        result = predict_connection(request.features.model_dump(), model=model)
        return PredictionResponse(**result)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
        results = predict_connections(
            [features.model_dump() for features in request.features], model=model
        )
        return BatchPredictionResponse(
            count=len(results),
            predictions=[PredictionResponse(**r) for r in results],
        )

    # --- Analyst workflow (orchestrates existing domain modules) -----------

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        """Detection -> SHAP explanation -> risk -> review priority.

        Pass exactly one of `features` (an ad-hoc record) or `sample_id`
        (a record of the committed risk-assessment report). SHAP runs once
        because the user explicitly requested analyst-level analysis; the
        plain /predict endpoint stays fast and SHAP-free.
        """
        provided = [
            name
            for name, value in (
                ("features", request.features),
                ("sample_id", request.sample_id is not None),
            )
            if value
        ]
        if len(provided) != 1:
            raise HTTPException(
                status_code=422,
                detail="pass exactly one of 'features' or 'sample_id'",
            )
        try:
            if request.sample_id is not None:
                result = analyze_module.analyze_report_sample(request.sample_id)
            else:
                result = analyze_module.analyze_record(
                    request.features.model_dump(), model=model
                )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown sample_id {exc.args[0]}: not a record of the "
                    "committed risk-assessment report"
                ),
            ) from exc
        return AnalyzeResponse(**result)

    @app.get("/alerts", response_model=AlertsResponse)
    def alerts() -> AlertsResponse:
        """The precomputed prioritization report, served verbatim.

        This endpoint computes nothing: ranking, risk and priority come from
        the committed report produced by 'python -m src.nids.prioritize'.
        """
        if not prioritize.ALERT_REPORT_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "prioritization report not found; run "
                    "'python -m src.nids.prioritize' first"
                ),
            )
        try:
            report = json.loads(
                prioritize.ALERT_REPORT_PATH.read_text(encoding="utf-8")
            )
            return AlertsResponse(
                report=report["report"],
                timestamp_utc=report["timestamp_utc"],
                total_alerts=report["total_alerts"],
                counts_by_priority=report["counts_by_priority"],
                note=report["note"],
                alerts=[AlertItem(**alert) for alert in report["alerts"]],
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Unreadable or structurally wrong artifact: name the file and
            # the fix instead of leaking a traceback (or silently serving
            # nothing). The committed report passes; only a corrupted one
            # can land here.
            raise HTTPException(
                status_code=500,
                detail=(
                    f"could not read the prioritization report at "
                    f"{prioritize.ALERT_REPORT_PATH} (it may be corrupted); "
                    "regenerate it with 'python -m src.nids.prioritize'"
                ),
            ) from exc

    # --- Dashboard support endpoints --------------------------------------

    @app.get("/features", response_model=FeaturesResponse)
    def features() -> FeaturesResponse:
        """Feature column metadata (the dashboard builds its form from this)."""
        return FeaturesResponse(
            feature_columns=config.FEATURE_COLUMNS,
            categorical_columns=config.CATEGORICAL_COLUMNS,
            category_values=load_category_values(),
        )

    @app.get("/sample", response_model=SampleResponse)
    def sample() -> SampleResponse:
        """Canonical sample flow record (the dashboard's Load Sample button)."""
        return SampleResponse(features=FlowFeatures(**dict(SAMPLE_CONNECTION)))

    @app.get("/demo/test-rows", response_model=DemoRowsResponse)
    def demo_test_rows(limit: int = 10) -> DemoRowsResponse:
        """Real rows from Test_data.csv for the batch demonstration button.

        Serves at most 100 rows per request. Raises 500 with a clear message
        if the test CSV is unavailable.
        """
        limit = max(1, min(limit, 100))
        try:
            df = load_inference_data()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Test data unavailable: {exc}",
            ) from exc
        rows = df.head(limit).to_dict(orient="records")
        return DemoRowsResponse(count=len(rows), rows=[FlowFeatures(**r) for r in rows])

    # Static assets for the dashboard (app.js) under /static.
    if config.STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

    return app


app = create_app()
