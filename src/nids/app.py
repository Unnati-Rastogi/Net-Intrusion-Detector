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
    GET  /evaluation/holdout -> saved labeled benchmark replay metrics
    GET  /evaluation/model-comparison -> saved LR/RF/XGBoost labeled-test comparison
    GET  /evaluation/cross-validation -> saved five-fold training-data comparison
    GET  /evaluation/cic-ids2017 -> latest fresh, day-split CIC model comparison
    POST /evaluation/cic-ids2017/run -> start a fresh local candidate comparison
    GET  /evaluation/cic-ids2017/status -> comparison progress for the dashboard
    POST /profiles/cic/train -> upload labeled CIC flows and train a dataset profile
    POST /profiles/cic/{id}/evaluate -> compare that profile on another labeled scenario
    POST /profiles/cic/{id}/predict -> scan an unlabeled same-schema CSV
    POST /profiles/cic/{id}/explain -> SHAP explanation for one selected-profile flow
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
import os
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import analyze as analyze_module
from . import holdout
from . import cic_detector, config, prioritize, profiles
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

CIC_COMPARISON_PATH = config.PROJECT_ROOT / "reports" / "cic_ids2017_model_comparison.json"
CIC_TRAINING_LOG_PATH = config.PROJECT_ROOT / "reports" / "cic_ids2017_training.log"
_CIC_TRAINING_LOCK = Lock()
_CIC_TRAINING_PROCESS: subprocess.Popen | None = None
_PROFILE_JOB_LOCK = Lock()
_PROFILE_JOBS: dict[str, dict] = {}
_PROFILE_TRAINER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nids-profile-trainer")
MAX_PROFILE_UPLOAD_BYTES = 1024 * 1024 * 1024


async def _save_csv_request(request: Request) -> Path:
    """Stream a raw CSV request body to a temporary file without buffering it."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"text/csv", "application/csv", "application/octet-stream"}:
        raise HTTPException(
            status_code=415,
            detail="send the CSV file as the request body with Content-Type: text/csv",
        )
    handle = tempfile.NamedTemporaryFile(prefix="nids-upload-", suffix=".csv", delete=False)
    path = Path(handle.name)
    size = 0
    try:
        with handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_PROFILE_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="CSV uploads are limited to 1 GB")
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="uploaded CSV is empty")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _run_profile_training(job_id: str, csv_path: Path, name: str) -> None:
    with _PROFILE_JOB_LOCK:
        _PROFILE_JOBS[job_id]["state"] = "running"
        _PROFILE_JOBS[job_id]["progress"] = "Training all candidate models on the labeled upload."
    try:
        report = profiles.train_profile(csv_path, name)
        with _PROFILE_JOB_LOCK:
            _PROFILE_JOBS[job_id].update(
                state="completed",
                progress="Comparison complete; the validation-selected profile is saved.",
                profile_id=report["profile_id"],
            )
    except Exception as exc:
        with _PROFILE_JOB_LOCK:
            _PROFILE_JOBS[job_id].update(state="failed", progress=str(exc)[:1000])
    finally:
        csv_path.unlink(missing_ok=True)


def _cic_training_status() -> dict:
    """Return the most recent local CIC comparison job state and log line."""
    with _CIC_TRAINING_LOCK:
        process = _CIC_TRAINING_PROCESS
        return_code = process.poll() if process is not None else None
        if process is None:
            state = "idle"
        elif return_code is None:
            state = "running"
        else:
            state = "completed" if return_code == 0 else "failed"
    try:
        lines = CIC_TRAINING_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        progress = next((line.strip() for line in reversed(lines) if line.strip()), "")
    except OSError:
        progress = ""
    return {"state": state, "exit_code": return_code, "progress": progress}


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

    @app.get("/evaluation/holdout")
    def holdout_evaluation() -> dict:
        """Return the latest labeled benchmark replay report, if generated."""
        if not holdout.REPORT_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "labeled benchmark report not found; place the KDDTest+ labeled "
                    "file as described in the README, then run 'python -m src.nids.holdout'"
                ),
            )
        try:
            report = json.loads(holdout.REPORT_PATH.read_text(encoding="utf-8"))
            if report.get("report") != "nids_labeled_holdout_evaluation":
                raise ValueError("unexpected report identity")
            provenance = report.get("provenance", {})
            if (
                provenance.get("model_sha256") != holdout.sha256_file(config.MODEL_PATH)
                or provenance.get("features_sha256") != holdout.sha256_file(config.TEST_CSV)
            ):
                raise HTTPException(
                    status_code=409,
                    detail="saved evaluation does not match the current model or data; regenerate it with 'python -m src.nids.holdout'",
                )
            return report
        except HTTPException:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=500,
                detail="could not read the labeled benchmark report; regenerate it with 'python -m src.nids.holdout'",
            ) from exc

    @app.get("/evaluation/model-comparison")
    def model_comparison() -> dict:
        """Return the saved labeled-holdout comparison of LR, RF, and XGBoost."""
        comparison_path = config.PROJECT_ROOT / "reports" / "holdout_model_comparison.json"
        if not comparison_path.exists():
            raise HTTPException(
                status_code=404,
                detail="labeled model comparison report not found; run 'python -m src.nids.compare_holdout' first",
            )
        try:
            report = json.loads(comparison_path.read_text(encoding="utf-8"))
            if report.get("report") != "nids_full_train_holdout_model_comparison":
                raise ValueError("unexpected report identity")
            return report
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=500,
                detail="could not read the labeled model comparison report; regenerate it with 'python -m src.nids.compare_holdout'",
            ) from exc

    @app.get("/evaluation/cic-ids2017")
    def cic_ids2017_comparison() -> dict:
        """Return the latest fresh CIC-IDS2017 day-split model comparison."""
        if not CIC_COMPARISON_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="CIC-IDS2017 comparison is not available; run 'python -m src.nids.compare_cic' first",
            )
        try:
            report = json.loads(CIC_COMPARISON_PATH.read_text(encoding="utf-8"))
            if report.get("report") != "nids_cic_ids2017_day_split_model_comparison":
                raise ValueError("unexpected CIC-IDS2017 report identity")
            return report
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=500,
                detail="could not read the CIC-IDS2017 comparison; regenerate it with 'python -m src.nids.compare_cic'",
            ) from exc

    @app.post("/evaluation/cic-ids2017/run", status_code=202)
    def run_cic_ids2017_comparison() -> dict:
        """Start a fresh, local comparison of all CIC candidate models."""
        global _CIC_TRAINING_PROCESS
        data_dir = config.DATA_DIR / "CIC-IDS2017"
        required_files = ["monday.csv", "tuesday.csv", "wednesday.csv", "thursday.csv", "friday.csv"]
        missing = [name for name in required_files if not (data_dir / name).is_file()]
        if missing:
            raise HTTPException(
                status_code=409,
                detail="CIC-IDS2017 daily files are missing: " + ", ".join(missing),
            )
        with _CIC_TRAINING_LOCK:
            if _CIC_TRAINING_PROCESS is not None and _CIC_TRAINING_PROCESS.poll() is None:
                return {"state": "running", "progress": "A comparison is already in progress."}
            CIC_TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                with CIC_TRAINING_LOG_PATH.open("w", encoding="utf-8") as log_file:
                    _CIC_TRAINING_PROCESS = subprocess.Popen(
                        [sys.executable, "-u", "-m", "src.nids.compare_cic"],
                        cwd=config.PROJECT_ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="could not start the CIC model comparison with this Python environment",
                ) from exc
        return {"state": "running", "progress": "Loading the CIC-IDS2017 daily files."}

    @app.get("/evaluation/cic-ids2017/status")
    def cic_ids2017_training_status() -> dict:
        """Report progress for a dashboard-triggered CIC comparison run."""
        return _cic_training_status()

    @app.get("/detections/cic/status")
    def cic_detector_status() -> dict:
        """Describe the ready-to-use detector for one-file CIC scans."""
        try:
            _, metadata = cic_detector.load_default_detector()
            return {"ready": True, **metadata}
        except FileNotFoundError:
            return {
                "ready": False,
                "detail": "The default CIC detector has not been built yet.",
            }

    @app.post("/detections/cic/scan")
    async def scan_cic_dataset(request: Request) -> dict:
        """Scan one uploaded CIC flow CSV with the default trained detector."""
        csv_path = await _save_csv_request(request)
        run_id = uuid.uuid4().hex
        output_path = profiles.REPORT_DIR / "scans" / f"{run_id}.csv"
        try:
            result = await run_in_threadpool(cic_detector.scan_csv, csv_path, output_path)
            return {
                **result,
                "run_id": run_id,
                "download_url": f"/detections/cic/results/{run_id}",
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            csv_path.unlink(missing_ok=True)

    @app.get("/detections/cic/results/{run_id}")
    def download_cic_scan(run_id: str) -> FileResponse:
        if len(run_id) != 32 or any(ch not in "0123456789abcdef" for ch in run_id.lower()):
            raise HTTPException(status_code=404, detail="scan result was not found")
        path = profiles.REPORT_DIR / "scans" / f"{run_id}.csv"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="scan result was not found")
        return FileResponse(path, media_type="text/csv", filename="intrusion-scan-results.csv")

    @app.post("/detections/cic/explain")
    async def explain_cic_detection(request: Request) -> dict:
        """Return SHAP evidence for one CIC flow using the default detector."""
        try:
            features = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="send one JSON object of CIC feature values") from exc
        if not isinstance(features, dict):
            raise HTTPException(status_code=422, detail="request body must be a feature-value object")
        try:
            return await run_in_threadpool(cic_detector.explain_record, features)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/profiles/cic/train", status_code=202)
    async def train_cic_profile(request: Request, name: str = "Uploaded CIC scenario") -> dict:
        """Train and compare all candidates from a labeled CIC-schema CSV upload.

        Send the CSV bytes as the request body with Content-Type: text/csv.
        The saved profile is specific to this dataset and feature schema.
        """
        csv_path = await _save_csv_request(request)
        job_id = uuid.uuid4().hex
        with _PROFILE_JOB_LOCK:
            _PROFILE_JOBS[job_id] = {
                "job_id": job_id,
                "state": "queued",
                "progress": "Upload received; comparison is queued.",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        try:
            _PROFILE_TRAINER.submit(_run_profile_training, job_id, csv_path, name[:80])
        except Exception as exc:
            csv_path.unlink(missing_ok=True)
            with _PROFILE_JOB_LOCK:
                _PROFILE_JOBS.pop(job_id, None)
            raise HTTPException(status_code=500, detail="could not queue model training") from exc
        return {"job_id": job_id, "state": "queued", "status_url": f"/profiles/cic/jobs/{job_id}"}

    @app.get("/profiles/cic/jobs/{job_id}")
    def cic_profile_job(job_id: str) -> dict:
        with _PROFILE_JOB_LOCK:
            job = _PROFILE_JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="training job was not found")
            return dict(job)

    @app.get("/profiles/cic")
    def list_cic_profiles() -> list[dict]:
        if not profiles.REPORT_DIR.exists():
            return []
        result = []
        for path in profiles.REPORT_DIR.glob("*.json"):
            try:
                result.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda item: item.get("created_at_utc", ""), reverse=True)

    @app.get("/profiles/cic/{profile_id}")
    def cic_profile_details(profile_id: str) -> dict:
        try:
            _, report = profiles.load_profile(profile_id)
            return report
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="dataset profile was not found") from exc

    @app.post("/profiles/cic/{profile_id}/evaluate")
    async def evaluate_cic_profile(profile_id: str, request: Request) -> dict:
        """Compare saved candidates on a separate labeled scenario CSV."""
        csv_path = await _save_csv_request(request)
        try:
            return await run_in_threadpool(profiles.evaluate_profile, profile_id, csv_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            csv_path.unlink(missing_ok=True)

    @app.post("/profiles/cic/{profile_id}/predict")
    async def predict_cic_profile(profile_id: str, request: Request) -> dict:
        """Scan an unlabeled CIC-schema CSV and return a downloadable result file."""
        if len(profile_id) != 32 or any(ch not in "0123456789abcdef" for ch in profile_id.lower()):
            raise HTTPException(status_code=404, detail="dataset profile was not found")
        csv_path = await _save_csv_request(request)
        run_id = uuid.uuid4().hex
        output_dir = profiles.REPORT_DIR / "predictions" / profile_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{run_id}.csv"
        try:
            result = await run_in_threadpool(
                profiles.predict_profile, profile_id, csv_path, output_path
            )
            return {**result, "run_id": run_id, "download_url": f"/profiles/cic/{profile_id}/predictions/{run_id}"}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            csv_path.unlink(missing_ok=True)

    @app.get("/profiles/cic/{profile_id}/predictions/{run_id}")
    def download_cic_predictions(profile_id: str, run_id: str) -> FileResponse:
        # Validate both IDs before joining them into a path.
        if (
            len(profile_id) != 32
            or len(run_id) != 32
            or any(ch not in "0123456789abcdef" for ch in (profile_id + run_id).lower())
        ):
            raise HTTPException(status_code=404, detail="prediction result was not found")
        path = profiles.REPORT_DIR / "predictions" / profile_id / f"{run_id}.csv"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="prediction result was not found")
        return FileResponse(path, media_type="text/csv", filename="intrusion-predictions.csv")

    @app.post("/profiles/cic/{profile_id}/explain")
    async def explain_cic_profile(profile_id: str, request: Request) -> dict:
        """Explain one record with SHAP under the profile's selected model."""
        try:
            features = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="send one JSON object of CIC feature values") from exc
        if not isinstance(features, dict):
            raise HTTPException(status_code=422, detail="request body must be a feature-value object")
        try:
            return await run_in_threadpool(profiles.explain_profile, profile_id, features)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/evaluation/cross-validation")
    def cross_validation_comparison() -> dict:
        """Return the saved five-fold comparison on labelled training data."""
        comparison_path = config.PROJECT_ROOT / "reports" / "model_comparison.json"
        if not comparison_path.exists():
            raise HTTPException(
                status_code=404,
                detail="cross-validation comparison report not found; run 'python -m src.nids.compare' first",
            )
        try:
            report = json.loads(comparison_path.read_text(encoding="utf-8"))
            if (
                report.get("report") != "nids_model_comparison"
                or report.get("evaluation", {}).get("kind")
                != "model_comparison_cross_validation_on_training_distribution"
            ):
                raise ValueError("unexpected cross-validation report identity")
            return report
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=500,
                detail="could not read the cross-validation report; regenerate it with 'python -m src.nids.compare'",
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
