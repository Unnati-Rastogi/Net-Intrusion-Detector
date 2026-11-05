# NetGuard — Network Intrusion Detection

NetGuard is a machine-learning network intrusion detection project for
reviewing exported network-flow data. Upload one CSV in the CIC-IDS2017 flow
format and the application will:

- assess whether the dataset contains suspicious activity;
- rank the flows that need attention first;
- show the model score and a plain-English reason for each flagged flow;
- provide SHAP feature evidence for a deeper technical review; and
- let you download the complete scan result.

It is an **offline CSV analysis tool**, not a live packet-capture or network
monitoring product.

## Quick start

### 1. Get the code

```bash
git clone https://github.com/Unnati-Rastogi/Net-Intrusion-Detector.git
cd Net-Intrusion-Detector
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install requirements and run

```bash
pip install -r requirements.txt
python run_api.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. The
included CIC detector is ready for scanning after installation; no training is
required for normal use.

## Using the application

1. Open **Scan dataset**.
2. Choose one CIC-compatible network-flow CSV.
3. Select **Scan dataset**.
4. Review the dataset conclusion, summary counts, and highest-scoring flows.
5. Select **Explain** beside a flow to see why it stood out. The first
   explanation is written in everyday language; the detailed table shows the
   model's SHAP evidence.
6. Download the full result CSV if needed.

A `Label` column is optional. When it is present, NetGuard also displays
accuracy, precision, and recall for that upload. Labels are never required to
make a detection decision.

### Input format

The detector expects the corrected **CIC-IDS2017 network-flow feature
schema**. A CSV must contain the same feature columns used by the CIC flow
export. Common context fields such as source/destination IP, ports, flow ID,
and timestamp are preserved in the downloadable output when present.

This project does not convert PCAP files or arbitrary datasets into CIC flow
features. Generate compatible flow records before uploading.

## Reliability

The built-in detector is a Random Forest selected from Logistic Regression,
Random Forest, and XGBoost using a time-aware CIC-IDS2017 evaluation:

| Stage | CIC-IDS2017 data |
|---|---|
| Training | Monday–Wednesday |
| Model/threshold selection | Thursday |
| Final held-out evaluation | Friday |

On the held-out Friday capture, the selected Random Forest achieved:

| Accuracy | Attack precision | Attack recall | Attack F1 |
|---:|---:|---:|---:|
| 92.48% | 97.48% | 86.08% | 91.42% |

These are controlled benchmark results for CIC-IDS2017, not a promise that
every environment or dataset will perform the same way. Use results to guide
analyst review rather than as a replacement for a security investigation.

The **Validation** page in the application shows the model comparison and the
held-out result separately from the scan workflow.

## Reproducing the CIC comparison (optional)

The original CIC-IDS2017 data is intentionally not included in this
repository because it is large. Download the corrected release from the
[CNS2022 corrected dataset page](https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html),
then place the extracted daily files in `data/CIC-IDS2017/` as:

```text
monday.csv
tuesday.csv
wednesday.csv
thursday.csv
friday.csv
```

Run the comparison and rebuild the detector:

```bash
python -m src.nids.compare_cic
python -m src.nids.cic_detector
```

The comparison trains Logistic Regression, Random Forest, and XGBoost on the
same day-based split. It may take substantial time and memory; it is not
needed to run the packaged application.

## Demonstration datasets

If you have the corrected CIC files locally, generate five ready-to-upload
scenario datasets with:

```bash
python -m src.nids.make_demo_datasets
```

They are written to `data/test_datasets/`. See
[docs/test_datasets.md](docs/test_datasets.md) for their scenarios and the
right way to present their results. The files are sampled from the held-out
Friday capture, not from the training records.

## Project layout

```text
src/nids/              API, CIC detector, model comparison, and utilities
static/                Browser interface
models/                Included detector artifact
reports/               Published evaluation reports
docs/                  Supporting methodology and demo-dataset notes
tests/                 Automated checks
```

## Development checks

```bash
python -m unittest discover -s tests -v
```

The original raw datasets, locally generated uploads, temporary server logs,
and per-user model profiles are excluded from Git. They are not needed to run
the included detector.

## Scope and limitations

- Designed for CIC-IDS2017-compatible flow CSVs.
- Performs offline analysis only; it does not capture live traffic.
- Model explanations describe why the model scored a flow as suspicious. They
  are not proof of an attack or a causal explanation of network behaviour.
- Treat flagged flows as review priorities, then validate them with normal
  security investigation practices.
