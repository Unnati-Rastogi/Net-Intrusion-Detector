"""Central configuration for the NIDS ML core."""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"

TRAIN_CSV = DATA_DIR / "Train_data.csv"   # 25,192 rows x 42 cols, has 'class'
TEST_CSV = DATA_DIR / "Test_data.csv"     # 22,544 rows x 41 cols, features only
MODEL_PATH = MODEL_DIR / "rf_ids_pipeline.joblib"

# --- Schema --------------------------------------------------------------
TARGET_COLUMN = "class"
POSITIVE_CLASS = "anomaly"   # encoded as 1
NEGATIVE_CLASS = "normal"    # encoded as 0

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# All 41 feature columns in dataset order (target 'class' excluded).
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# Constant in the training data (no information): dropped inside the pipeline.
# 'num_outbound_cmds' is constant 0 in both files; 'is_host_login' is constant 0
# in train (the API contract still accepts both fields in a raw sample).
CONSTANT_COLUMNS = ["num_outbound_cmds", "is_host_login"]

# --- Model ---------------------------------------------------------------
RANDOM_STATE = 42

# Approved after 5-fold stratified CV on Train_data.csv only:
# accuracy 0.9975 +/- 0.0013, F1(anomaly) 0.9973 +/- 0.0014.
# Do not change: this is the production classifier configuration.
RF_PARAMS = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# XGBoost comparison candidate (src/nids/compare.py). NOT the production model.
# scale_pos_weight mirrors RF's class_weight='balanced' (n_negative/n_positive).
# n_jobs is set at build time in compare.py (n_training_rows / n_estimators per
# thread would vary), so the serializable config records the seed strategy only.
XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}
