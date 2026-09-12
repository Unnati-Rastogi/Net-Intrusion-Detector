"""Preprocessing + model pipeline for the NIDS ML core.

One sklearn Pipeline is the single source of truth for how a raw
NSL-KDD-style connection record becomes a prediction:

1. 'const_drop'   : remove the two columns that are constant in training
                    (num_outbound_cmds, is_host_login) - no information.
2. 'categorical'  : one-hot encode protocol_type / service / flag with
                    handle_unknown='ignore' (safe for unseen categories such
                    as service='tftp_u' that appear only in test data).
3. remainder      : all other numeric columns pass through unchanged -
                    tree models need no scaling, data has zero NaNs.
4. classifier     : RandomForest (200 trees, class_weight='balanced',
                    random_state=42) - the approved baseline.

Because preprocessing lives inside the pipeline, it is always fitted on
training data only, within every evaluation process (CV included).
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config


def build_pipeline() -> Pipeline:
    """Build the (unfitted) preprocessing + RandomForest pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("const_drop", "drop", config.CONSTANT_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", RandomForestClassifier(**config.RF_PARAMS)),
        ]
    )
