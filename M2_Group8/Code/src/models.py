"""
ElderGuard - occupancy models (M2)
ECE 310 Group 8

Three models, matching the comparison in the report:
  * ThresholdBaseline - non-AI reference on the single best feature
                        (amplitude variance), the classic CSI occupancy rule.
  * Random Forest     - 300 trees, the best model at M2.
  * MLP               - 64-32 hidden units, standardised inputs.

All three expose fit / predict so evaluate.py can treat them identically.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_INDEX = {name: i for i, name in enumerate(
    ["amp_var_mean", "amp_var_std", "amp_mean", "amp_range_mean",
     "subcarrier_corr_mean", "phase_var_mean", "doppler_energy",
     "doppler_peak_freq", "doppler_centroid", "rssi_mean", "rssi_std", "pdr"]
)}


class ThresholdBaseline:
    """Single-feature threshold rule.

    The feature is amplitude variance (feature 0). Thresholds are learned on
    the training split by minimising 3-class error over the two cut points
    that separate 0 / 1 / 2 people. This is the honest non-AI reference the
    learned models must beat.
    """

    def __init__(self, feature="amp_var_mean"):
        self.feature = feature
        self.t_low_ = None
        self.t_high_ = None

    def fit(self, X, y):
        col = X[:, FEATURE_INDEX[self.feature]]
        y = np.asarray(y)
        # Candidate cut points = midpoints of sorted unique values.
        vals = np.unique(col)
        cands = (vals[:-1] + vals[1:]) / 2.0
        if cands.size > 200:  # keep the search cheap on large datasets
            cands = np.quantile(cands, np.linspace(0, 1, 200))
        best = (None, None, -1.0)
        for i, lo in enumerate(cands):
            # second cut only above the first
            for hi in cands[i + 1:]:
                pred = np.digitize(col, [lo, hi])
                acc = np.mean(pred == y)
                if acc > best[2]:
                    best = (lo, hi, acc)
        self.t_low_, self.t_high_, _ = best
        if self.t_low_ is None:  # degenerate single-value feature
            self.t_low_ = self.t_high_ = float(np.median(col))
        return self

    def predict(self, X):
        col = X[:, FEATURE_INDEX[self.feature]]
        return np.digitize(col, [self.t_low_, self.t_high_])


def build_random_forest(n_estimators=300, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )


def build_mlp(hidden=(64, 32), random_state=42):
    """MLP wrapped in a scaler so raw feature magnitudes do not dominate."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_models():
    """Fresh, unfitted model dict used by run_pipeline / evaluate."""
    return {
        "Threshold (amp-variance)": ThresholdBaseline("amp_var_mean"),
        "Random Forest": build_random_forest(),
        "MLP": build_mlp(),
    }
