
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import FEATURE_NAMES

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


class ThresholdBaseline:

    def __init__(self, feature="amp_var_mean"):
        self.feature = feature
        self.t_low_ = None
        self.t_high_ = None

    def fit(self, X, y):
        col = X[:, FEATURE_INDEX[self.feature]]
        y = np.asarray(y)

        vals = np.unique(col)
        cands = (vals[:-1] + vals[1:]) / 2.0
        if cands.size > 200:
            cands = np.quantile(cands, np.linspace(0, 1, 200))
        best = (None, None, -1.0)
        for i, lo in enumerate(cands):

            for hi in cands[i + 1:]:
                pred = np.digitize(col, [lo, hi])
                acc = np.mean(pred == y)
                if acc > best[2]:
                    best = (lo, hi, acc)
        self.t_low_, self.t_high_, _ = best
        if self.t_low_ is None:
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
    return {
        "Threshold (amp-variance)": ThresholdBaseline("amp_var_mean"),
        "Random Forest": build_random_forest(),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            l2_regularization=1.0, random_state=42,
        ),
        "MLP": build_mlp(),
    }
