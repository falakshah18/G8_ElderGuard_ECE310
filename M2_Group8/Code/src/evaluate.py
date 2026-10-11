"""
ElderGuard - evaluation protocol (M2)
ECE 310 Group 8

Leave-one-session-out cross-validation.

Why not a random window split? Consecutive 1 s windows from the same capture
are highly correlated, so a random split leaks near-duplicate windows from a
session into both train and test and inflates the score. We therefore hold out
an entire session at a time: every window from the held-out session is
excluded from training. In the RadioPoints run the sessions are the
(room x occupancy) captures, so this is also leave-one-room-out in effect.

Metrics: accuracy, macro-F1, count-MAE, and the confusion matrix.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models import build_models

CLASS_NAMES = ["0_people", "1_person", "2_people"]


def count_mae(y_true, y_pred):
    """Mean absolute error on the *count* (0/1/2), not the class index."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def leave_one_session_out(X, y, session_ids, models=None, verbose=True):
    """Run LOSO CV. Returns a nested dict: model -> metrics.

    X            [n_windows, 12]
    y            [n_windows] integer class labels 0/1/2
    session_ids  [n_windows] session identifier per window
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    session_ids = np.asarray(session_ids)
    if models is None:
        models = build_models()

    unique_sessions = np.unique(session_ids)
    results = {}

    for name, model in models.items():
        all_true, all_pred = [], []
        for held in unique_sessions:
            test_mask = session_ids == held
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                continue
            model.fit(X[train_mask], y[train_mask])
            pred = model.predict(X[test_mask])
            all_true.append(y[test_mask])
            all_pred.append(pred)
            if verbose:
                print(f"  [{name}] held-out {held}: "
                      f"{np.mean(pred == y[test_mask]):.3f} acc "
                      f"({test_mask.sum()} windows)")

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        results[name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "count_mae": count_mae(y_true, y_pred),
            "confusion_matrix": cm.tolist(),
            "n_windows": int(y_true.size),
        }
        if verbose:
            r = results[name]
            print(f"[{name}] accuracy={r['accuracy']:.4f} "
                  f"macro_f1={r['macro_f1']:.4f} "
                  f"count_mae={r['count_mae']:.4f}")

    # Feature importances: refit the forest on all data for the summary figure.
    rf = build_models()["Random Forest"]
    rf.fit(X, y)
    results["_feature_importance"] = rf.feature_importances_.tolist()
    results["_class_names"] = CLASS_NAMES
    results["_n_sessions"] = int(len(unique_sessions))
    return results
