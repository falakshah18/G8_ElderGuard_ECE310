#!/usr/bin/env python3
"""Train and evaluate the 0/1/2 CSI classifier from labeled capture sessions.

Validation is grouped by whole recording.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from parse_csi import load_esp32_csv
from run_pipeline import build_windows
from features import FEATURE_NAMES


def models(seed=42):
    return {
        "Dummy most_frequent": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVM RBF": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, l2_regularization=1.0, random_state=seed),
        "KNN 5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "Data", "classroom"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "Results"))
    ap.add_argument("--port", default="COM15", help="documented receiver port; training is offline")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sessions = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))):
        try:
            session = load_esp32_csv(path)
        except Exception as exc:
            print(f"SKIP {os.path.basename(path)}: {exc}")
            continue
        if session.get("label") not in (0, 1, 2):
            print(f"SKIP unlabeled capture {os.path.basename(path)}")
            continue
        print(f"LOAD {os.path.basename(path)} label={session['label']} "
              f"frames={len(session['amplitude'])} subcarriers={session['amplitude'].shape[1]} "
              f"fs={session['fs']:.2f}Hz ({session.get('fs_source')})")
        sessions.append(session)

    counts = {c: sum(s["label"] == c for s in sessions) for c in (0, 1, 2)}
    if not sessions or any(counts[c] == 0 for c in (0, 1, 2)):
        raise SystemExit(f"Need labeled real captures for all classes 0,1,2; found {counts}")

    X, y, groups = build_windows(sessions, args.window, args.overlap, max_people=2)
    print(f"Window matrix: {X.shape}; sessions/class={counts}")
    enough_groups = all(counts[c] >= 2 for c in (0, 1, 2))
    results = {"validation_protocol": "LeaveOneRecordingOut", "validation_available": enough_groups,
               "recordings_per_class": counts, "window_count": int(len(y)),
               "class_mapping": {"0": "empty", "1": "one person", "2": "two people"},
               "feature_names": FEATURE_NAMES, "window_seconds": args.window,
               "overlap": args.overlap, "random_seed": args.seed,
               "session_baseline_normalization": {
                   "features": ["rssi_mean"],
                   "applied": True,
                   "rationale": ("rssi_mean is expressed relative to the median link level of its "
                                 "own recording (per-session calibration baseline). Absolute RSSI "
                                 "encodes rig geometry (board positions / gains), which changed "
                                 "between recording batches; the deviation is the physical "
                                 "presence signal and mirrors the startup link calibration the "
                                 "deployed system performs."),
               },
               "limitation": None if enough_groups else "Only one or more classes have a single recording; no held-out session can test generalization."}
    candidate_models = models(args.seed)
    scores = {}

    if enough_groups:
        logo = LeaveOneGroupOut()
        for name, estimator in candidate_models.items():
            truth, pred = [], []
            fold_rows = []
            for train_idx, test_idx in logo.split(X, y, groups):
                if set(np.unique(y[train_idx])) != {0, 1, 2}:
                    continue
                model = clone(estimator)
                start = time.perf_counter()
                model.fit(X[train_idx], y[train_idx])
                fit_s = time.perf_counter() - start
                start = time.perf_counter()
                p = model.predict(X[test_idx])
                infer_ms = (time.perf_counter() - start) * 1000 / len(test_idx)
                truth.extend(y[test_idx].tolist())
                pred.extend(p.tolist())
                fold_rows.append({"held_session": str(groups[test_idx[0]]), "fit_seconds": fit_s, "inference_ms_per_window": infer_ms})
            if truth:
                scores[name] = {"accuracy": accuracy_score(truth, pred), "macro_f1": f1_score(truth, pred, average="macro"),
                                "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
                                "classification_report": classification_report(truth, pred, labels=[0, 1, 2], target_names=["0", "1", "2"], output_dict=True, zero_division=0),
                                "folds": fold_rows}
        eligible = {name: value for name, value in scores.items() if not name.startswith("Dummy")}
        chosen = max(eligible, key=lambda name: (eligible[name]["macro_f1"], eligible[name]["accuracy"])) if eligible else "Random Forest"
        results["validation_scores"] = scores
        results["selection_basis"] = "highest leave-one-recording-out macro-F1, then accuracy"
    else:
        chosen = "Random Forest"
        results["validation_scores"] = None
        results["selection_basis"] = "predeclared baseline; no independent validation possible"

    final_model = clone(candidate_models[chosen])
    final_model.fit(X, y)
    os.makedirs(args.out_dir, exist_ok=True)
    import joblib
    model_path = os.path.join(args.out_dir, "model_count.joblib")
    joblib.dump(final_model, model_path)
    results.update({"selected_model": chosen, "model_path": model_path,
                    "model_size_bytes": os.path.getsize(model_path), "training_window_count": int(len(y)),
                    "training_recordings": [os.path.basename(s["session_id"]) for s in sessions],
                    "provisional": not enough_groups,
                    "probabilities_calibrated": False,
                    "notes": "CSI windows within a recording are correlated. Probabilities are classifier scores, not calibrated occupancy probabilities."})
    with open(os.path.join(args.out_dir, "training_report.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    config = {"features": FEATURE_NAMES, "window_seconds": args.window, "overlap": args.overlap,
              "preprocessing": "src.preprocess.preprocess_session then src.features.featurize_session; rssi_mean centered on per-session median (calibration baseline)",
              "sample_rate_source": "ESP32 receiver timestamp median interval when available; filename tag fallback"}
    with open(os.path.join(args.out_dir, "feature_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    metadata = {"model": chosen, "classes": [0, 1, 2], "provisional": not enough_groups,
                "recordings_per_class": counts, "dataset_dir": os.path.abspath(args.data_dir)}
    with open(os.path.join(args.out_dir, "model_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Saved {chosen} -> {model_path}")


if __name__ == "__main__":
    main()
