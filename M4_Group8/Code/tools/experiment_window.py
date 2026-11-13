#!/usr/bin/env python3
"""Experiment 2: window size, training balance, recording-level voting, binary.

Follows experiment_norm.py. All variants use the honest LeaveOneRecordingOut
protocol. B-normalization (rssi_mean centered on per-session median) is always
applied, since experiment 1 showed it is the strongest single improvement.

Metrics reported per configuration:
  - window-level accuracy / macro-F1 (same protocol as train_model.py)
  - recording-level accuracy: all windows of the held-out recording vote;
    correct if the majority prediction equals the recording's label.
"""

import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_csi import load_esp32_csv
from preprocess import preprocess_session
from features import featurize_session, FEATURE_NAMES
from train_model import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
RSSI_COL = IDX["rssi_mean"]

MODELS = ["Random Forest", "SVM RBF", "Logistic Regression", "Gradient Boosting"]


def featurize_sessions(data_dir, window_s, overlap=0.5):
    sessions = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        try:
            s = load_esp32_csv(path)
        except Exception:
            continue
        if s.get("label") not in (0, 1, 2):
            continue
        amp, phase = preprocess_session(s["amplitude"], s["phase"], s["fs"], do_phase=True)
        feats, _ = featurize_session(amp, phase, s.get("rssi", 0.0), s["fs"], window_s, overlap,
                                     pdr=s.get("pdr", 1.0))
        if feats.size == 0:
            continue
        feats = feats.copy()
        feats[:, RSSI_COL] -= np.median(feats[:, RSSI_COL])
        sid = os.path.basename(path)
        sessions.append({"sid": sid, "label": int(s["label"]), "X": feats})
    return sessions


def run_loso(blocks, seed=42, balance_train=True, binary=False):
    model_set = models(seed)
    results = {}
    for name in MODELS:
        estimator = model_set[name]
        truth, pred = [], []
        votes = []
        for held in blocks:
            train_blocks = [b for b in blocks if b["sid"] != held["sid"]]
            if binary:
                classes = {0 if b["label"] == 0 else 1 for b in train_blocks}
                if classes != {0, 1}:
                    continue
            else:
                if {b["label"] for b in train_blocks} != {0, 1, 2}:
                    continue
            Xtr = np.vstack([b["X"] for b in train_blocks])
            if binary:
                ytr = np.array([0 if b["label"] == 0 else 1 for b in train_blocks
                                for _ in range(b["X"].shape[0])])
                yte = 0 if held["label"] == 0 else 1
            else:
                ytr = np.concatenate([np.full(b["X"].shape[0], b["label"]) for b in train_blocks])
                yte = held["label"]
            if balance_train:
                rng = np.random.RandomState(seed)
                uniq = np.unique(ytr)
                n_min = min(int((ytr == c).sum()) for c in uniq)
                keep = []
                for c in uniq:
                    idx = np.where(ytr == c)[0]
                    if len(idx) > n_min:
                        idx = rng.choice(idx, n_min, replace=False)
                    keep.extend(idx.tolist())
                keep = np.array(sorted(keep))
                Xtr, ytr = Xtr[keep], ytr[keep]
            model = estimator
            model.fit(Xtr, ytr)
            p = model.predict(held["X"])
            truth.extend([yte] * len(p))
            pred.extend(p.tolist())
            top = Counter(p.tolist()).most_common(1)[0][0]
            votes.append(1 if top == yte else 0)
        if truth:
            truth = np.array(truth)
            pred = np.array(pred)
            results[name] = {
                "window_acc": float(accuracy_score(truth, pred)),
                "window_macro_f1": float(f1_score(truth, pred, average="macro")),
                "confusion_matrix": confusion_matrix(truth, pred).astype(int).tolist(),
                "recording_vote_acc": float(np.mean(votes)),
            }
    return results


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Data", "classroom")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Results", "window_experiment.json")
    results = {"data_dir": data_dir, "normalization": "B (rssi_mean centered on per-session median)",
               "balance_train": True, "configs": {}}
    for window_s in (1.0, 2.0, 2.5, 3.5):
        t0 = time.perf_counter()
        sessions = featurize_sessions(data_dir, window_s)
        counts = {c: sum(s["label"] == c for s in sessions) for c in (0, 1, 2)}
        scores3 = run_loso(sessions, binary=False)
        scores_bin = run_loso(sessions, binary=True)
        dt = time.perf_counter() - t0
        key = f"window_{window_s}s"
        results["configs"][key] = {"sessions_per_class": counts, "loso_3class": scores3,
                                   "loso_binary": scores_bin}
        print(f"\n===== window {window_s}s ({dt:.0f}s) sessions/class={counts} =====")
        print("-- 3-class --")
        for n, sc in sorted(scores3.items(), key=lambda kv: -kv[1]["window_acc"]):
            print(f"  {n:22s} win_acc={sc['window_acc']:.3f} macroF1={sc['window_macro_f1']:.3f} "
                  f"rec_vote={sc['recording_vote_acc']:.3f}")
        print("-- binary (occupied/empty) --")
        for n, sc in sorted(scores_bin.items(), key=lambda kv: -kv[1]["window_acc"]):
            print(f"  {n:22s} win_acc={sc['window_acc']:.3f} macroF1={sc['window_macro_f1']:.3f} "
                  f"rec_vote={sc['recording_vote_acc']:.3f}")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
