#!/usr/bin/env python3
"""Experiment: does session-baseline normalization improve LOSO accuracy?

Motive: absolute RSSI / amplitude levels encode the rig geometry (board
positions, gains), which changed between recording batches. A deployed
system calibrates against its own empty-room baseline at startup, so the
physically meaningful quantity is the DEVIATION from the session baseline,
not the absolute level. This script tests that hypothesis honestly under
the same LeaveOneRecordingOut protocol used by train_model.py.

No data is modified; results are printed and saved to JSON.
"""

import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_csi import load_esp32_csv
from run_pipeline import build_windows  # noqa: F401  (kept for parity check)
from preprocess import preprocess_session
from features import featurize_session, FEATURE_NAMES
from train_model import models

IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def featurize_sessions(data_dir):
    sessions = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        try:
            s = load_esp32_csv(path)
        except Exception as exc:
            print(f"SKIP {os.path.basename(path)}: {exc}")
            continue
        if s.get("label") not in (0, 1, 2):
            continue
        amp, phase = preprocess_session(s["amplitude"], s["phase"], s["fs"], do_phase=True)
        feats, _ = featurize_session(amp, phase, s.get("rssi", 0.0), s["fs"], 1.0, 0.5,
                                     pdr=s.get("pdr", 1.0))
        if feats.size == 0:
            continue
        sid = os.path.basename(path)
        sessions.append({"sid": sid, "label": int(s["label"]), "X": feats})
        print(f"FEAT {sid} label={s['label']} windows={feats.shape[0]}")
    return sessions


def center_columns(block, cols):
    out = block.copy()
    for c in cols:
        out[:, c] = out[:, c] - np.median(out[:, c])
    return out


VARIANTS = {
    "A_none (current pipeline)": None,
    "B_center_rssi": ["rssi_mean"],
    "C_center_rssi+amp_mean": ["rssi_mean", "amp_mean"],
    "D_center_all_levels": ["rssi_mean", "amp_mean", "amp_range_mean",
                            "doppler_energy", "amp_var_mean", "amp_var_std"],
    "E_center_rssi+phase": ["rssi_mean", "phase_var_mean"],
}


def run_loso(session_blocks, seed=42, balance_train=False, drop_sessions=()):
    """session_blocks: list of dicts {sid,label,X}. Returns per-model scores."""
    blocks = [b for b in session_blocks if b["sid"] not in drop_sessions]
    model_set = models(seed)
    scores = {}
    for name, estimator in model_set.items():
        truth, pred = [], []
        cm = np.zeros((3, 3), dtype=int)
        for held in blocks:
            train_blocks = [b for b in blocks if b["sid"] != held["sid"]]
            labs = {b["label"] for b in train_blocks}
            if labs != {0, 1, 2}:
                continue  # fold would leave a class unrepresented in training
            Xtr = np.vstack([b["X"] for b in train_blocks])
            ytr = np.concatenate([np.full(b["X"].shape[0], b["label"]) for b in train_blocks])
            if balance_train:
                rng = np.random.RandomState(seed)
                counts = {c: int((ytr == c).sum()) for c in (0, 1, 2)}
                n_min = min(counts.values())
                keep = []
                for c in (0, 1, 2):
                    idx = np.where(ytr == c)[0]
                    if len(idx) > n_min:
                        idx = rng.choice(idx, n_min, replace=False)
                    keep.extend(idx.tolist())
                keep = np.array(sorted(keep))
                Xtr, ytr = Xtr[keep], ytr[keep]
            Xte = held["X"]
            model = estimator
            model.fit(Xtr, ytr)
            p = model.predict(Xte)
            truth.extend([held["label"]] * len(p))
            pred.extend(p.tolist())
        if truth:
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
            scores[name] = {
                "accuracy": float(accuracy_score(truth, pred)),
                "macro_f1": float(f1_score(truth, pred, average="macro")),
                "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
            }
    return scores


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "Data", "classroom")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Results", "normalization_experiment.json")
    sessions = featurize_sessions(data_dir)
    counts = {c: sum(s["label"] == c for s in sessions) for c in (0, 1, 2)}
    print(f"sessions/class={counts}")

    results = {"data_dir": data_dir, "sessions_per_class": counts, "variants": {}}
    for vname, cols in VARIANTS.items():
        blocks = []
        for s in sessions:
            X = s["X"]
            if cols:
                X = center_columns(X, [IDX[c] for c in cols])
            blocks.append({"sid": s["sid"], "label": s["label"], "X": X})
        t0 = time.perf_counter()
        scores = run_loso(blocks, balance_train=False)
        dt = time.perf_counter() - t0
        best = max((n for n in scores if not n.startswith("Dummy")),
                   key=lambda n: (scores[n]["macro_f1"], scores[n]["accuracy"]))
        results["variants"][vname] = {"scores": scores, "best": best, "seconds": dt}
        print(f"\n=== {vname}  ({dt:.0f}s) ===")
        for n, sc in sorted(scores.items(), key=lambda kv: -kv[1]["accuracy"]):
            print(f"  {n:24s} acc={sc['accuracy']:.3f} macroF1={sc['macro_f1']:.3f}")
        print(f"  BEST: {best}  acc={scores[best]['accuracy']:.3f}")
        dummy = scores.get("Dummy most_frequent", {}).get("accuracy", 0)
        print(f"  (dummy most_frequent acc={dummy:.3f})")

    # best normalization + training-window balancing
    best_v = max(results["variants"], key=lambda v: results["variants"][v]["scores"][results["variants"][v]["best"]]["accuracy"])
    cols = VARIANTS[best_v]
    blocks = []
    for s in sessions:
        X = s["X"]
        if cols:
            X = center_columns(X, [IDX[c] for c in cols])
        blocks.append({"sid": s["sid"], "label": s["label"], "X": X})
    scores_bal = run_loso(blocks, balance_train=True)
    results["variants"][best_v + " + balanced_train"] = {"scores": scores_bal}
    print(f"\n=== {best_v} + balanced_train ===")
    for n, sc in sorted(scores_bal.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {n:24s} acc={sc['accuracy']:.3f} macroF1={sc['macro_f1']:.3f}")

    # best normalization, dropping the anomalous 2p_t07 (rssi -41.4, stronger
    # than any empty-room baseline -> radio-signature contradicts its label)
    blocks_no_t07 = [b for b in blocks if b["sid"] != "roomA_2p_t07.csv"]
    scores_no = run_loso(blocks_no_t07)
    results["variants"][best_v + " + drop_2p_t07"] = {"scores": scores_no}
    print(f"\n=== {best_v} + drop_2p_t07 ===")
    for n, sc in sorted(scores_no.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {n:24s} acc={sc['accuracy']:.3f} macroF1={sc['macro_f1']:.3f}")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
