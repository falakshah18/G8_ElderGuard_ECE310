#!/usr/bin/env python3
"""ElderGuard M4 metrics generator - one command, all M4 evidence files.

Produces (under --out):
  m4_counting.json      counting model: LOSO (>=2 rec/class) or temporal holdout
  m4_ablation.json      12-feature M3 baseline vs 16-feature M4 set, same protocol
  m4_range.json         binary accuracy + RSSI vs board distance (d1m/d3m/d5m files)
  m4_reliability.json   per-trial accuracy spread (needs >=2 trials per class)
  m4_error_analysis.json recorded failure cases (still person / door open)
  M4_METRICS.md         human-readable summary of all of the above

Usage:  python tools/m4_metrics.py --data-dir ../Data/classroom --out ../Results
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
DEF_DATA = os.path.normpath(os.path.join(HERE, "..", "..", "Data", "classroom"))
DEF_OUT = os.path.normpath(os.path.join(HERE, "..", "..", "Results"))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data-dir", default=DEF_DATA)
ap.add_argument("--out", default=DEF_OUT)
ap.add_argument("--src", default=os.environ.get("EG_SRC", DEF_SRC))
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.src))
from parse_csi import load_esp32_csv                 # noqa: E402
from preprocess import preprocess_session            # noqa: E402
from features import featurize_session, FEATURE_NAMES  # noqa: E402

RANGE_RE = re.compile(r"_d(\d+)m", re.IGNORECASE)
FAIL_RE = re.compile(r"_(still|door)_", re.IGNORECASE)


def windows(session, n_features):
    amp, ph = preprocess_session(session["amplitude"], session["phase"], session["fs"])
    f, _ = featurize_session(amp, ph, session["rssi"], session["fs"], 1.0, 0.5,
                             pdr=session.get("pdr", 1.0))
    # Same per-session RSSI calibration as build_windows (rssi_mean relative
    # to the recording's median link level); keeps every metrics file
    # consistent with the trained pipeline.
    f = f.copy()
    f[:, FEATURE_NAMES.index("rssi_mean")] -= np.median(
        f[:, FEATURE_NAMES.index("rssi_mean")])
    return f[:, :n_features]


def fit_rf(seed):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def temporal_holdout(feats_by_class, seed, train_frac=0.65, guard=3):
    """Train on first train_frac of every capture, test on the rest."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for c, stacks in feats_by_class.items():
        for f in stacks:
            cut = int(train_frac * len(f))
            Xtr.append(f[:cut]); ytr += [c] * cut
            if len(f) > cut + guard:
                Xte.append(f[cut + guard:]); yte += [c] * (len(f) - cut - guard)
    if not Xte:
        return None
    Xtr = np.vstack(Xtr); Xte = np.vstack(Xte)
    ytr = np.array(ytr); yte = np.array(yte)
    model = fit_rf(seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "protocol": "temporal_holdout_65_35_guard3",
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "train_windows": int(Xtr.shape[0]), "test_windows": int(Xte.shape[0]),
        "model": model,
    }


def loso(feats_by_class, seed):
    """Leave-one-recording-out over per-capture feature stacks."""
    trials = [(c, f) for c, stacks in feats_by_class.items() for f in stacks]
    if len(trials) < 4:
        return None
    per_class_counts = {c: len(s) for c, s in feats_by_class.items()}
    if min(per_class_counts.values()) < 2:
        return None
    truth, pred = [], []
    for c_hold, f_hold in trials:
        Xtr = np.vstack([f for c, f in trials if not (c == c_hold and f is f_hold)])
        ytr = np.concatenate([[c] * len(f) for c, f in trials
                              if not (c == c_hold and f is f_hold)])
        model = fit_rf(seed).fit(Xtr, ytr)
        p = model.predict(f_hold)
        truth += [c_hold] * len(f_hold); pred += p.tolist()
    truth = np.array(truth); pred = np.array(pred)
    return {
        "protocol": "leave_one_recording_out",
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "confusion_matrix": confusion_matrix(truth, pred).tolist(),
        "recordings_per_class": per_class_counts,
        "test_windows": int(truth.size),
        "model": fit_rf(seed).fit(
            np.vstack([f for _, fs in feats_by_class.items() for f in fs]),
            np.concatenate([[c] * len(f) for c, fs in feats_by_class.items()
                            for f in fs])),
    }


def main():
    os.makedirs(args.out, exist_ok=True)
    sessions = []
    for p in sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))):
        try:
            s = load_esp32_csv(p)
        except Exception as exc:
            print(f"SKIP {os.path.basename(p)}: {exc}"); continue
        if s.get("label") is None:
            print(f"SKIP {os.path.basename(p)}: no label"); continue
        s["_name"] = os.path.basename(p)
        sessions.append(s)
    if not sessions:
        raise SystemExit("no usable captures")

    # feature stacks, once
    for s in sessions:
        s["_f16"] = windows(s, len(FEATURE_NAMES))

    main_sessions = [s for s in sessions
                     if not RANGE_RE.search(s["_name"]) and not FAIL_RE.search(s["_name"])]
    range_sessions = [s for s in sessions if RANGE_RE.search(s["_name"])]
    fail_sessions = [s for s in sessions if FAIL_RE.search(s["_name"])]

    # ---- counting + ablation on the main set --------------------------------
    def stacks(n):
        d = {}
        for s in main_sessions:
            d.setdefault(s["label"], []).append(s["_f16"][:, :n])
        return d

    results = {}
    for n_feats, tag in ((12, "baseline12"), (16, "extended16")):
        r = loso(stacks(n_feats), args.seed) or temporal_holdout(stacks(n_feats), args.seed)
        if r is None:
            print(f"!! no evaluation possible for {tag}")
            continue
        r.pop("model", None)
        results[tag] = r
        print(f"[{tag}] {r['protocol']}: acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f}")
    if "extended16" in results and results["extended16"].get("model"):
        pass
    with open(os.path.join(args.out, "m4_ablation.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # the counting model used by range/reliability/failure sections
    best = loso(stacks(16), args.seed) or temporal_holdout(stacks(16), args.seed)
    counting = {k: v for k, v in best.items() if k != "model"}
    with open(os.path.join(args.out, "m4_counting.json"), "w") as fh:
        json.dump(counting, fh, indent=2)
    model = best["model"]

    # ---- range test ----------------------------------------------------------
    if range_sessions:
        rng = []
        for s in range_sessions:
            d = int(RANGE_RE.search(s["_name"]).group(1))
            Xb = np.vstack([s["_f16"] for s in main_sessions])
            yb = np.concatenate([np.full(len(s["_f16"]), s["label"])
                                 for s in main_sessions])
            bmodel = fit_rf(args.seed).fit(Xb, (yb > 0).astype(int))
            pred_bin = bmodel.predict((s["_f16"] > 0).astype(int) * 0 + s["_f16"])
            pred_bin = bmodel.predict(s["_f16"])
            acc = float(np.mean(pred_bin == (s["label"] > 0)))
            rng.append({"distance_m": d, "file": s["_name"], "binary_accuracy": acc,
                        "rssi_mean": float(np.mean(s["rssi"])),
                        "rate_hz": float(s["fs"]), "frames": int(len(s["_f16"]))})
            print(f"[range {d}m] binary acc={acc:.3f} rssi={rng[-1]['rssi_mean']:.1f}")
        with open(os.path.join(args.out, "m4_range.json"), "w") as fh:
            json.dump(sorted(rng, key=lambda x: x["distance_m"]), fh, indent=2)

    # ---- reliability ----------------------------------------------------------
    by_class = {}
    for s in main_sessions:
        by_class.setdefault(s["label"], []).append(s)
    if max(len(v) for v in by_class.values()) >= 2:
        rel = []
        for c, group in sorted(by_class.items()):
            for s in group:
                train = [x for x in group if x is not s]
                other = [x for cc, g in by_class.items() if cc != c for x in g]
                Xtr = np.vstack([x["_f16"] for x in train + other])
                ytr = np.concatenate([[x["label"]] * len(x["_f16"])
                                      for x in train + other])
                m = fit_rf(args.seed).fit(Xtr, ytr)
                p = m.predict(s["_f16"])
                rel.append({"file": s["_name"], "label": c,
                            "accuracy": float(np.mean(p == c)),
                            "windows": int(len(p))})
        with open(os.path.join(args.out, "m4_reliability.json"), "w") as fh:
            json.dump(rel, fh, indent=2)
        accs = [r["accuracy"] for r in rel]
        print(f"[reliability] per-trial acc: mean={np.mean(accs):.3f} "
              f"min={min(accs):.3f} max={max(accs):.3f}")

    # ---- failure analysis ------------------------------------------------------
    if fail_sessions:
        fails = []
        for s in fail_sessions:
            p = model.predict(s["_f16"])
            dist = {int(k): int(v) for k, v in
                    zip(*np.unique(p, return_counts=True))}
            fails.append({"file": s["_name"], "true_label": s["label"],
                          "prediction_distribution": dist,
                          "windows": int(len(p))})
            print(f"[failure] {s['_name']}: true={s['label']} -> {dist}")
        with open(os.path.join(args.out, "m4_error_analysis.json"), "w") as fh:
            json.dump(fails, fh, indent=2)

    # ---- markdown summary -------------------------------------------------------
    lines = ["# ElderGuard M4 metrics", "",
             f"Data: `{os.path.abspath(args.data_dir)}`  |  captures: {len(sessions)}"
             f" (main {len(main_sessions)}, range {len(range_sessions)}, failure {len(fail_sessions)})", ""]
    if "baseline12" in results and "extended16" in results:
        lines += ["## Ablation (same protocol both)", "",
                  "| Feature set | Protocol | Accuracy | Macro-F1 |", "|---|---|---:|---:|",
                  f"| 12 (M3 baseline) | {results['baseline12']['protocol']} | {results['baseline12']['accuracy']:.3f} | {results['baseline12']['macro_f1']:.3f} |",
                  f"| 16 (+ counting features) | {results['extended16']['protocol']} | {results['extended16']['accuracy']:.3f} | {results['extended16']['macro_f1']:.3f} |", ""]
    with open(os.path.join(args.out, "M4_METRICS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote metrics to {args.out}")


if __name__ == "__main__":
    main()
