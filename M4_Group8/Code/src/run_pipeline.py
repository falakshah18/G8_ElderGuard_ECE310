

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_session
from features import featurize_session, FEATURE_NAMES
from evaluate import leave_one_session_out
import synthetic as synth


def load_sessions(args):
    if args.synthetic:
        print("Loading synthetic sessions (pipeline self-test, not reported results)")
        return synth.generate_dataset()
    from parse_csi import load_radiopoints
    sessions = load_radiopoints(args.data_dir)
    if not sessions:
        raise SystemExit(f"No sessions found under {args.data_dir}")
    print(f"Loaded {len(sessions)} sessions from {args.data_dir}")
    return sessions


def build_windows(sessions, window_s=1.0, overlap=0.5, max_people=2,
                  center_rssi=True):
    X, y, sid = [], [], []
    rssi_col = FEATURE_NAMES.index("rssi_mean")
    for s in sessions:
        label = s.get("label")
        if label is None or label > max_people:
            continue
        amp, phase = preprocess_session(
            s["amplitude"], s["phase"], s["fs"], do_phase=True
        )
        feats, _ = featurize_session(
            amp, phase, s.get("rssi", 0.0), s["fs"], window_s, overlap,
            pdr=s.get("pdr", 1.0),
        )
        if feats.size == 0:
            continue
        if center_rssi:
            feats = feats.copy()
            feats[:, rssi_col] -= np.median(feats[:, rssi_col])
        X.append(feats)
        y.extend([int(label)] * feats.shape[0])
        sid.extend([s["session_id"]] * feats.shape[0])
    if not X:
        raise SystemExit("No analysis windows were produced.")
    return np.vstack(X), np.array(y, dtype=int), np.array(sid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="../../Data/radiopoints",
                    help="Root of the RadioPoints dataset")
    ap.add_argument("--synthetic", action="store_true",
                    help="Run on generated sessions instead of real data")
    ap.add_argument("--out", default="../../Results",
                    help="Output directory for metrics.json and figures")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    sessions = load_sessions(args)
    X, y, sid = build_windows(sessions, args.window, args.overlap)
    print(f"Feature matrix: {X.shape}, classes={np.bincount(y).tolist()}, "
          f"sessions={len(np.unique(sid))}")

    results = leave_one_session_out(X, y, sid)
    results["_feature_names"] = FEATURE_NAMES
    results["_dataset"] = "synthetic" if args.synthetic else "RadioPoints WiFi-CrowdCounting"
    results["_n_windows"] = int(X.shape[0])
    results["_n_sessions"] = int(len(np.unique(sid)))

    out_path = os.path.join(args.out, "metrics.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {out_path}")


    np.savez(os.path.join(args.out, "features.npz"), X=X, y=y, sid=sid)

    try:
        from make_figures import make_all_figures
        make_all_figures(out_path, args.out)
        print(f"Figures written to {args.out}")
    except Exception as exc:
        print(f"! figure generation skipped: {exc}")


if __name__ == "__main__":
    main()
