#!/usr/bin/env python3
"""
ElderGuard - train + evaluate occupancy counting on YOUR OWN ESP32 captures.

The repo's run_pipeline.py only knows the RadioPoints dataset or synthetic
sessions. This script runs the exact same chain (preprocess -> features ->
threshold / Random Forest / MLP -> leave-one-session-out CV) on CSV files
produced by collect_serial.py from your receiver board.

Filename convention (encodes the ground truth):
    room<Name>_<N>p[_anything].csv     with <N> in {0, 1, 2}
    e.g.  room1_0p_t01_fs60.csv, room1_2p_t05_fs60.csv
    Optional sampling rate: add _fs100 anywhere in the name (default 100).

Usage (from build-help/tools, or adjust --src):
    python run_own.py --data-dir PATH_TO_CSVS --out PATH_TO_RESULTS
Also saves the best cross-validated model to --save-model for the live demo.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

DEF_SRC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data-dir", required=True, help="folder of labelled capture CSVs")
ap.add_argument("--out", default="Results_own")
ap.add_argument("--src", default=os.environ.get("EG_SRC", DEF_SRC),
                help="path to the repo's M2 Code/src folder")
ap.add_argument("--window", type=float, default=1.0)
ap.add_argument("--overlap", type=float, default=0.5)
ap.add_argument("--max-people", type=int, default=3,
                help="highest labelled head-count kept (3 enables 3p files)")
ap.add_argument("--task", choices=["count", "occupy"], default="count",
                help="occupy = collapse to empty vs occupied (easier, and it "
                     "is what a fall detector really needs first)")
ap.add_argument("--save-model", default=None, help="e.g. model_rf.joblib")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.src))
from parse_csi import load_esp32_csv            # noqa: E402
from run_pipeline import build_windows          # noqa: E402
from evaluate import leave_one_session_out      # noqa: E402
from features import FEATURE_NAMES              # noqa: E402
from models import build_models                 # noqa: E402


def main():
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSVs found in {args.data_dir}")

    sessions = []
    for p in paths:
        try:
            s = load_esp32_csv(p)
        except ValueError as exc:
            print(f"! skipping {os.path.basename(p)}: {exc}")
            continue
        if s.get("label") is None or s["label"] > 3:
            print(f"! skipping {os.path.basename(p)}: filename needs 0p/1p/2p/3p")
            continue
        print(f"  + {os.path.basename(p)}: {s['amplitude'].shape[0]} frames, "
              f"{s['amplitude'].shape[1]} subcarriers, label={s['label']}")
        sessions.append(s)

    if len(sessions) < 2:
        raise SystemExit("Need at least 2 usable labelled sessions.")

    X, y, sid = build_windows(sessions, args.window, args.overlap,
                              max_people=args.max_people)
    if args.task == "occupy":
        # The 2-class question ("is anyone here?") is measurably easier than
        # the head count, so it gets its own model and its own demo mode.
        y = (y > 0).astype(int)
    n_cls = len(np.unique(y))
    counts = np.bincount(y, minlength=n_cls)
    names = (["empty", "occupied"] if args.task == "occupy"
             else [f"{i}p" for i in range(n_cls)])
    print(f"\nFeature matrix: {X.shape}, windows per "
          f"{'/'.join(names)} = {counts.tolist()}, "
          f"sessions={len(np.unique(sid))}")

    results = leave_one_session_out(X, y, sid)
    results["_feature_names"] = FEATURE_NAMES
    results["_dataset"] = f"own-hardware ESP32 captures ({len(sessions)} sessions)"
    results["_n_windows"] = int(X.shape[0])
    results["_n_sessions"] = int(len(np.unique(sid)))

    os.makedirs(args.out, exist_ok=True)
    stem = "metrics_own" if args.task == "count" else "metrics_occupy"
    metrics_path = os.path.join(args.out, f"{stem}.json")
    with open(metrics_path, "w") as fh:
        json.dump(results, fh, indent=2)

    best = max((m for m in results if not m.startswith("_")),
               key=lambda m: results[m]["accuracy"])
    acc = results[best]["accuracy"]
    print(f"\nBest model: {best}  accuracy={acc:.3f}  "
          f"(chance = {1.0 / n_cls:.3f})")
    print("VERDICT:", "BEATS CHANCE - good for the MVP" if acc > 1.0 / n_cls
          else "AT/BELOW CHANCE - collect more varied trials (see guide)")

    try:
        from make_figures import make_all_figures
        fig_dir = args.out if args.task == "count" else os.path.join(args.out, "occupy")
        os.makedirs(fig_dir, exist_ok=True)
        make_all_figures(metrics_path, fig_dir)
        print(f"Figures written to {fig_dir}")
    except Exception as exc:
        print(f"! figure generation skipped: {exc}")

    if args.save_model:
        import joblib
        # Save the model that actually won the CV, not always the forest: the
        # live demo is only as good as the artifact it loads.
        best_model = build_models()[best]
        best_model.fit(X, y)
        joblib.dump(best_model, args.save_model)
        print(f"Saved trained model ({best}) -> {args.save_model}")


if __name__ == "__main__":
    main()
