#!/usr/bin/env python3
"""ElderGuard capture quality control - run this BEFORE you train.

A mislabelled recording cannot be fixed by any model. This tool reads each
capture, measures how much motion the radio path actually saw, and compares it
with the head-count written in the file name.

Verdicts
  OK      motion level matches the label
  SUSPECT usable, but the room was quieter/livelier than the label suggests
  REJECT  the label and the signal disagree - record this one again

Rules are written against the amplitude-variance feature (the same number the
threshold baseline and the models use), measured on real captures:

  quiet empty room        median ~0.12, p90 ~0.23
  one person moving       median 0.3 - 2.3, p90 3.7 - 5.6
  something moving that should not have been: median > 0.5 in an empty room

Usage:
    python check_captures.py --data-dir PATH_TO_CSVS
"""

import argparse
import glob
import os
import sys

import numpy as np

DEF_SRC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--src", default=os.environ.get("EG_SRC", DEF_SRC))
ap.add_argument("--quiet-max", type=float, default=0.30,
                help="max median variance still believable for an empty room")
ap.add_argument("--busy-min", type=float, default=0.45,
                help="min p90 variance expected while people are moving")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.src))
from parse_csi import load_esp32_csv                  # noqa: E402
from preprocess import preprocess_session             # noqa: E402
from features import featurize_session, FEATURE_NAMES  # noqa: E402

VARI = FEATURE_NAMES.index("amp_var_mean")

paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
if not paths:
    raise SystemExit(f"No CSVs in {args.data_dir}")

print(f"{'file':36s} {'lab':3s} {'med':>6s} {'p90':>6s} {'busy%':>6s}  verdict")
rejects = []
for p in paths:
    name = os.path.basename(p)
    try:
        s = load_esp32_csv(p)
    except Exception as exc:
        print(f"{name:36s}  -      -      -      -  SKIP ({exc})")
        continue
    label = s.get("label")
    if label is None:
        print(f"{name:36s}  -      -      -      -  SKIP (name needs 0p/1p/2p/3p)")
        continue
    amp, ph = preprocess_session(s["amplitude"], s["phase"], s["fs"], do_phase=True)
    f, _ = featurize_session(amp, ph, s.get("rssi", 0.0), s["fs"], 1.0, 0.5,
                             pdr=s.get("pdr", 1.0))
    if f.shape[0] < 30:
        print(f"{name:36s} {label:<3d}    -      -      -  REJECT (too few windows)")
        rejects.append(name)
        continue
    col = f[:, VARI]
    med = float(np.median(col))
    p90 = float(np.percentile(col, 90))
    busy = float(np.mean(col > 1.0) * 100.0)

    if label == 0:
        verdict = "OK" if med <= args.quiet_max else (
            "SUSPECT" if med <= 0.6 else "REJECT empty room was not quiet")
    else:
        # someone was supposed to be moving: the path must show it
        verdict = "OK" if p90 >= args.busy_min else (
            "SUSPECT" if p90 >= 0.3 else "REJECT no motion seen for this label")
    if verdict.startswith("REJECT"):
        rejects.append(name)
    print(f"{name:36s} {label:<3d} {med:6.2f} {p90:6.2f} {busy:6.1f}  {verdict}")

print()
if rejects:
    print(f"{len(rejects)} capture(s) to re-record: " + ", ".join(rejects))
    print("Fix the room or the choreography, then record those again. Do not")
    print("edit the file name to make a bad recording fit - that is how a")
    print("model ends up confident and wrong.")
else:
    print("All captures match their labels. Safe to train.")
