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

STILL captures (filename contains "still"): the person is present but
motionless, so variance must be LOW (not high), and the QC gate instead checks
link absorption: a body on the Tx-Rx line lowers the mean RSSI versus the
empty-room captures. A still capture whose RSSI shows no absorption is
invisible to the model - re-record with the person on the link.

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

print(f"{'file':36s} {'lab':3s} {'med':>6s} {'p90':>6s} {'busy%':>6s} {'rssi':>6s}  verdict")
rows = []
for p in paths:
    name = os.path.basename(p)
    try:
        s = load_esp32_csv(p)
    except Exception as exc:
        print(f"{name:36s}  -      -      -      -      -  SKIP ({exc})")
        continue
    label = s.get("label")
    if label is None:
        print(f"{name:36s}  -      -      -      -      -  SKIP (name needs 0p/1p/2p/3p)")
        continue
    amp, ph = preprocess_session(s["amplitude"], s["phase"], s["fs"], do_phase=True)
    f, _ = featurize_session(amp, ph, s.get("rssi", 0.0), s["fs"], 1.0, 0.5,
                             pdr=s.get("pdr", 1.0))
    if f.shape[0] < 30:
        rows.append((name, label, None, None, None, None, "REJECT (too few windows)"))
        continue
    col = f[:, VARI]
    med = float(np.median(col))
    p90 = float(np.percentile(col, 90))
    busy = float(np.mean(col > 1.0) * 100.0)
    rssi_arr = np.asarray(s.get("rssi", 0.0), dtype=float)
    rssi_mean = float(np.mean(rssi_arr)) if rssi_arr.size else 0.0
    rows.append((name, label, med, p90, busy, rssi_mean, None))

# Empty-room RSSI baseline for the still-capture absorption check.
empty_rssi = [r[5] for r in rows if r[1] == 0 and r[5] is not None]
base_rssi = float(np.median(empty_rssi)) if empty_rssi else None

rejects = []
for name, label, med, p90, busy, rssi_mean, preset in rows:
    if preset is not None:
        verdict = preset
    elif label == 0:
        verdict = "OK" if med <= args.quiet_max else (
            "SUSPECT" if med <= 0.6 else "REJECT empty room was not quiet")
    elif "still" in name.lower():
        # Still person: variance must NOT be high, and RSSI must show the
        # person's absorption of the link versus the empty-room baseline.
        if p90 >= 3.0:
            verdict = "REJECT too much movement for a still capture"
        elif p90 >= 1.0:
            verdict = "SUSPECT some motion seen in a still capture"
        elif base_rssi is None:
            verdict = "OK (no 0p baseline - absorption not checked)"
        else:
            drop = base_rssi - rssi_mean
            if drop >= 2.0:
                verdict = "OK"
            elif drop >= 0.5:
                verdict = "SUSPECT weak link absorption - person may be off the link"
            else:
                verdict = "REJECT still person invisible - record with them ON the link"
    else:
        # someone was supposed to be moving: the path must show it
        verdict = "OK" if p90 >= args.busy_min else (
            "SUSPECT" if p90 >= 0.3 else "REJECT no motion seen for this label")
    if verdict.startswith("REJECT"):
        rejects.append(name)
    rssi_txt = f"{rssi_mean:6.1f}" if rssi_mean is not None else "     -"
    med_txt = f"{med:6.2f}" if med is not None else "     -"
    p90_txt = f"{p90:6.2f}" if p90 is not None else "     -"
    busy_txt = f"{busy:6.1f}" if busy is not None else "     -"
    print(f"{name:36s} {label:<3d} {med_txt} {p90_txt} {busy_txt} {rssi_txt}  {verdict}")

if base_rssi is not None:
    print(f"\nempty-room RSSI baseline: {base_rssi:.1f} dBm "
          f"(still captures must sit >= 2 dB below this)")

print()
if rejects:
    print(f"{len(rejects)} capture(s) to re-record: " + ", ".join(rejects))
    print("Fix the room or the choreography, then record those again. Do not")
    print("edit the file name to make a bad recording fit - that is how a")
    print("model ends up confident and wrong.")
else:
    print("All captures match their labels. Safe to train.")
