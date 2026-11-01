#!/usr/bin/env python3
"""Camera-free CSI preprocessing for one ElderGuard receiver CSV.

Decodes the project's ESP32-S3 serial capture format, applies its existing
Hampel/phase/band-pass pipeline, extracts per-window CSI features, and writes
those features to a CSV. No image, video, webcam, or camera dependency.
For model training and live prediction, use run_own.py/live_demo.py instead.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from parse_csi import load_esp32_csv  # noqa: E402
from preprocess import preprocess_session  # noqa: E402
from features import featurize_session, FEATURE_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", help="ESP32-S3 CSI CSV captured by collect_serial.py")
    ap.add_argument("--out", required=True, help="output feature CSV")
    ap.add_argument("--window", type=float, default=1.0, help="window size in seconds")
    ap.add_argument("--overlap", type=float, default=0.5, help="window overlap (0..1)")
    args = ap.parse_args()

    session = load_esp32_csv(args.capture)
    amp, phase = preprocess_session(session["amplitude"], session["phase"], session["fs"])
    matrix, starts = featurize_session(
        amp, phase, session["rssi"], session["fs"], args.window, args.overlap,
        pdr=session.get("pdr", 1.0),
    )
    if not len(matrix):
        raise SystemExit("No feature windows produced; capture may be too short or malformed.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session_id", "label", "start_frame", *FEATURE_NAMES])
        for start, row in zip(starts, matrix):
            writer.writerow([session["session_id"], session.get("label"), int(start), *row.tolist()])
    print(f"CSI-only preprocessing: {len(matrix)} windows, "
          f"{session['amplitude'].shape[1]} subcarriers, fs={session['fs']:g} Hz")
    print(f"Wrote features: {args.out}")


if __name__ == "__main__":
    main()
