#!/usr/bin/env python3
"""ElderGuard two-layer PRESENCE detector (M4) - the faculty-cabin product mode.

Layer 1 - motion (fast):   amplitude variance vs calibrated empty-room baseline
                           over ~2 s windows. Catches walking / moving people.
Layer 2 - breathing (slow): 30 s window of mean CSI amplitude, detrended,
                           FFT in the 0.15-0.6 Hz band. A standing spectral
                           peak = a living, still human (DeMan, IEEE JSAC 2015).
Presence = motion OR breathing. VACANT only when both layers are silent
for --vacant-need consecutive updates.

References:
  - Wang et al., "Non-invasive Detection of Moving and Stationary Human"
    (DeMan), IEEE JSAC 33(9), 2015: motion-only baselines misclassify a
    large share of stationary humans as empty; a breathing layer restores
    ~94-96% detection.
  - Espressif esp-csi: vendor reference implementation of Wi-Fi presence
    detection with on-site calibration on ESP32 hardware.

Modes:
  --replay CSV : run over a recorded capture (validation, no hardware)
  --port auto  : live from the receiver board

Example:
  python tools/presence_demo.py --replay ../Data/classroom/roomA_1p_still_t01.csv
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--port", default=None, help="receiver port, or 'auto'")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--replay", metavar="CSV", default=None)
ap.add_argument("--replay-fs", type=float, default=0.0)
ap.add_argument("--src", default=os.environ.get("EG_SRC", DEF_SRC))
ap.add_argument("--motion-calib", type=float, default=10.0,
                help="seconds of empty-room baseline for the motion layer")
ap.add_argument("--motion-mult", type=float, default=4.0,
                help="motion threshold multiplier over baseline variance")
ap.add_argument("--breath-win", type=float, default=30.0,
                help="breathing window seconds (>=20 recommended)")
ap.add_argument("--breath-lo", type=float, default=0.15, help="breathing band low Hz")
ap.add_argument("--breath-hi", type=float, default=0.6, help="breathing band high Hz")
ap.add_argument("--breath-snr", type=float, default=3.0,
                help="min spectral peak SNR (peak vs band median) to accept breathing")
ap.add_argument("--vacant-need", type=int, default=3,
                help="consecutive silent updates before VACANT")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.src))
from parse_csi import load_esp32_csv, parse_esp32_csi_data, _select_ht20_subcarriers  # noqa: E402
from preprocess import preprocess_session  # noqa: E402


def breath_snr(mean_amp, fs):
    """Differential respiration SNR.

    Peak power in the breathing band (0.15-0.6 Hz) divided by the peak
    power in a CONTROL band (1.5-4 Hz). Room-wide disturbances (fan, AC,
    link drift, motion) raise both bands together and cancel; a genuine
    still-human breathing peak stands out only in the breathing band.
    Uses RAW (Hampel-only) amplitude - a 0.5-10 Hz bandpass would erase
    exactly the frequencies breathing lives in.
    """
    need = int(args.breath_win * fs)
    if len(mean_amp) < max(need // 2, int(15 * fs)):
        return 0.0, 0.0
    x = np.asarray(mean_amp, dtype=float)
    x = x - np.polyval(np.polyfit(np.arange(x.size), x, 2), np.arange(x.size))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    b = (freqs >= args.breath_lo) & (freqs <= args.breath_hi)
    c = (freqs >= 1.5) & (freqs <= 4.0)
    if not b.any() or not c.any():
        return 0.0, 0.0
    pb, pc = spec[b], spec[c]
    control = float(np.percentile(pc, 90)) + 1e-12
    i = int(np.argmax(pb))
    return float(pb[i] / control), float(freqs[b][i])


def analyse_frames(amp_stack, baseline_var):
    """Return (motion, breath_snr_value, breath_hz, fs_est)."""
    fs = args.replay_fs if (args.replay and args.replay_fs) else 62.0
    a2, _ = preprocess_session(amp_stack, np.zeros_like(amp_stack),
                               fs, do_phase=False)
    var = float(np.mean(np.var(a2, axis=0)))
    motion = var > args.motion_mult * baseline_var
    snr, hz = breath_snr(a2.mean(axis=1), fs)
    return motion, snr, hz, fs, var


def replay(path):
    s = load_esp32_csv(path)
    fs = s["fs"]
    n = s["amplitude"].shape[0]
    print(f"REPLAY {os.path.basename(path)}  label={s['label']}  "
          f"{n} frames @ {fs:.1f} Hz  ({n/fs:.0f} s)")

    # motion baseline: first motion-calib seconds
    cal_n = int(args.motion_calib * fs)
    base_var = float(np.mean(np.var(s["amplitude"][:cal_n], axis=0)))
    print(f"motion baseline variance = {base_var:.3f} (from first {args.motion_calib:.0f} s)")

    w = int(args.breath_win * fs)
    step = int(2.0 * fs)
    vacant_run = 0
    print(f"{'time':>6} {'motion':>7} {'breathSNR':>10} {'bpm':>6}  STATE")
    for start in range(0, n - w, step):
        chunk = s["amplitude"][start:start + w]
        # Hampel only - NO 0.5-10 Hz bandpass (it would erase the breathing band)
        chunk = chunk.copy()
        med = np.median(chunk, axis=0)
        mad = np.median(np.abs(chunk - med), axis=0)
        bad = np.abs(chunk - med) > 3.0 * 1.4826 * mad
        chunk = np.where(bad, med[None, :], chunk)
        var = float(np.mean(np.var(chunk, axis=0)))
        motion = var > args.motion_mult * base_var
        bs, hz = breath_snr(chunk.mean(axis=1), fs)
        occupied = motion or (bs >= args.breath_snr)
        vacant_run = 0 if occupied else vacant_run + 1
        state = "OCCUPIED" if occupied else ("VACANT" if vacant_run >= args.vacant_need
                                             else "vacant(pending)")
        print(f"{start/fs:6.0f} {'YES' if motion else 'no':>7} {bs:10.2f} "
              f"{hz*60:6.1f}  {state}")
    print("\nNOTE: ground truth for this file =",
          "OCCUPIED" if s["label"] > 0 else "VACANT")


if args.replay:
    replay(args.replay)
else:
    raise SystemExit("live mode: TODO after tomorrow's validation; "
                     "use --replay for now")
