#!/usr/bin/env python3
"""
ElderGuard - generate FAKE captures in the exact CSV format that
rx_esp32_csi.ino + collect_serial.py produce, so you can validate the whole
Python chain (parse -> preprocess -> features -> models -> CV) BEFORE the
hardware day. Nothing here touches the boards.

    python make_fake_captures.py --out fake_captures
    python run_own.py --data-dir fake_captures --out Results_fake --save-model model_rf.joblib

Fake data is 64 subcarriers/frame at 100 Hz; motion amplitude/noise grows
with the number of people, which is exactly the signal the amp-variance
feature uses. Expected: accuracy well above the 0.333 chance level.
"""

import argparse
import os

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="fake_captures")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--trials", type=int, default=2)
args = ap.parse_args()

rng = np.random.default_rng(11)
FS, NSUB = 100.0, 64
HEADER = ("type,channel,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,"
          "not_sounding,aggregation,stbc,fec_coding,sgi,ampdu_cnt,ant,"
          "timestamp,data")
MAC = "aa:bb:cc:dd:ee:01"


def one_capture(n_people, n_frames):
    t = np.arange(n_frames) / FS
    base = rng.uniform(8.0, 40.0, NSUB)
    amp = np.tile(base, (n_frames, 1))
    for _ in range(n_people):
        movers = rng.permutation(NSUB)[: NSUB // 2]
        f = rng.uniform(0.5, 3.0)
        amp[:, movers] += (4.0 + 3.0 * n_people) * np.abs(
            np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        )[:, None]
    amp += rng.normal(0.0, 1.0 + 0.8 * n_people, amp.shape)
    amp = np.clip(amp, 0.0, 120.0)
    phase = rng.normal(0.0, 0.2, amp.shape)
    rssi = int(-40 - 5 * n_people + rng.normal(0, 2))
    return amp, phase, rssi


def write_csv(path, amp, phase, rssi):
    n, m = amp.shape
    with open(path, "w", newline="") as fh:
        fh.write(HEADER + "\n")
        for k in range(n):
            i8 = np.empty(2 * m, dtype=int)
            i8[0::2] = np.clip(np.round(amp[k] * np.sin(phase[k])), -127, 127)
            i8[1::2] = np.clip(np.round(amp[k] * np.cos(phase[k])), -127, 127)
            # Match the real receiver's microsecond timestamp field; otherwise
            # the parser correctly estimates an impossible 1 MHz rate.
            fh.write(f"CSI,1,{MAC},{rssi},0,1,7,0,1,1,0,0,0,0,0,0,{int(k / FS * 1e6) % 2**32},"
                     f"[{' '.join(str(v) for v in i8)}]\n")


os.makedirs(args.out, exist_ok=True)
total = 0
for room in ("room1",):
    for label in (0, 1, 2):
        for trial in range(1, 3 * args.trials + 1):
            n_frames = int(args.seconds * FS)
            amp, phase, rssi = one_capture(label, n_frames)
            name = f"{room}_{label}p_trial{trial}_fs100.csv"
            write_csv(os.path.join(args.out, name), amp, phase, rssi)
            total += 1
print(f"Wrote {total} fake captures to {args.out}/")
