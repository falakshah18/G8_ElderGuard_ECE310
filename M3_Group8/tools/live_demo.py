#!/usr/bin/env python3
"""
ElderGuard live demo, straight from the receiver board's serial port.

  --mode count : real-time 0/1/2 occupancy using the model trained by run_own.py
  --mode fall  : rule-based fall detector (motion burst, then prolonged still)

    python live_demo.py --port COM7 --mode count --model model_rf.joblib
    python live_demo.py --port COM7 --mode fall  --calib 45

Rules of thumb:
  * Keep the Arduino Serial Monitor CLOSED while this runs - a COM port can be
    opened by only one program at a time.
  * For fall mode, the first --calib seconds MUST be a still/empty room; the
    script then learns the room's baseline noise.
  * TX board must be flooding (its sketch running) or no CSI arrives.
"""

import argparse
import os
import sys
import time
from collections import Counter, deque

import numpy as np

DEF_SRC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--port", default="COM15", help="Laptop 1 receiver COM port (default COM15)")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--mode", choices=["count", "fall"], default="count")
ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "model_count.joblib"))
ap.add_argument("--src", default=os.environ.get("EG_SRC", DEF_SRC))
ap.add_argument("--chunk", type=float, default=2.0, help="update period (s)")
ap.add_argument("--calib", type=float, default=45.0,
                help="fall mode: still-empty-room baseline seconds")
ap.add_argument("--binary", action="store_true",
                help="count mode: answer OCCUPIED / EMPTY instead of a head "
                     "count. Measured on the same captures the 2-class "
                     "question is far easier than the 3-class one.")
ap.add_argument("--replay", metavar="CSV", default=None,
                help="no hardware: feed a recorded capture through the exact "
                     "same live path. Use this to prove the demo logic (and "
                     "your model file) work before you touch the boards.")
ap.add_argument("--replay-fs", type=float, default=0.0,
                help="frames/s for --replay (default: from the _fsNN in the "
                     "file name, else 90)")
args = ap.parse_args()
if not args.replay and not args.port:
    ap.error("--port is required unless --replay is used")

sys.path.insert(0, os.path.abspath(args.src))
import serial                                          # noqa: E402
from eg_port import resolve_port                       # noqa: E402
from parse_csi import parse_esp32_csi_data, _select_ht20_subcarriers  # noqa: E402
from preprocess import preprocess_session             # noqa: E402
from features import featurize_session, FEATURE_NAMES  # noqa: E402

if args.port and not args.replay:
    args.port = resolve_port(args.port)

VARI_IDX = FEATURE_NAMES.index("amp_var_mean")

model = None
if args.mode == "count":
    import joblib
    if not os.path.exists(args.model):
        raise SystemExit(f"Train a model first: run_own.py --save-model {args.model}")
    model = joblib.load(args.model)


def beep():
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 400)
            time.sleep(0.15)
    except Exception:
        print("\a" * 5)


def read_row(line):
    """CSI row -> (amp, phase, rssi) or None."""
    if not line.startswith("CSI"):
        return None
    parts = line.split(",", 17)
    if len(parts) < 18:
        return None
    try:
        rssi = float(parts[3])
    except ValueError:
        rssi = 0.0
    try:
        amp, phase = parse_esp32_csi_data(parts[17])
    except ValueError:
        return None  # USB dropped a byte mid-row
    if amp.size == 0:
        return None
    if amp.size > 64 and amp.size % 64 == 0:
        # Firmware enables LLTF, HT-LTF, and STBC HT-LTF2; CSI vectors can
        # contain one, two, or three 64-complex-sample LTF sections. Keep the
        # first (LLTF) section consistently; do not pad/truncate arbitrarily.
        amp, phase = amp[:64], phase[:64]
    amp, phase = _select_ht20_subcarriers(amp, phase)
    if amp.size != 52:
        return None
    return amp, phase, rssi


def chunk_score(frames, phases, rssis, elapsed):
    """One update period of frames -> last 1 s feature window."""
    rate = len(frames) / max(elapsed, 1e-3)
    if rate < 25.0:
        # Below this the 0.5-10 Hz band filter is impossible (Nyquist);
        # almost always means the CSI stream stalled, not a slow room.
        print(f"... only {rate:.0f} frames/s - CSI stream too slow, waiting")
        return None
    fs = rate
    amp = np.vstack(frames)
    ph = np.vstack(phases)
    r = np.asarray(rssis, dtype=float)
    a2, p2 = preprocess_session(amp, ph, fs)
    feats, _ = featurize_session(a2, p2, r, fs, window_s=1.0, overlap=0.5)
    if feats.size == 0:
        return None
    return feats[-1], len(frames), rate, float(np.mean(r)) if r.size else float("nan")


def open_stream():
    """(Re)open the receiver port and wait until CSI rows are flowing."""
    while True:
        try:
            s = serial.Serial(args.port, args.baud, timeout=0.5)
        except serial.SerialException:
            print(f"waiting for {args.port} to reappear ...")
            time.sleep(2)
            continue
        # Opening the CDC port asserts DTR/RTS which can park the board in
        # the bootloader; release and pulse to reboot into the sketch.
        s.setDTR(False)
        s.setRTS(True)
        time.sleep(0.2)
        s.setRTS(False)
        print("Reset sent. Waiting for CSI header line ...")
        t_wait = time.time()
        got = False
        while time.time() - t_wait < 30:
            if s.readline().decode("utf-8", errors="ignore").startswith("type,"):
                got = True
                break
        if got:
            print("CSI streaming. Board B is now resetting; "
                  "keep clear of the boards.")
            return s
        print("no CSI header within 30 s - retrying ...")
        try:
            s.close()
        except Exception:
            pass
        time.sleep(2)


def new_fall():
    return dict(baseline=None, calib_scores=[], state="ARMED", t_spike=0.0,
                quiet_since=None, t_cool=0.0)


def process(vec, now, fall, recent, stream=None):
    """One update period. Shared byte-for-byte by live and replay paths."""
    v = float(vec[VARI_IDX])

    if args.mode == "count":
        pred = int(model.predict(vec[None, :])[0])
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(vec[None, :])[0]
            except (ValueError, AttributeError):
                pass
        recent.append(pred)
        shown = Counter(recent).most_common(1)[0][0]
        raw = pred
        if args.binary:
            shown = 0 if shown == 0 else 1
            raw = 0 if pred == 0 else 1
        label = ("EMPTY" if shown == 0 else "OCCUPIED") if args.binary \
            else str(shown)
        tag = "" if shown == raw else f"   (raw={raw}, smoothed)"
        stream = stream or {}
        score_text = ""
        if probabilities is not None:
            score_text = " scores=" + ",".join(
                f"{int(cls)}:{prob:.0%}" for cls, prob in zip(model.classes_, probabilities)
            )
        rssi = stream.get("rssi", float("nan"))
        rate = stream.get("rate", 0.0)
        n = stream.get("packets", 0)
        quality = "GOOD" if rate >= 45 else "FAIR" if rate >= 25 else "LOW"
        print(f"[{now % 100000:9.2f}] PEOPLE = {label}{tag}"
              f"{score_text} | CSI packets={n} rate={rate:.0f}/s"
              f" RSSI={rssi:.0f}dBm quality={quality} | receiver=CONNECTED")
        return

    # ---- fall mode ----
    baseline = fall["baseline"]
    if baseline is None:
        fall["calib_scores"].append(v)
        if len(fall["calib_scores"]) * args.chunk >= args.calib:
            fall["baseline"] = float(np.median(fall["calib_scores"]))
            print(f"== Baseline set to {fall['baseline']:.1f} "
                  f"(max={max(fall['calib_scores']):.1f}). "
                  f"Arm the room: simulate a FALL now ==")
            return
        print(f"calibrating empty room... score={v:.1f}")
        return

    state = fall["state"]
    print(f"score={v:9.1f} (baseline {baseline:.1f}) state={state}")
    if state == "ARMED":
        if v > max(6.0 * baseline, baseline + 40.0):
            fall["state"] = "SPIKE"
            fall["t_spike"], fall["quiet_since"] = now, None
            print(">> motion burst detected - watching for stillness...")
    elif state == "SPIKE":
        if v <= 1.5 * baseline:
            if fall["quiet_since"] is None:
                fall["quiet_since"] = now
            elif now - fall["quiet_since"] >= 3.0:
                print("\n" + "=" * 46)
                print("FALL ALARM !!!  (burst then 3 s of stillness)")
                print("=" * 46 + "\n")
                beep()
                fall["state"], fall["t_cool"] = "COOLDOWN", now
        else:
            fall["quiet_since"] = None
        if fall["state"] == "SPIKE" and now - fall["t_spike"] > 10.0:
            fall["state"] = "ARMED"
    elif state == "COOLDOWN" and now - fall["t_cool"] > 15.0:
        fall["state"] = "ARMED"


def run_live():
    print(f"Live {args.mode} demo on {args.port} @ {args.baud}. Ctrl-C to stop.")
    ser = open_stream()
    frames, phs, rss = [], [], []
    t0 = time.time()
    last_stale_notice = 0.0
    fall, recent = new_fall(), deque(maxlen=5)

    try:
        while True:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                # Board B's USB link dropped (cable/power glitch). The device
                # re-enumerates within seconds; reconnect instead of dying.
                print("!! USB link dropped - Board B reconnecting ...")
                try:
                    ser.close()
                except Exception:
                    pass
                time.sleep(3)
                ser = open_stream()
                frames, phs, rss = [], [], []
                t0 = time.time()
                fall, recent = new_fall(), deque(maxlen=5)
                continue
            row = read_row(line) if line else None
            if row:
                frames.append(row[0])
                phs.append(row[1])
                rss.append(row[2])

            elapsed = time.time() - t0
            if elapsed < args.chunk:
                continue

            if not frames:
                if elapsed >= 3.0 and time.time() - last_stale_notice >= 5.0:
                    print("WARNING: CSI stream stale; no complete inference window available.")
                    last_stale_notice = time.time()
                continue

            result = chunk_score(frames, phs, rss, elapsed)
            frames, phs, rss = [], [], []
            t0 = time.time()
            if result is None:
                continue
            vec, n, rate, mean_rssi = result
            process(vec, time.time(), fall, recent,
                    {"packets": n, "rate": rate, "rssi": mean_rssi})

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


def _replay_fs(path):
    if args.replay_fs > 0:
        return args.replay_fs
    # Same rule as parse_csi: strip ".csv" or the tag reads "fs60.csv".
    base = os.path.splitext(os.path.basename(path))[0]
    for token in base.replace("-", "_").split("_"):
        if token.startswith("fs") and token[2:].isdigit():
            return float(token[2:])
    return 90.0


def run_replay():
    fs = _replay_fs(args.replay)
    need = max(1, int(round(args.chunk * fs)))
    print(f"REPLAY {os.path.basename(args.replay)} - {fs:.0f} frames/s, "
          f"one update every {args.chunk}s. No hardware, no COM port.")
    frames, phs, rss = [], [], []
    fall, recent = new_fall(), deque(maxlen=5)
    clock = 0.0
    with open(args.replay, newline="", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            row = read_row(line.strip())
            if not row:
                continue
            frames.append(row[0])
            phs.append(row[1])
            rss.append(row[2])
            if len(frames) < need:
                continue
            result = chunk_score(frames, phs, rss, len(frames) / fs)
            frames, phs, rss = [], [], []
            clock += args.chunk
            if result is None:
                continue
            vec, n, rate, mean_rssi = result
            process(vec, clock, fall, recent,
                    {"packets": n, "rate": rate, "rssi": mean_rssi})
    print("Replay finished.")


if args.replay:
    run_replay()
else:
    run_live()
