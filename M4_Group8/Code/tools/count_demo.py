#!/usr/bin/env python3
r"""
ElderGuard video-demo display - "NO PERSON / ONE PERSON / TWO PEOPLE PRESENT".

Deployment-style layer over the two trained models:

  1. binary model (model_occupy.joblib, 2 s windows) answers "is anyone here?"
  2. if occupied, the count model (model_count.joblib) answers "1 or 2?"

Both models run on the SAME live feature window (2 s, per-session RSSI
calibration). The displayed state is the majority vote of the last --smooth
updates so the video shows a stable verdict instead of per-window flicker.
This is standard deployment hysteresis, not a modification of the models.

Usage (live hardware):
    python tools/count_demo.py --port COM8

No-hardware preview (replays a recorded capture through the exact live path):
    python tools/count_demo.py --replay ../Data/classroom/roomA_2p_t09.csv
"""

import argparse
import os
import sys
import time
from collections import Counter, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
DEF_RESULTS = os.path.normpath(os.path.join(HERE, "..", "..", "Results"))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--port", default="auto",
                help="COM port of the RX board (default: auto-detect the only ESP32)")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--model-binary", default=os.path.join(DEF_RESULTS, "model_occupy.joblib"))
ap.add_argument("--model-count", default=os.path.join(DEF_RESULTS, "model_count.joblib"))
ap.add_argument("--src", default=os.environ.get("EG_SRC", DEF_SRC))
ap.add_argument("--chunk", type=float, default=2.0, help="update period (s)")
ap.add_argument("--window", type=float, default=2.0,
                help="feature window (s); must match the trained models")
ap.add_argument("--calib-rssi", type=float, default=15.0,
                help="empty-room link calibration seconds")
ap.add_argument("--smooth", type=int, default=3,
                help="majority vote over the last N updates")
ap.add_argument("--replay", metavar="CSV", default=None,
                help="no hardware: feed a recorded capture")
ap.add_argument("--replay-fs", type=float, default=0.0,
                help="frames/s for --replay (default: from _fsNN name, else 90)")
ap.add_argument("--no-clear", action="store_true", help="do not clear the screen")
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.src))
import serial                                              # noqa: E402
from eg_port import resolve_port                           # noqa: E402
from parse_csi import parse_esp32_csi_data, _select_ht20_subcarriers  # noqa: E402
from preprocess import preprocess_session                  # noqa: E402
from features import featurize_session, FEATURE_NAMES      # noqa: E402
import joblib

if args.window > args.chunk:
    args.chunk = args.window


def load_model(path, what):
    candidates = [path,
                  os.path.join(DEF_RESULTS, os.path.basename(path)),
                  os.path.join(HERE, os.path.basename(path))]
    for cand in candidates:
        if os.path.exists(cand):
            path = cand
            break
    if not os.path.exists(path):
        raise SystemExit(f"{what} not found: {path}\n"
                         f"Train it first (see README / report §9).")
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:      # run_own artifact format
        return obj["model"]
    return obj                                        # bare estimator (train_model)


model_binary = load_model(args.model_binary, "binary model")
model_count = load_model(args.model_count, "count model")

RSSI_CAL = {"buf": deque(maxlen=240)}


def read_row(line):
    """CSI row -> (amp, phase, rssi) or None. Same parser as live_demo.py."""
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
        return None
    if amp.size == 0:
        return None
    if amp.size > 64 and amp.size % 64 == 0:
        amp, phase = amp[:64], phase[:64]
    amp, phase = _select_ht20_subcarriers(amp, phase)
    if amp.size != 52:
        return None
    return amp, phase, rssi


def chunk_score(frames, phases, rssis, elapsed):
    """One update period -> feature vector (last full window)."""
    rate = len(frames) / max(elapsed, 1e-3)
    if rate < 25.0:
        return None, rate
    fs = rate
    amp = np.vstack(frames)
    ph = np.vstack(phases)
    r = np.asarray(rssis, dtype=float)
    RSSI_CAL["buf"].append(float(np.mean(r)) if r.size else 0.0)
    covered = len(RSSI_CAL["buf"]) * args.chunk
    if covered < args.calib_rssi:
        return "CALIBRATING", covered
    r = r - float(np.median(RSSI_CAL["buf"]))
    a2, p2 = preprocess_session(amp, ph, fs)
    feats, _ = featurize_session(a2, p2, r, fs, window_s=args.window, overlap=0.5)
    if feats.size == 0:
        return None, rate
    return feats[-1], rate


def predict_state(vec):
    """(state_key, occupied_conf, count_conf) from the two models."""
    occ = int(model_binary.predict(vec[None, :])[0])
    occ_conf = 0.0
    if hasattr(model_binary, "predict_proba"):
        try:
            occ_conf = float(model_binary.predict_proba(vec[None, :])[0].max())
        except Exception:
            pass
    if occ == 0:
        return "empty", occ_conf, 0.0
    cnt = int(model_count.predict(vec[None, :])[0])
    cnt_conf = 0.0
    if hasattr(model_count, "predict_proba"):
        try:
            cnt_conf = float(model_count.predict_proba(vec[None, :])[0].max())
        except Exception:
            pass
    if cnt == 0:
        return "uncertain", occ_conf, cnt_conf
    return ("one" if cnt == 1 else "two"), occ_conf, cnt_conf


LABELS = {
    "empty": "NO PERSON PRESENT",
    "one": "ONE PERSON PRESENT",
    "two": "TWO PEOPLE PRESENT",
    "uncertain": "PRESENCE DETECTED (count uncertain)",
    "calibrating": "CALIBRATING EMPTY ROOM - PLEASE STAND BY",
    "starting": "CONNECTING TO RECEIVER...",
}


def render(state, rate, occ_conf, cnt_conf, log):
    bar = "=" * 52
    conf_txt = ""
    if state in ("empty", "one", "two", "uncertain") and occ_conf:
        conf_txt = f"   confidence: presence {occ_conf:.0%}"
        if cnt_conf:
            conf_txt += f" | count {cnt_conf:.0%}"
    if state == "calibrating":
        rate_txt = f"   calibrating link baseline: {rate:.0f}/{args.calib_rssi:.0f}s"
    else:
        rate_txt = f"   CSI rate: {rate:.0f} frames/s   |   {time.strftime('%H:%M:%S')}"
    lines = [bar, "", f"   STATUS:  {LABELS[state]}", conf_txt, "",
             rate_txt,
             "", bar, "", "   recent:"]
    lines += [f"     {t}  {LABELS[s]}" for t, s in log[-6:]]
    return "\n".join(lines)


def handle(vec, rate, history, log, wallclock):
    occ_conf, cnt_conf = 0.0, 0.0
    if isinstance(vec, str) and vec == "CALIBRATING":
        state = "calibrating"
    elif vec is None or isinstance(vec, str):
        state = "starting"
    else:
        raw_state, occ_conf, cnt_conf = predict_state(vec)
        history.append(raw_state)
        state = Counter(history).most_common(1)[0][0]
    if state != (log[-1][1] if log else None):
        log.append((time.strftime("%H:%M:%S"), state))
        del log[:-8]
    if not args.no_clear:
        os.system("cls")
    print(render(state, rate, occ_conf if state != "calibrating" else 0.0,
                 cnt_conf if state not in ("empty", "calibrating") else 0.0, log))
    return state


def open_stream():
    while True:
        try:
            s = serial.Serial(args.port, args.baud, timeout=0.5)
        except serial.SerialException:
            print(f"waiting for {args.port} to reappear ...")
            time.sleep(2)
            continue
        s.setDTR(False)
        s.setRTS(True)
        time.sleep(0.2)
        s.setRTS(False)
        print("Reset sent. Waiting for CSI header ...")
        t_wait = time.time()
        got = False
        while time.time() - t_wait < 30:
            if s.readline().decode("utf-8", errors="ignore").startswith("type,"):
                got = True
                break
        if got:
            print("CSI streaming.")
            return s
        print("no CSI header within 30 s - retrying ...")
        try:
            s.close()
        except Exception:
            pass
        time.sleep(2)


def run_live():
    print(f"ElderGuard live demo on {args.port} @ {args.baud}. Ctrl-C to stop.")
    s = open_stream()
    frames, phs, rss = [], [], []
    t0 = time.time()
    history = deque(maxlen=max(args.smooth, 1))
    log = []
    try:
        while True:
            try:
                line = s.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                print("!! USB link dropped - reconnecting ...")
                try:
                    s.close()
                except Exception:
                    pass
                time.sleep(3)
                s = open_stream()
                frames, phs, rss = [], [], []
                t0 = time.time()
                continue
            row = read_row(line) if line else None
            if row:
                frames.append(row[0])
                phs.append(row[1])
                rss.append(row[2])
            if time.time() - t0 < args.chunk:
                continue
            vec, rate = (None, 0.0)
            if frames:
                vec, rate = chunk_score(frames, phs, rss, time.time() - t0)
            frames, phs, rss = [], [], []
            t0 = time.time()
            handle(vec, rate, history, log, time.time())
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            s.close()
        except Exception:
            pass


def run_replay():
    fs = args.replay_fs
    if fs <= 0:
        base = os.path.splitext(os.path.basename(args.replay))[0]
        fs = 90.0
        for token in base.replace("-", "_").split("_"):
            if token.startswith("fs") and token[2:].isdigit():
                fs = float(token[2:])
                break
    need = max(1, int(round(args.chunk * fs)))
    print(f"REPLAY {os.path.basename(args.replay)} - {fs:.0f} frames/s")
    frames, phs, rss = [], [], []
    history = deque(maxlen=max(args.smooth, 1))
    log = []
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
            vec, rate = chunk_score(frames, phs, rss, len(frames) / fs)
            frames, phs, rss = [], [], []
            time.sleep(args.chunk)          # real-time pace for the video
            handle(vec, rate, history, log, None)
    print("\nReplay finished.")


if args.replay:
    run_replay()
else:
    args.port = resolve_port(args.port)
    run_live()
