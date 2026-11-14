
import os
import glob
import csv
import numpy as np


def parse_esp32_csi_data(data_field):
    txt = data_field.strip().strip("[]")
    if not txt:
        return np.array([]), np.array([])
    raw = np.array([int(x) for x in txt.split()], dtype=float)
    if raw.size % 2:
        raise ValueError(f"CSI byte count must be even, got {raw.size}")
    if not np.all(np.isfinite(raw)) or np.any(raw < -128) or np.any(raw > 127):
        raise ValueError("CSI values must be finite signed bytes")
    imag = raw[0::2]
    real = raw[1::2]
    amp = np.sqrt(real ** 2 + imag ** 2)
    phase = np.arctan2(imag, real)
    return amp, phase


def _select_ht20_subcarriers(amp, phase):
    if amp.size == 64:
        idx = np.r_[6:32, 33:59]
        return amp[idx], phase[idx]
    return amp, phase


def load_esp32_csv(path):
    amps, phases, rssis, timestamps = [], [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:


            if row.get("type") != "CSI" or row.get("data") is None:
                continue
            try:
                a, p = parse_esp32_csi_data(row["data"])
            except ValueError:
                continue
            if a.size == 0:
                continue
            if a.size > 64 and a.size % 64 == 0:


                a, p = a[:64], p[:64]
            a, p = _select_ht20_subcarriers(a, p)
            if a.size != 52:
                continue
            try:
                rssi = float(row.get("rssi", 0) or 0)
            except ValueError:
                rssi = 0.0
            try:
                stamp = int(row.get("timestamp", ""))
            except (TypeError, ValueError):
                stamp = None
            amps.append(a)
            phases.append(p)
            rssis.append(rssi)
            timestamps.append(stamp)

    if not amps:
        raise ValueError(f"No CSI rows parsed from {path}")

    amplitude = np.vstack(amps)
    phase = np.vstack(phases)
    rssi = np.array(rssis, dtype=float)

    fs, fs_source = _estimate_fs(timestamps, path)
    meta = _parse_meta_from_name(path)
    return {
        "amplitude": amplitude,
        "phase": phase,
        "rssi": rssi,
        "fs": fs,
        "fs_source": fs_source,
        "timestamp_us": np.asarray([t if t is not None else -1 for t in timestamps], dtype=np.int64),
        "pdr": 1.0,
        **meta,
    }


def _estimate_fs(timestamps, path):
    ts = np.asarray([t for t in timestamps if t is not None and t >= 0], dtype=np.int64)
    if ts.size >= 3:
        delta = np.diff(ts)

        wrapped = (delta < -(2**31))
        delta[wrapped] += 2**32
        usable = delta[(delta > 0) & (delta <= 100_000)]
        if usable.size >= 3:
            return float(1_000_000.0 / np.median(usable)), "receiver_timestamp_median_delta"


    base = os.path.splitext(os.path.basename(path))[0]
    for token in base.replace("-", "_").split("_"):
        if token.startswith("fs") and token[2:].isdigit():
            return float(token[2:]), "filename_tag"
    return 100.0, "default_fallback"


def _parse_meta_from_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    meta = {"room": "unknown", "label": None, "session_id": base}
    for token in base.replace("-", "_").split("_"):
        if token.startswith("room"):
            meta["room"] = token[4:] or "unknown"
        if token.endswith("p") and token[:-1].isdigit():
            meta["label"] = int(token[:-1])
        if token.endswith("people") and token[:-6].isdigit():
            meta["label"] = int(token[:-6])
    return meta


def load_intel5300_dat(path, room=None, label=None, fs=100.0):
    try:
        import csiread
    except ImportError as exc:
        raise ImportError(
            "Parsing Intel 5300 .dat files needs `pip install csiread`. "
            "Alternatively provide a decoded .csv next to the .dat file."
        ) from exc

    csidata = csiread.Intel(path, nrxnum=3, ntxnum=2, if_report=False)
    csidata.read()
    csi = csidata.get_csi()
    amp = np.abs(csi).mean(axis=2)
    phase = np.angle(csi).mean(axis=2)
    rssi = np.asarray(csidata.get_rssi()).mean(axis=1)

    base = os.path.splitext(os.path.basename(path))[0]
    return {
        "amplitude": amp,
        "phase": phase,
        "rssi": rssi,
        "fs": fs,
        "pdr": 1.0,
        "room": room or "unknown",
        "label": label,
        "session_id": base,
    }


def _load_decoded_csv(path):
    import csv as _csv

    amp_rows, rssi = [], []
    with open(path, newline="") as fh:
        reader = _csv.DictReader(fh)
        amp_cols = [c for c in reader.fieldnames if c.startswith("amp_")]
        for row in reader:
            amp_rows.append([float(row[c]) for c in amp_cols])
            if "rssi" in row:
                rssi.append(float(row["rssi"]))
    amp = np.array(amp_rows)
    phase = np.zeros_like(amp)
    meta = _parse_meta_from_name(path)
    return {
        "amplitude": amp,
        "phase": phase,
        "rssi": np.array(rssi) if rssi else np.zeros(amp.shape[0]),
        "fs": _estimate_fs_from_name(path),
        "pdr": 1.0,
        **meta,
    }


def load_radiopoints(root_dir):
    sessions = []
    for path in sorted(glob.glob(os.path.join(root_dir, "**", "*"),
                                 recursive=True)):
        if not (path.endswith(".dat") or path.endswith(".csv")):
            continue
        label = _label_from_path(path)
        room = _room_from_path(path, root_dir)
        try:
            if path.endswith(".dat"):
                sessions.append(load_intel5300_dat(path, room=room, label=label))
            else:
                s = _load_decoded_csv(path)
                s["room"] = s.get("room", room)
                s["label"] = label if label is not None else s.get("label")
                sessions.append(s)
        except Exception as exc:
            print(f"  ! skipped {path}: {exc}")
    return sessions


def _label_from_path(path, root_dir=None):
    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts):
        low = part.lower()
        if "people" in low or "person" in low:
            digits = "".join(ch for ch in low if ch.isdigit())
            if digits:
                return int(digits)
            if "person" in low and "people" not in low:
                return 1
    return None


def _room_from_path(path, root_dir):
    rel = os.path.relpath(path, root_dir)
    parts = rel.replace("\\", "/").split("/")
    for part in parts:
        low = part.lower()
        if low.startswith("room"):
            return part[4:] or part
    return "unknown"
