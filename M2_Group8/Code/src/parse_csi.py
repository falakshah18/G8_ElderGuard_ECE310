"""
ElderGuard - CSI parsers (M2)
ECE 310 Group 8

Two input formats:

  * ESP32-CSI-Tool CSV  - produced by our own receiver firmware
                          (Data/arduino/receiver_csi). Columns:
                          type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,
                          smoothing,not_sounding,aggregation,stbc,fec_coding,
                          sgi,ampdu,amsdu,length,first_word,data
                          `data` is the raw signed-byte CSI buffer; pairs are
                          (imag, real). We keep subcarriers 6..31 and 33..58
                          (the 52 non-null 20 MHz subcarriers) when present.

  * Intel 5300 .dat     - the RadioPoints WiFi-CrowdCounting dataset used for
                          the M2 measured results. Parsing is delegated to the
                          `csiread` package when available; if it is not
                          installed we fall back to a pre-decoded .csv beside
                          each .dat file (see Data/README.md).

A "session" is one capture and carries its own metadata (room, occupancy).
"""

import os
import glob
import csv
import numpy as np


# ---------------------------------------------------------------------------
# ESP32-CSI-Tool CSV
# ---------------------------------------------------------------------------

def parse_esp32_csi_data(data_field):
    """Decode the `data` field of an ESP32-CSI-Tool CSV row into (amp, phase).

    The field is a bracketed list of signed bytes; every consecutive pair is
    (imag, real). Returns 1-D amplitude and phase arrays over subcarriers.
    """
    txt = data_field.strip().strip("[]")
    if not txt:
        return np.array([]), np.array([])
    raw = np.array([int(x) for x in txt.split()], dtype=float)
    if raw.size % 2:
        raw = raw[:-1]
    imag = raw[0::2]
    real = raw[1::2]
    amp = np.sqrt(real ** 2 + imag ** 2)
    phase = np.arctan2(imag, real)
    return amp, phase


def _select_ht20_subcarriers(amp, phase):
    """Keep the 52 usable 20 MHz subcarriers when a 64-subcarrier buffer is
    given (drop null/guard/pilot edges), otherwise return as-is."""
    if amp.size == 64:
        idx = np.r_[6:32, 33:59]
        return amp[idx], phase[idx]
    return amp, phase


def load_esp32_csv(path):
    """Load one ESP32-CSI-Tool CSV capture into a session dict."""
    amps, phases, rssis = [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("data") is None:
                continue
            a, p = parse_esp32_csi_data(row["data"])
            if a.size == 0:
                continue
            a, p = _select_ht20_subcarriers(a, p)
            amps.append(a)
            phases.append(p)
            rssis.append(float(row.get("rssi", 0) or 0))

    if not amps:
        raise ValueError(f"No CSI rows parsed from {path}")

    amplitude = np.vstack(amps)
    phase = np.vstack(phases)
    rssi = np.array(rssis, dtype=float)
    # ESP32-CSI-Tool reports RSSI in dBm already.
    fs = _estimate_fs_from_name(path)
    meta = _parse_meta_from_name(path)
    return {
        "amplitude": amplitude,
        "phase": phase,
        "rssi": rssi,
        "fs": fs,
        "pdr": 1.0,
        **meta,
    }


def _estimate_fs_from_name(path):
    """Sessions may encode their sampling rate as `..._fs100_...`; default 100."""
    base = os.path.basename(path)
    for token in base.replace("-", "_").split("_"):
        if token.startswith("fs") and token[2:].isdigit():
            return float(token[2:])
    return 100.0


def _parse_meta_from_name(path):
    """Best-effort metadata from a filename like `roomA_1p_session03.csv`."""
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


# ---------------------------------------------------------------------------
# Intel 5300 (RadioPoints) loader
# ---------------------------------------------------------------------------

def load_intel5300_dat(path, room=None, label=None, fs=100.0):
    """Parse one Intel 5300 .dat capture with csiread.

    Returns a session dict with amplitude/phase [T, n_subcarriers].
    """
    try:
        import csiread  # noqa: WPS433 (optional dependency)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Parsing Intel 5300 .dat files needs `pip install csiread`. "
            "Alternatively provide a decoded .csv next to the .dat file."
        ) from exc

    csidata = csiread.Intel(path, nrxnum=3, ntxnum=2, if_report=False)
    csidata.read()
    csi = csidata.get_csi()          # [n_packets, 30, 3]
    amp = np.abs(csi).mean(axis=2)   # average the 3 RX antennas
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
    """Fallback: a decoded CSV with one row per packet and amplitude columns
    `amp_0 ... amp_N` plus an optional `rssi` column."""
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
    phase = np.zeros_like(amp)  # decoded CSVs carry amplitude only
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
    """Load the RadioPoints WiFi-CrowdCounting dataset directory.

    Expected layout (documented in Data/README.md):
        root_dir/
          roomA/0_people/*.dat  (or .csv)
          roomA/1_person/*.dat
          ...
    Directory names encode the occupancy; the parent encodes the room.
    """
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
        except Exception as exc:  # keep going; report the bad file
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
