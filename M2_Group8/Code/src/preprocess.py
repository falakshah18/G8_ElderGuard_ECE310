"""
ElderGuard - CSI preprocessing pipeline (M2)
ECE 310 Group 8

Stages:
  1. Hampel filter        - removes impulsive CSI spikes (per subcarrier).
  2. Phase sanitization   - unwrap across subcarriers, then remove the linear
                            CFO / sampling-time-offset trend per packet.
  3. Bandpass filter      - 4th-order zero-phase Butterworth, 0.5-10 Hz, the
                            band where human movement (and falls) live.
  4. Sliding windows      - 1 s windows with 50% overlap.

Everything downstream (features.py, models.py, evaluate.py) consumes the
output of this module, so the same code path is used for the public
RadioPoints dataset and for our own ESP32 captures at M3.
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# 1. Hampel outlier removal
# ---------------------------------------------------------------------------

def hampel_filter(signal, window=5, n_sigma=3.0):
    """Per-subcarrier Hampel filter.

    For every sample we take a centred window, compute the median and the
    median absolute deviation (MAD), and replace the sample by the median when
    it deviates by more than n_sigma * 1.4826 * MAD. This removes the
    impulsive spikes that CSI captures occasionally contain without smearing
    genuine motion.

    `signal` is [T, n_subcarriers]; filtering is applied along axis 0.
    """
    signal = np.asarray(signal, dtype=float)
    out = signal.copy()
    half = window // 2
    n = signal.shape[0]
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        block = signal[lo:hi]
        med = np.median(block, axis=0)
        mad = np.median(np.abs(block - med), axis=0)
        threshold = n_sigma * 1.4826 * mad
        replace = np.abs(signal[i] - med) > threshold
        out[i, replace] = med[replace]
    return out


# ---------------------------------------------------------------------------
# 2. Phase sanitization
# ---------------------------------------------------------------------------

def sanitize_phase(raw_phase):
    """Unwrap phase across subcarriers, then subtract the per-packet linear
    trend (the CFO / timing-offset component) by fitting a straight line over
    the subcarrier index and removing it.
    """
    unwrapped = np.unwrap(np.asarray(raw_phase, dtype=float), axis=1)
    subcarrier_idx = np.arange(unwrapped.shape[1])
    cleaned = np.empty_like(unwrapped)
    for i in range(unwrapped.shape[0]):
        slope, intercept = np.polyfit(subcarrier_idx, unwrapped[i], 1)
        cleaned[i] = unwrapped[i] - (slope * subcarrier_idx + intercept)
    return cleaned


# ---------------------------------------------------------------------------
# 3. Bandpass filter (0.5-10 Hz human-movement band)
# ---------------------------------------------------------------------------

def bandpass_filter(signal, fs, low=0.5, high=10.0, order=4):
    """4th-order Butterworth bandpass applied along the time axis (axis 0).
    filtfilt gives zero phase distortion, so the time location of a motion
    event is preserved - important for fall-phase detection later.
    """
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid band {low}-{high} Hz for fs={fs} Hz")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


# ---------------------------------------------------------------------------
# 4. Sliding windows
# ---------------------------------------------------------------------------

def sliding_windows(signal, fs, window_s=1.0, overlap=0.5):
    """Yield (start, stop) sample indices for window_s windows with the given
    fractional overlap. Used to chop a session into analysis windows.
    """
    win = int(round(window_s * fs))
    step = max(1, int(round(win * (1.0 - overlap))))
    for start in range(0, signal.shape[0] - win + 1, step):
        yield start, start + win


# ---------------------------------------------------------------------------
# Convenience: run the full chain on one session
# ---------------------------------------------------------------------------

def preprocess_session(amplitude, raw_phase, fs, do_phase=True):
    """Apply the complete preprocessing chain and return the cleaned amplitude
    and phase arrays (same shape as the input)."""
    amp = hampel_filter(amplitude)
    amp = bandpass_filter(amp, fs)
    if do_phase:
        phase = sanitize_phase(raw_phase)
        phase = bandpass_filter(phase, fs)
    else:
        phase = raw_phase
    return amp, phase
