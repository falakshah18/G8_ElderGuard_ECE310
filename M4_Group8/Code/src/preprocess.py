
import numpy as np
from scipy.signal import butter, filtfilt


def hampel_filter(signal, window=5, n_sigma=3.0):
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


def sanitize_phase(raw_phase):
    unwrapped = np.unwrap(np.asarray(raw_phase, dtype=float), axis=1)
    subcarrier_idx = np.arange(unwrapped.shape[1])
    cleaned = np.empty_like(unwrapped)
    for i in range(unwrapped.shape[0]):
        slope, intercept = np.polyfit(subcarrier_idx, unwrapped[i], 1)
        cleaned[i] = unwrapped[i] - (slope * subcarrier_idx + intercept)
    return cleaned


def bandpass_filter(signal, fs, low=0.5, high=10.0, order=4):
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid band {low}-{high} Hz for fs={fs} Hz")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def sliding_windows(signal, fs, window_s=1.0, overlap=0.5):
    win = int(round(window_s * fs))
    step = max(1, int(round(win * (1.0 - overlap))))
    for start in range(0, signal.shape[0] - win + 1, step):
        yield start, start + win


def preprocess_session(amplitude, raw_phase, fs, do_phase=True):
    amp = hampel_filter(amplitude)
    amp = bandpass_filter(amp, fs)
    if do_phase:
        phase = sanitize_phase(raw_phase)
        phase = bandpass_filter(phase, fs)
    else:
        phase = raw_phase
    return amp, phase
