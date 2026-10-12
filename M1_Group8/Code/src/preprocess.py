"""
ElderGuard - CSI preprocessing pipeline (M1 first pass)
ECE 310 Group 8

Stages implemented:
  1. Phase sanitization: unwrap raw CSI phase across subcarriers and remove the
     linear component caused by timing offset / carrier frequency offset (CFO).
  2. Bandpass filtering: keep only the 0.5-10 Hz band where human movement lives,
     using a 4th-order Butterworth filter (matches the range described in the
     proposal).
  3. Doppler spectrogram: Short-Time Fourier Transform (STFT) of the filtered
     amplitude/phase signal, producing a time-frequency map that a fall shows up
     in as a short burst of high-Doppler energy followed by near-silence.

This script currently runs on a synthetically generated CSI-like signal (a
walking-like oscillation followed by a fall-like impulse and stillness) so the
pipeline can be validated end-to-end before the public dataset (Widar 3.0 /
UT-HAR) loader is finished. Swap `generate_synthetic_csi()` for a real loader
in load_dataset.py once that is ready -- everything downstream is unchanged.
"""

import numpy as np
from scipy.signal import butter, filtfilt, stft

# ---------------------------------------------------------------------------
# 1. Synthetic CSI generator (stand-in for the public dataset loader)
# ---------------------------------------------------------------------------

def generate_synthetic_csi(duration_s=6.0, fs=100, n_subcarriers=52, seed=0):
    """Builds a fake but structurally realistic CSI amplitude+phase trace:
    - 0-3s: quasi-static (small random jitter only)
    - 3-3.5s: a fast, high-amplitude fall-like burst
    - 3.5-6s: stillness again (person on the floor)
    Returns amplitude [T, n_subcarriers] and raw phase [T, n_subcarriers].
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs)
    n = len(t)

    base_amp = 20 + rng.normal(0, 0.3, size=(n, n_subcarriers))
    fall_mask = (t >= 3.0) & (t < 3.5)
    fall_signal = 8 * np.sin(2 * np.pi * 6 * (t - 3.0))[:, None]
    amplitude = base_amp.copy()
    amplitude[fall_mask] += fall_signal[fall_mask]

    # Raw phase = true small motion phase + CFO-driven linear drift + noise
    true_phase = 0.05 * np.sin(2 * np.pi * 1.5 * t)[:, None] * np.ones((1, n_subcarriers))
    true_phase[fall_mask] += 0.6 * np.sin(2 * np.pi * 6 * (t[fall_mask] - 3.0))[:, None]
    cfo_drift = (0.02 * np.arange(n))[:, None] * np.ones((1, n_subcarriers))
    noise = rng.normal(0, 0.05, size=(n, n_subcarriers))
    raw_phase = true_phase + cfo_drift + noise

    return t, amplitude, raw_phase


# ---------------------------------------------------------------------------
# 2. Phase sanitization
# ---------------------------------------------------------------------------

def sanitize_phase(raw_phase):
    """Unwraps phase along the subcarrier axis, then removes the CFO/timing
    linear trend per packet by fitting and subtracting a straight line across
    subcarriers (a standard CSI phase-calibration step).
    """
    unwrapped = np.unwrap(raw_phase, axis=1)
    subcarrier_idx = np.arange(unwrapped.shape[1])
    cleaned = np.empty_like(unwrapped)
    for i in range(unwrapped.shape[0]):
        # Fit a line (slope, intercept) to this packet's phase vs subcarrier index
        slope, intercept = np.polyfit(subcarrier_idx, unwrapped[i], 1)
        cleaned[i] = unwrapped[i] - (slope * subcarrier_idx + intercept)
    return cleaned


# ---------------------------------------------------------------------------
# 3. Bandpass filter (0.5-10 Hz human-movement band)
# ---------------------------------------------------------------------------

def bandpass_filter(signal, fs, low=0.5, high=10.0, order=4):
    """4th-order Butterworth bandpass, applied per-subcarrier along the time
    axis (axis=0). filtfilt is used for zero phase distortion.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


# ---------------------------------------------------------------------------
# 4. Doppler spectrogram via STFT
# ---------------------------------------------------------------------------

def doppler_spectrogram(signal_1d, fs, nperseg=64, noverlap=48):
    """Runs an STFT on a single (already filtered) time series -- e.g. the
    mean across subcarriers -- and returns frequency bins, time bins, and the
    magnitude spectrogram.
    """
    f, t_stft, Zxx = stft(signal_1d, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t_stft, np.abs(Zxx)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fs = 100
    t, amplitude, raw_phase = generate_synthetic_csi(fs=fs)

    clean_phase = sanitize_phase(raw_phase)
    filtered_amp = bandpass_filter(amplitude, fs)
    filtered_phase = bandpass_filter(clean_phase, fs)

    mean_filtered_amp = filtered_amp.mean(axis=1)
    f, t_stft, S = doppler_spectrogram(mean_filtered_amp, fs)

    print("Pipeline ran successfully on synthetic CSI.")
    print(f"  Input shape (T x subcarriers): {amplitude.shape}")
    print(f"  Sanitized phase shape:         {clean_phase.shape}")
    print(f"  Filtered amplitude shape:      {filtered_amp.shape}")
    print(f"  Spectrogram shape (freq x time): {S.shape}")

    # Quick sanity check: energy in the filtered signal should spike during
    # the synthetic fall window (t in [3.0, 3.5]) relative to before/after.
    fall_idx = (t >= 3.0) & (t < 3.5)
    quiet_idx = (t < 2.5)
    fall_energy = np.mean(np.abs(mean_filtered_amp[fall_idx]))
    quiet_energy = np.mean(np.abs(mean_filtered_amp[quiet_idx]))
    print(f"  Mean |signal| during fall window:  {fall_energy:.3f}")
    print(f"  Mean |signal| during quiet window: {quiet_energy:.3f}")
    assert fall_energy > quiet_energy, "Expected higher energy during the fall window"
    print("  Sanity check passed: fall window shows higher movement energy.")
