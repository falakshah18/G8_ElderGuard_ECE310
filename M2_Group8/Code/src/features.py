"""
ElderGuard - 12-feature CSI descriptor (M2)
ECE 310 Group 8

A single 1 s analysis window is described by 12 features spanning three
families:

  Amplitude  : amp_var_mean, amp_var_std, amp_mean, amp_range_mean,
               subcarrier_corr_mean
  Phase      : phase_var_mean
  Doppler    : doppler_energy, doppler_peak_freq, doppler_centroid
  Link       : rssi_mean, rssi_std, pdr

These are the features whose Random Forest importances are reported in
Results/feature_importance.png.
"""

import numpy as np
from scipy.signal import periodogram

FEATURE_NAMES = [
    "amp_var_mean",
    "amp_var_std",
    "amp_mean",
    "amp_range_mean",
    "subcarrier_corr_mean",
    "phase_var_mean",
    "doppler_energy",
    "doppler_peak_freq",
    "doppler_centroid",
    "rssi_mean",
    "rssi_std",
    "pdr",
]


def _mean_pairwise_subcarrier_corr(window_amp):
    """Mean absolute correlation between neighbouring subcarrier time series.

    Motion decorrelates subcarriers; a still / empty room keeps them highly
    correlated. Computed on up to 8 evenly spaced subcarriers for speed.
    """
    n_sub = window_amp.shape[1]
    if n_sub < 2:
        return 0.0
    idx = np.linspace(0, n_sub - 1, min(8, n_sub)).astype(int)
    sub = window_amp[:, idx]
    corr = np.corrcoef(sub, rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(np.abs(vals))) if vals.size else 0.0


def _doppler_features(mean_amp_1d, fs):
    """Doppler-spectrum descriptors of the window-averaged amplitude."""
    freqs, psd = periodogram(mean_amp_1d, fs=fs)
    band = (freqs >= 0.5) & (freqs <= 10.0)
    f = freqs[band]
    p = psd[band]
    if p.sum() <= 0:
        return 0.0, 0.0, 0.0
    energy = float(p.sum())
    peak_freq = float(f[np.argmax(p)])
    centroid = float(np.sum(f * p) / np.sum(p))
    return energy, peak_freq, centroid


def extract_features(window_amp, window_phase, window_rssi, window_pdr, fs):
    """Return the 12-element feature vector for one analysis window.

    window_amp   [T, n_subcarriers]  filtered amplitude
    window_phase [T, n_subcarriers]  sanitized + filtered phase
    window_rssi  [T] or scalar
    window_pdr   scalar packet-delivery ratio
    """
    amp_var_per_sub = np.var(window_amp, axis=0)
    amp_mean_per_sub = np.mean(window_amp, axis=0)
    amp_range_per_sub = np.max(window_amp, axis=0) - np.min(window_amp, axis=0)

    mean_amp_1d = window_amp.mean(axis=1)
    doppler_energy, doppler_peak_freq, doppler_centroid = _doppler_features(
        mean_amp_1d, fs
    )

    rssi = np.atleast_1d(np.asarray(window_rssi, dtype=float))

    return np.array(
        [
            float(np.mean(amp_var_per_sub)),          # amp_var_mean
            float(np.std(amp_var_per_sub)),           # amp_var_std
            float(np.mean(amp_mean_per_sub)),         # amp_mean
            float(np.mean(amp_range_per_sub)),        # amp_range_mean
            _mean_pairwise_subcarrier_corr(window_amp),
            float(np.var(window_phase)),              # phase_var_mean
            doppler_energy,
            doppler_peak_freq,
            doppler_centroid,
            float(np.mean(rssi)) if rssi.size else 0.0,
            float(np.std(rssi)) if rssi.size else 0.0,
            float(window_pdr),
        ],
        dtype=float,
    )


def featurize_session(amplitude, phase, rssi, fs, window_s=1.0, overlap=0.5,
                      pdr=1.0):
    """Slide over a preprocessed session and return (X, starts).

    X has shape [n_windows, 12]. `starts` holds the window start sample index
    so callers can keep windows aligned with their session metadata.
    """
    from preprocess import sliding_windows  # local import to avoid cycles

    feats = []
    starts = []
    for start, stop in sliding_windows(amplitude, fs, window_s, overlap):
        w_amp = amplitude[start:stop]
        w_phase = phase[start:stop]
        w_rssi = rssi[start:stop] if np.ndim(rssi) else rssi
        feats.append(extract_features(w_amp, w_phase, w_rssi, pdr, fs))
        starts.append(start)
    if not feats:
        return np.empty((0, len(FEATURE_NAMES))), np.array([], dtype=int)
    return np.vstack(feats), np.array(starts, dtype=int)
