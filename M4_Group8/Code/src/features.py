
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
    "eig_amp1",
    "eig_amp2",
    "eig_effrank",
    "doppler_peak_spread",
]


BASE12_FEATURE_NAMES = FEATURE_NAMES[:12]


def _mean_pairwise_subcarrier_corr(window_amp):
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


def _eigen_features(window_amp):
    n_sub = window_amp.shape[1]
    if n_sub < 3 or window_amp.shape[0] < 5:
        return 0.0, 0.0, 1.0
    corr = np.corrcoef(window_amp, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    ev = np.linalg.eigvalsh(corr)
    ev = np.clip(ev, 0.0, None)
    total = float(ev.sum())
    if total <= 0:
        return 0.0, 0.0, 1.0
    p = ev / total
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    top = np.sort(ev)[::-1] / total
    return float(top[0]), float(top[1] if top.size > 1 else 0.0), eff_rank


def _doppler_peak_spread(window_amp, fs):
    if window_amp.shape[0] < 8:
        return 0.0
    freqs, psd = periodogram(window_amp, fs=fs, axis=0)
    band = (freqs >= 0.5) & (freqs <= 10.0)
    p = psd[band]
    if p.shape[0] < 2:
        return 0.0
    peaks = np.argmax(p, axis=0) / float(p.shape[0])
    return float(np.std(peaks))

def extract_features(window_amp, window_phase, window_rssi, window_pdr, fs):
    amp_var_per_sub = np.var(window_amp, axis=0)
    amp_mean_per_sub = np.mean(window_amp, axis=0)
    amp_range_per_sub = np.max(window_amp, axis=0) - np.min(window_amp, axis=0)

    mean_amp_1d = window_amp.mean(axis=1)
    doppler_energy, doppler_peak_freq, doppler_centroid = _doppler_features(
        mean_amp_1d, fs
    )

    eig1, eig2, eff_rank = _eigen_features(window_amp)
    peak_spread = _doppler_peak_spread(window_amp, fs)

    rssi = np.atleast_1d(np.asarray(window_rssi, dtype=float))

    return np.array(
        [
            float(np.mean(amp_var_per_sub)),
            float(np.std(amp_var_per_sub)),
            float(np.mean(amp_mean_per_sub)),
            float(np.mean(amp_range_per_sub)),
            _mean_pairwise_subcarrier_corr(window_amp),
            float(np.var(window_phase)),
            doppler_energy,
            doppler_peak_freq,
            doppler_centroid,
            float(np.mean(rssi)) if rssi.size else 0.0,
            float(np.std(rssi)) if rssi.size else 0.0,
            float(window_pdr),
            eig1,
            eig2,
            eff_rank,
            peak_spread,
        ],
        dtype=float,
    )


def featurize_session(amplitude, phase, rssi, fs, window_s=1.0, overlap=0.5,
                      pdr=1.0):
    from preprocess import sliding_windows

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
