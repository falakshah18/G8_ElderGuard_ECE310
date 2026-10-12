# Code — M1 signal-processing pipeline

## Files

- `preprocess.py` — the pipeline itself: synthetic CSI generator (stand-in for
  real data until the loader below is finished), phase sanitization, bandpass
  filter (0.5–10 Hz), and STFT-based Doppler spectrogram. Running it directly
  also runs a self-test that checks the fall window shows higher movement
  energy than the quiet window.
- `load_dataset.py` — defines the `CSISample` interface and the
  leave-one-person/environment-out split logic that the rest of the project
  will use once the Widar 3.0 / UT-HAR loaders are filled in (M2).

## Requirements

```
pip install numpy scipy matplotlib
```

## Run

```
python3 preprocess.py
```

Expected output: shapes of each intermediate array, plus a sanity check
confirming the synthetic fall window has higher filtered-signal energy than
the quiet window before/after it.

To regenerate the demo plot in `../../Results/synthetic_fall_spectrogram.png`,
see the plotting snippet referenced in the top-level README, or adapt the
`__main__` block in `preprocess.py` to call `matplotlib`.

## Next steps (M2)

- Implement `load_widar3` / `load_ut_har` in `load_dataset.py` against the
  real downloaded dataset files.
- Replace the synthetic input in `preprocess.py`'s demo with real loaded
  samples.
- Add the 1D-CNN model and the RSSI-threshold / Random Forest baselines under
  `/ml_models`.
