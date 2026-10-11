# Code — M2 occupancy pipeline

End-to-end implementation of the M2 core function: CSI parsing → preprocessing →
12-feature extraction → threshold / Random Forest / MLP models →
leave-one-session-out evaluation → metrics + figures.

## Files (`src/`)

| File | Role |
|---|---|
| `parse_csi.py` | ESP32-CSI-Tool CSV → amplitude/phase; Intel 5300 (RadioPoints) loader |
| `preprocess.py` | Hampel filter, phase sanitization, 0.5–10 Hz Butterworth bandpass, sliding windows |
| `features.py` | 12-feature descriptor per 1 s window |
| `models.py` | Threshold baseline, Random Forest (300 trees), MLP (64-32) |
| `evaluate.py` | Leave-one-session-out CV; accuracy / macro-F1 / count-MAE / confusion matrix |
| `run_pipeline.py` | End-to-end entry point; writes `Results/metrics.json` + `features.npz` |
| `make_figures.py` | Regenerates all figures from `metrics.json` |
| `make_report.py` | Builds the report .docx from `metrics.json` |
| `synthetic.py` | Controlled CSI session generator for pipeline self-test |
| `collect_serial.py` | Logs CSI frames from the ESP32 to CSV |
| `flood_udp.py` | Laptop-side UDP traffic source (fallback transmitter) |
| `list_ports.py` | Lists available serial ports |

## Requirements

```bash
pip install -r requirements.txt
```

`csiread` is only needed to parse Intel 5300 `.dat` files directly; the loader
falls back to decoded `.csv` files if it is not installed.

## Run

```bash
# Real dataset (reported results)
python run_pipeline.py --data-dir ../../Data/radiopoints --out ../../Results

# Synthetic self-test (no dataset required)
python run_pipeline.py --synthetic --out /tmp/eg_synth

# Regenerate figures only
python make_figures.py --metrics ../../Results/metrics.json --out ../../Results
```

## Evaluation note

Windows from the same capture are correlated, so a random window split leaks
near-duplicate windows into both train and test. `evaluate.py` holds out a
whole session at a time (leave-one-session-out); with the RadioPoints data the
sessions are (room × occupancy) captures, making this effectively
leave-one-room-out as well.
