# Code — ElderGuard M4 pipeline

Final implementation: CSI parsing → preprocessing → 16-feature descriptor
(12-feature M3 baseline + 4 counting features) → models → grouped validation →
metrics generation.

## Layout

| Path | Role |
|---|---|
| `src/parse_csi.py` | ESP32-S3 CSI CSV parser (I/Q, 52 HT20 subcarriers, timestamp-based fs) |
| `src/preprocess.py` | Hampel filter, phase sanitization, 0.5–10 Hz Butterworth, sliding windows |
| `src/features.py` | 16-feature descriptor; `BASE12_FEATURE_NAMES` = M3 baseline for the ablation |
| `src/models.py` | Threshold baseline, Random Forest, MLP, Gradient Boosting |
| `src/evaluate.py` | Leave-one-session-out cross-validation |
| `src/run_pipeline.py` | End-to-end entry point (public-dataset / synthetic modes) |
| `src/make_figures.py` | Figures from metrics JSON |
| `src/synthetic.py` | Pipeline self-test fixture generator |
| `src/collect_serial.py` | Serial CSI logger with USB-drop reconnect |
| `src/eg_port.py` | USB VID-based port autodetection (`--port auto`) |
| `tools/validate_dataset.py` | Read-only capture integrity audit |
| `tools/check_captures.py` | Label-vs-signal QC gate |
| `tools/train_model.py` | Trainer: LOSO over 7 models, writes model + report artifacts |
| `tools/live_demo.py` | Live count / fall / binary modes; hardware or `--replay` |
| `tools/m4_metrics.py` | **M4 evidence generator: validation, ablation, range, reliability** |
| `tools/run_own.py` | Own-data trainer (count / occupy tasks) |
| `tools/make_fake_captures.py` | Synthetic CSV generator (fixtures, never results) |
| `tools/selftest.py` | 9-stage no-hardware environment verification |
| `firmware/` | TX (soft-AP UDP flood) and RX (CSI capture) sketches |

## Install & verify

```powershell
python -m pip install -r requirements.txt
python tools\selftest.py          # expect 9/9 PASS, no hardware needed
```

## Reproduce all M4 numbers

```powershell
python tools\validate_dataset.py --data-dir ..\Data\classroom
python tools\check_captures.py --data-dir ..\Data\classroom
python tools\train_model.py --data-dir ..\Data\classroom --out-dir ..\Results
python tools\m4_metrics.py --data-dir ..\Data\classroom --out ..\Results
```

`m4_metrics.py` writes `m4_counting.json`, `m4_ablation.json` (12 vs 16 features,
same protocol), `m4_range.json`, `m4_reliability.json`, `m4_error_analysis.json`
and `M4_METRICS.md` — the numbers in the report come from these files.

## Live demo

```powershell
python tools\live_demo.py --port auto --baud 921600 --mode count --binary ^
    --model ..\Results\model_occupy.joblib
```
