# M4_Group8 — ElderGuard: Final — Wi-Fi CSI Presence & Counting MVP

ECE 310 (Wireless Communications), Monsoon Semester 2026 — Group 8
Project category: **Minimum Viable Product (MVP)**

ElderGuard is a camera-free occupancy sensor built from two ESP32-S3 boards and
laptop-side Python. Board A floods Wi-Fi packets; Board B captures Channel State
Information (CSI); the laptop pipeline (parse → preprocess → 12+ features →
Random Forest) classifies room occupancy live. The M4 release consolidates the
M3 prototype into its final form with grouped cross-validation, reliability,
latency and range measurements, and an ablation study.

## Headline results (final numbers in Results/)

| Question | Result |
|---|---|
| Is the room occupied? (binary) | see `Results/metrics_*.json` |
| How many people (0/1/2)? | see `Results/training_report.json` |
| Latency per decision | see `training_report.json` → `folds[].inference_ms_per_window` |
| Range behaviour | see `Results/range_test.json` |
| Reliability across trials | see `Results/reliability.json` |

## Repository structure

```
M4_Group8/
├── README.md            ← this file
├── Report/
│   └── M4_Group8_Report.pdf
├── Video/
│   └── M4_Group8_video_Link.md
├── Code/
│   ├── src/             ← shared pipeline library (parse/preprocess/features/models/evaluate)
│   ├── tools/           ← 8 command-line tools (validate, QC, train, live demo, self-test…)
│   └── requirements.txt ← exact pinned versions
├── Data/
│   ├── README.md        ← capture format + collection protocol
│   └── classroom/       ← labeled CSI captures (raw, committed)
├── Results/             ← metrics JSON, figures, validation reports
└── LAB_DAY_RUNBOOK.md   ← the exact protocol used to collect Data/ (in M3_Group8/)
```

## Quick start (Windows, Python 3.11+)

```powershell
cd M4_Group8\Code
python -m pip install -r requirements.txt

# 1. Validate the committed captures (read-only):
python tools\validate_dataset.py --data-dir ..\Data\classroom

# 2. Train with leave-one-recording-out validation:
python tools\train_model.py --data-dir ..\Data\classroom --out-dir ..\Results

# 3. Live demo (Board B on USB, Board A powered):
python tools\live_demo.py --port auto --baud 921600 --mode count --binary ^
    --model ..\Results\model_occupy.joblib
```

The 9-stage self-test (`python tools\selftest.py`) verifies the whole software
chain without hardware.

## What changed since M3

- Real leave-one-recording-out validation (≥2 recordings per class).
- Extended feature set (motion-source counting via subcarrier correlation
  eigen-spectrum; per-subcarrier Doppler statistics; measured PDR).
- Range, reliability and latency measurements on the target link.
- Ablation study: 12-feature baseline vs extended feature set.

See `Report/M4_Group8_Report.pdf` for the full analysis.
