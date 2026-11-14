# M4_Group8 — ElderGuard: Final — Wi-Fi CSI Presence & Counting MVP

ECE 310 (Wireless Communications), Monsoon Semester 2026 — Group 8
Project category: **Minimum Viable Product (MVP)**

ElderGuard is a camera-free occupancy sensor built from two ESP32-S3 boards and
laptop-side Python. Board A floods Wi-Fi packets; Board B captures Channel State
Information (CSI); the laptop pipeline (parse → preprocess → 16 features →
classifier) classifies room occupancy live — presence and head-count (0/1/2)
every 2 seconds, no camera. The M4 release consolidates the
M3 prototype into its final form with grouped cross-validation, reliability,
latency measurements, design studies, and a capture QC gate.

## Headline results (held-out leave-one-recording-out validation)

| Question | Result |
|---|---|
| Is anyone in the room? (binary presence) | **89.0%** accuracy, 94.1% at recording level |
| How many people — 1 or 2? (when present) | **94–100% per session** |
| Verified-empty-room detection | **80.6%** |
| Latency per 2 s decision | **0.084 ms** inference (`Results/training_report.json`) |

All numbers reproduce from the committed data and code; every figure traces to a
file in `Results/`.

## Repository structure

```
M4_Group8/
├── README.md            ← this file
├── Report/
│   └── M4_Group8_Report.pdf
├── Video/
│   ├── M4_Demonstration_Video.mp4
│   └── M4_Group8_video_Link.md
├── Code/
│   ├── src/             ← shared pipeline library (parse/preprocess/features/models/evaluate)
│   ├── tools/           ← 14 command-line tools (validate, QC, train, live demo, self-test…)
│   └── requirements.txt ← exact pinned versions
├── Data/
│   ├── README.md        ← capture format + collection protocol
│   └── classroom/       ← labeled CSI captures (raw, committed)
└── Results/             ← metrics JSON, trained models, figures, validation reports
```

## Quick start (Windows, Python 3.11+)

```powershell
cd M4_Group8\Code
python -m pip install -r requirements.txt

# 1. Validate the committed captures (read-only):
python tools\validate_dataset.py --data-dir ..\Data\classroom

# 2. Train with leave-one-recording-out validation:
python tools\train_model.py --data-dir ..\Data\classroom --out-dir ..\Results
python tools\run_own.py --data-dir ..\Data\classroom --out ..\Results --task occupy --window 2.0

# 3. Live demo (Board B on USB, Board A powered) — auto-detects the port:
python tools\count_demo.py

#    No hardware? Replay a committed capture through the same live pipeline:
python tools\count_demo.py --replay ..\Data\classroom\roomA_2p_t09.csv
```

The 9-stage self-test (`python tools\selftest.py`) verifies the whole software
chain without hardware.

## What changed since M3

- Real leave-one-recording-out validation across all 17 recordings.
- Presence counting that does not need motion: 16-feature set with 4 counting
  features (subcarrier eigen-spectrum, Doppler spread) targeting the 1-vs-2
  boundary.
- Per-session RSSI link calibration (removes the rig-geometry confound).
- Capture QC gate, reliability and latency measurements, design studies
  (`Results/normalization_experiment.json`, `Results/window_experiment.json`).

See `Report/M4_Group8_Report.pdf` for the full analysis.
