# M2_Group8 — ElderGuard: Wi-Fi CSI Occupancy Sensing (M2 Technical Artifact)

ECE 310 (Wireless Communications), Monsoon Semester 2026 — Group 8
Project category: **Minimum Viable Product (MVP)**

ElderGuard is a low-cost, privacy-preserving fall-detection system built on
commodity Wi-Fi hardware. At M2 we implement and evaluate the system's first
working core function — **counting the number of people (0 / 1 / 2) in a small
room** from Wi-Fi Channel State Information (CSI) — which is the
presence/occupancy gate any fall detector must pass.

## Headline result

| Model | Accuracy | Macro-F1 | Count-MAE |
|---|---|---|---|
| Threshold (amp-variance) | 0.5154 | 0.4681 | 0.5899 |
| **Random Forest** | **0.5333** | **0.5460** | **0.4681** |
| MLP | 0.5113 | 0.5225 | 0.4976 |

Leave-one-session-out protocol, 2914 analysis windows, real CSI from the public
RadioPoints WiFi-CrowdCounting dataset (Intel 5300, 3 rooms, cited in the
report). All three models beat the 3-class chance level of 33%.

## Folder structure (M2 submission layout)

```
M2_Group8/
├── README.md              ← this file
├── Report/
│   └── M2_Group8_Report.pdf
├── Video/
│   └── M2_Group8_video_Link.md
├── Code/
│   ├── README.md
│   └── src/               ← full pipeline (parse → preprocess → features → models → evaluate)
├── Data/
│   ├── README.md
│   └── arduino/           ← ESP8266 transmitter + ESP32 receiver firmware
└── Results/
    ├── README.md
    ├── metrics.json
    └── *.png              ← model comparison, confusion matrices, feature importance
```

## Reproduce the results

```bash
cd Code/src
pip install -r requirements.txt

# Full run on the public RadioPoints dataset (writes Results/metrics.json + figures)
python run_pipeline.py --data-dir ../../Data/radiopoints --out ../../Results

# No dataset? Validate the whole chain on synthetic sessions:
python run_pipeline.py --synthetic --out /tmp/eg_synth
```

The report is generated from the same `metrics.json`, so numbers and figures
cannot drift apart:

```bash
python make_report.py --metrics ../../Results/metrics.json --out ../../Report/M2_Group8_Report.docx
```

## Status at M2

- Working core function: occupancy counting (0/1/2) from real CSI — **done**.
- System architecture, technology stack, implementation plan — **done** (report §5–§8).
- SOTA positioning — **done** (report §4).
- Own-hardware capture (ESP32-WROOM-32D RX + ESP8266 TX) — firmware complete,
  capture is the immediate M3 task.
- Next (M3): motion/activity phase detection, fall classifier, multi-room test.
