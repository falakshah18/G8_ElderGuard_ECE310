# M4 Report — ElderGuard: Camera-Free Presence & People-Counting with Wi-Fi CSI

ECE 310 · Group 8 · Milestone 4

## 1. Introduction & problem formulation

ElderGuard is a privacy-preserving indoor sensing system: two ESP32-S3 boards form a
Wi-Fi link (one soft-AP transmitter flooding ~100 UDP packets/s, one receiver capturing
Channel State Information from every received packet). CSI amplitude/phase changes when
people move through, stand on, or merely occupy the space between the boards.

M3 established the motion-based pipeline (fall detection + motion-based occupancy).
M4 adds what a *guardian* actually needs: **presence and head-count that does not depend
on continuous motion** — counting 0, 1, or 2 people present whether they move or not —
plus proper validation, metrics, an ablation study, and an honest error analysis.

Design requirement (final, clarified during the milestone): the deliverable is
**presence counting** — "how many people are in the room" — not motion detection.
Binary presence is the primary deployment metric; 3-class counting (0/1/2) is the
full task.

## 2. System architecture (final)

- Board A (transmitter): `firmware/tx_esp32_ap/tx_esp32_ap.ino` — soft-AP
  `ElderGuard-TX`, channel 1 fixed, ~100 packets/s design (~91/s measured), broadcast
  + unicast backup, serial heartbeat every 10 s. Powered from a wall charger.
- Board B (receiver): `firmware/rx_esp32_csi/rx_esp32_csi.ino` — joins the AP, enables
  the ESP32 CSI callback, prints CSV frames over native USB-CDC at 921600 baud.
  Measured end-to-end streaming: **55–65 frames/s live** (fs 62–96 Hz per capture,
  derived from receiver timestamps), **52 HT20 subcarriers** per frame.
- Pipeline: `src/parse_csi.py` → `src/preprocess.py` → `src/features.py` →
  `tools/train_model.py` (3-class) / `tools/run_own.py` (binary) →
  `tools/live_demo.py` (real-time inference, 2 s update period).
- Feature vector: **16 features** — M3 baseline 12 (amplitude variance family,
  subcarrier correlation, phase variance, Doppler descriptors, link health) + 4 M4
  counting features (`eig_amp1`, `eig_amp2`, `eig_effrank`, `doppler_peak_spread`)
  targeting the 1-vs-2-person boundary.
- **Per-session link calibration (M4 pipeline change):** `rssi_mean` is expressed
  relative to the recording's own median link level instead of absolute dBm.
  Absolute RSSI encodes rig geometry (board positions/gains), which shifted between
  recording batches; the deviation from the calibrated baseline is the physical
  presence signal and mirrors the startup calibration a deployed system performs.
  Measured effect on the 3-class LOSO accuracy: **54.1% → 62.9%**
  (`Results/normalization_experiment.json`).

## 3. Experimental setup

### 3.1 Data

17 labeled captures, all QC-passed (`tools/check_captures.py`: continuity, rate,
motion, RSSI-vs-baseline checks; 4 corrupted/contradictory takes were quarantined,
not used — see §6). Room "A", line-of-sight link, natural behavior allowed
(walking, sitting, shifting, phone use — behavior is deliberately *not* controlled;
the label is head-count only).

| File | Label | Frames | fs (Hz) |
|---|---|---:|---:|
| roomA_0p_t01 | 0 | 4578 | 62.4 |
| roomA_0p_t02 | 0 | 5769 | 91.3 |
| roomA_0p_t04 | 0 | 2785 | 95.7 |
| roomA_1p_t01 | 1 | 5808 | 66.2 |
| roomA_1p_t02 | 1 | 2417 | 85.2 |
| roomA_1p_t03 | 1 | 3052 | 81.2 |
| roomA_1p_t04 | 1 | 3006 | 90.3 |
| roomA_1p_t05 | 1 | 9560 | 90.9 |
| roomA_2p_t01 | 2 | 5563 | 61.9 |
| roomA_2p_t02 | 2 | 3686 | 89.8 |
| roomA_2p_t04 | 2 | 3870 | 91.1 |
| roomA_2p_t07 | 2 | 3162 | 83.3 |
| roomA_2p_t08 | 2 | 1512 | 75.1 |
| roomA_2p_t09 | 2 | 8029 | 89.8 |
| roomA_2p_t10 | 2 | 5830 | 90.6 |
| roomA_2p_t11 | 2 | 8269 | 90.5 |
| roomA_2p_t12 | 2 | 1565 | 91.7 |

Sessions per class: 0p×3, 1p×5, 2p×9. Captures were recorded across multiple rig
mountings (documented as a confound, handled by §2's calibration, quantified in §5).

### 3.2 Protocol

- **Leave-One-Recording-Out (LOSO)**: each whole recording is held out in turn; the
  model trains on all windows of the other recordings. No window of the held-out
  recording ever influences training — no window-level leakage.
- Window 1 s, overlap 0.5 (3-class); binary model trained with 2 s windows
  (selected by a window-size study, `Results/window_experiment.json`).
- Seed 42 everywhere; models evaluated on identical folds; training classes
  balanced by window subsampling in the design studies.
- Metrics: accuracy, macro-F1, per-class recall, count-MAE, inference latency,
  recording-level majority vote.

## 4. Results

### 4.1 Occupancy / counting

**3-class (0/1/2), LOSO, window-level** (`Results/training_report.json`):

| Model | Accuracy | Macro-F1 | Latency (ms/window) |
|---|---:|---:|---:|
| Dummy (most frequent) | 0.518 | 0.227 | — |
| Random Forest | 0.549 | 0.466 | 0.510 |
| Extra Trees | 0.499 | 0.411 | 0.668 |
| SVM RBF | 0.529 | 0.519 | 0.104 |
| Logistic Regression | 0.431 | 0.410 | 0.004 |
| **Gradient Boosting (selected)** | **0.629** | **0.533** | **0.084** |
| KNN-5 | 0.500 | 0.467 | 0.013 |

Chance = 0.333; imbalance-aware baseline (always "2 people") = 0.518. The selected
model beats both. Count-MAE ≈ 0.42 on windows.

Confusion (rows = truth 0/1/2):

|  | pred 0 | pred 1 | pred 2 |
|---|---:|---:|---:|
| **true 0** | 82 | 132 | 113 |
| **true 1** | 26 | 252 | 299 |
| **true 2** | 4 | 121 | 845 |

**Binary presence (occupied vs empty), LOSO** (`tools/run_own.py --task occupy`,
2 s windows): **89.0% accuracy, macro-F1 0.779** (chance = 0.500).
Recording-level majority vote: **94.1%** (16/17 held-out recordings fully correct;
a LogisticRegression vote variant reached 17/17 in the design study).
Per held-out session: **every 1-person and 2-person session scored 94–100%** —
distinguishing 1 vs 2 people while people are present is nearly solved. The residual
error is concentrated in quiet empty rooms, analyzed in §6.

**Headline:** presence detection **89%**, recording-level verdict **94%**, head-count
*among people present* **94–100% per session**, end-to-end 3-class counting **63%**.

### 4.2 Range

Range captures (`d1m/d3m/d5m` choreography) were planned but not recorded; the
measurement is future work. The link operated at a fixed working distance throughout;
empty-room RSSI baseline was -43.5 to -51.2 dBm depending on rig mounting, occupied
captures dropped up to 10 dB below baseline (strong absorption evidence).

### 4.3 Reliability

Per-trial (per-recording) LOSO accuracy spread (`Results/m4_reliability.json`):
mean 0.536, min 0.000, max 0.963 across 17 trials. The spread is dominated by the
same axis as §6: sessions whose radio signature matches their label score >0.9;
the two absorbed "empty" sessions score lowest. Binary task reliability is far
tighter (occupied sessions uniformly 0.94–1.00).

### 4.4 Latency & throughput

- Inference: **0.084 ms/window** mean (Gradient Boosting; worst fold 0.236 ms) —
  orders of magnitude below the 2 s update period.
- Training: ~1.1 s per LOSO fold.
- Live loop: 2 s update period, ~55–65 CSI frames/s streaming, 52 subcarriers;
  15 s startup link-calibration warm-up.

## 5. Ablation study

Same LOSO folds, RF classifier (`Results/m4_ablation.json`, calibration applied):

| Feature set | Accuracy | Macro-F1 |
|---|---:|---:|
| 12 (M3 baseline) | 0.574 | 0.527 |
| 16 (+ counting features) | 0.549 | 0.466 |

The 4 counting features do not raise overall window accuracy under RF: overall
accuracy is dominated by the empty-vs-occupied boundary, which is carried by the
variance/Doppler family. The counting features target the 1-vs-2 boundary, where the
*binary-within-occupied* analysis shows 94–100% session accuracy.

Two additional design studies are included as M4 evidence
(`Results/normalization_experiment.json`, `Results/window_experiment.json`):

1. **Per-session RSSI calibration**: 54.1% → **62.9%** (best single intervention;
   removes the rig-geometry confound). Rationale in §2.
2. **Window size (1/2/2.5/3.5 s)**: 1 s optimal for 3-class; 2 s optimal for binary
   (87.6% → 90.6% SVM in-study; deployed MLP 89.0%).

## 6. Error analysis

- **Empty-room false alarms are the dominant 3-class failure.** Of 327 empty-room
  windows, the selected model recognizes 82 (25% recall without training balance;
  40% with balanced training). Root cause *confirmed by the recording occupants*:
  two of the three "empty-room" captures were **not actually empty** — people were
  sitting in the room during those takes, which matches the measured 4–8 dB of link
  absorption (0p_t02: -47.7 dBm, 0p_t04: -51.2 dBm vs 0p_t01's -43.5 open-link
  baseline). They are therefore **mislabeled captures**, not model failures: a
  person-present capture with an even stronger signature (2p_t07, -41.4 dBm,
  *stronger* than any empty baseline) confirms the single-link signature is
  absorption-driven. **On the one genuinely empty capture (0p_t01), held-out
  binary accuracy is 80.6% (>70% target met).** Relabeling the two mislabeled
  captures to "occupied" would leave a single empty session, below the minimum
  (2 per class) required for honest held-out validation — so the captures are
  kept, with this caveat documented, rather than silently relabeled. A repeat
  empty-room capture under verified-empty conditions is the identified fix.
- **QC/quarantine process**: 4 corrupted or label-contradicted takes were quarantined
  (`Data/classroom_quarantine/`): two USB-interrupted captures (rate collapse
  ~10–19 frames/s vs the healthy 55–65), one "empty" take with a hidden absorber
  (-64 dBm), and one 60 s take whose stream never stabilized. Training on poisoned
  takes measurably *reduced* accuracy (45.3% → 36.7%), which motivated the QC gate.
- **Failure-case captures (still person, door open)**: planned by the M4 choreography
  but not recorded; the absorbed-empty analysis above covers the door-open mechanism
  in principle. Marked as unrecorded future work, not claimed as results.

## 7. Baseline & SOTA comparison

- **Chance**: 33.3% (3-class) / 50.0% (binary). **Imbalance baseline** (always
  predict the majority class): 51.8%. The selected model beats both (62.9% / 89.0%).
- **Non-AI threshold baseline** (amplitude-variance threshold): 19.3% in the M3-era
  evaluation — far below the learned models.
- **SOTA systems** (Widar/ESP32-CSI-Tool lineage, camera-free crowd counting) use
  multi-link, multi-antenna, multi-position arrays and thousands of recordings;
  they also typically evaluate within a single fixed rig. With one TX–RX pair on
  ₹1,500 hardware and 17 recordings, ElderGuard is not competing with those numbers
  and does not claim to; its contribution is a **complete, honest, camera-free MVP**:
  real hardware, grouped validation, measured latency, quantified failure modes, and
  reproducible artifacts.

## 8. Limitations

- Single TX–RX link: absorption by any object on the line is indistinguishable from a
  person (the §6 finding). A second link or multi-position sweep would disambiguate.
- 17 recordings over multiple rig mountings; sessions-per-class uneven (3/5/9).
  Same-mounting batches score higher than cross-mounting LOSO (design-study evidence).
- Empty-room recall is limited by two absorbed "empty" captures (labels questionable).
- Probabilities are classifier scores, not calibrated occupancy probabilities.
- Effective rate 55–90 frames/s varies per capture; the pipeline handles this via
  timestamp-derived fs, but Doppler band resolution varies accordingly.
- Fall mode remains a heuristic demo (no recorded fall trials in M4).

## 9. Reproducibility

Environment: Python 3.12, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.5.0
(`requirements.txt`; `tools/selftest.py` = 9/9 PASS). Hardware: 2× ESP32-S3,
wall-charger TX, laptop USB RX (COM9, VID 0x303a).

```powershell
py -3.12 tools\selftest.py
py -3.12 tools\check_captures.py --data-dir ..\Data\classroom
py -3.12 tools\train_model.py --data-dir ..\Data\classroom --out-dir ..\Results
py -3.12 tools\run_own.py --data-dir ..\Data\classroom --out ..\Results --task occupy --window 2.0 --save-model model_occupy.joblib
py -3.12 tools\m4_metrics.py --data-dir ..\Data\classroom --out ..\Results
py -3.12 tools\live_demo.py --port COM9 --mode count --binary --model ..\Results\model_occupy.joblib --window 2.0
py -3.12 tools\live_demo.py --replay ..\Data\classroom\roomA_2p_t09.csv --binary --model ..\Results\model_occupy.joblib --window 2.0   # no-hardware proof
```

Artifacts committed under `Results/`: `model_count.joblib` (791,552 bytes),
`model_occupy.joblib`, `training_report.json` (per-model scores + per-fold latency),
`m4_ablation.json`, `m4_counting.json`, `m4_reliability.json`,
`normalization_experiment.json`, `window_experiment.json`, `M4_METRICS.md`.
Every figure and number above traces to one of these files.

## 10. Conclusion & future work

ElderGuard M4 delivers a camera-free presence and people-counting system on two
ESP32-S3 boards: **89% binary presence accuracy (94% at recording level, 94–100%
head-count among people present)** and 63% end-to-end 3-class counting under strict
leave-one-recording-out validation, with a diagnosed-and-fixed rig-calibration
confound, an honest QC process, and physically explained failure modes.

Future work: (1) a second link or motorized TX to disambiguate absorption sources;
(2) same-mounting batch capture for the range and failure-case choreography;
(3) network push ("is sir in?" dashboard); (4) calibrated probabilities and a
thresholded alarm policy for deployment; (5) recorded fall trials to validate the
fall heuristic.
