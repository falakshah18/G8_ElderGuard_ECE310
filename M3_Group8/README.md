# ElderGuard M3 — CSI-only people-counting prototype

This project uses two ESP32-S3-N16R8 boards. **It does not use a camera.** Board A transmits Wi-Fi probe packets; Board B receives them, exports CSI by USB serial, and Laptop 1 runs Python preprocessing, training, and inference.

## Fixed two-laptop roles

| Device | Role | Port | What runs there |
|---|---|---|---|
| Laptop 1 (DELL) | Board B / RX | COM15, 921600 baud | Receiver firmware upload/monitor; Python capture, validation, training, live inference |
| Laptop 2 (Admin) | Board A / TX | COM4 for diagnostics | Transmitter firmware upload; keeps the transmitter powered and active |

Do not swap the roles. Close the receiver Serial Monitor on Laptop 1 before Python opens COM15. Laptop 2 may keep its transmitter diagnostic monitor open at 115200 baud; it is not part of the Python data path.

## Raw recordings found

`Data/classroom` contains the three labeled captures: 4,578 empty-room frames, 5,808 one-person frames, and 5,563 two-person frames. It also contains one unlabeled diagnostic recording and two zero-byte CSVs. The raw files are preserved. Full checks are written to `Results/dataset_validation/`.

The CSI field is a whitespace-separated signed-byte vector encoded as interleaved imaginary/real components. The receiver enables LLTF, HT-LTF, and STBC HT-LTF2, producing one, two, or three 64-complex-sample LTF sections (128, 256, or 384 bytes). The current parser consistently uses the first LLTF section and rejects malformed odd-length vectors; it does not pad missing CSI. ESP-IDF documents `rx_ctrl.timestamp` in microseconds. The parser estimates sample rate from robust positive timestamp deltas, not from an assumed filename rate.

## Laptop 1 setup and execution

Open this folder in VS Code and use its PowerShell terminal:

```powershell
cd "C:\Users\DELL\Downloads\ElderGuard_M3_WiVi32_adapted\ElderGuard_M3"
py -3.11 -m pip install -r requirements.txt
```

### Validate raw captures

```powershell
py -3.11 tools\validate_dataset.py
```

The tool writes JSON, Markdown, and a plot under `Results\dataset_validation`. It does not modify captures.

### Train a model artifact

```powershell
py -3.11 tools\train_model.py --data-dir Data\classroom --out-dir Results
```

This writes `model_count.joblib`, `model_metadata.json`, `feature_config.json`, and `training_report.json`. The model is **provisional** because there is only one recording for each count. That dataset cannot support a session-held-out test with all three classes represented in training, so the script deliberately does not report a validation accuracy.

### Optional CSI feature export

```powershell
py -3.11 tools\preprocess_csi.py Data\classroom\roomA_1p_t01.csv --out Results\roomA_1p_features.csv
```

This is CSI-only and diagnostic; it is not a separate model or camera workflow.

### Live inference from Board B

With Board A powered on, Board B connected to Laptop 1, and its Serial Monitor closed:

```powershell
py -3.11 tools\live_demo.py --port COM15 --baud 921600 --mode count --model Results\model_count.joblib
```

It reads real CSI rows, uses the shared CSI parser/preprocessor/features, reports row rate/RSSI/model scores, and smooths predictions. The displayed model scores are uncalibrated. Press Ctrl+C to stop.

## Laptop 2 transmitter setup

Install Arduino-ESP32 core 3.3.12 as already used for these boards. Open `firmware\tx_esp32_ap\tx_esp32_ap.ino`, select **ESP32S3 Dev Module** and COM4, then compile/upload only if the existing transmitter firmware is absent or failing. It creates `ElderGuard-TX` with password `elderguard` on channel 1. Keep Board A powered while Laptop 1 collects CSI. No Python, model, or camera process runs on Laptop 2.

## Evaluation limits and next data needed

Offline replay of the three training recordings is a software-path smoke check, not independent validation. It showed unstable window-level predictions on the one-person and two-person recordings. Do not quote those replays as accuracy. Collect at least 3–5 independent sessions per class (more is better), with varied positions and still/moving subjects, then rerun validation and training. Preserve the same room and board placement for the first evaluation. Current capability is limited to labels 0, 1, and 2; no higher-count support is claimed.

The optional Streamlit dashboard is deferred until fresh-session model validation is available; a dashboard cannot remedy insufficient or confounded training data.
