# Data — ElderGuard M4

## Raw captures

`classroom/` contains the labeled CSI captures used for all M4 results.
Each CSV is one continuous recording from the ESP32-S3 receiver, one 18-column
row per Wi-Fi frame (format documented in `../Code/src/parse_csi.py` and
validated by `../Code/tools/validate_dataset.py`).

The occupancy label (0/1/2 people) is encoded in the filename: `roomX_Np_tNN.csv`.

## Collection protocol (fixed for all captures)

- Room: tinkerlab; boards taped at fixed positions, TX powered from wall charger.
- TX: `ElderGuard-TX` soft-AP, channel 1, ~100 UDP pkt/s (design target;
  measured rate ~62–66 Hz after station power-save).
- RX: ESP32-S3 CSI callback, LLTF+HT-LTF+STBC sections, 52 HT20 subcarriers.
- Choreography per class is scripted and identical across trials
  (see `LAB_DAY_RUNBOOK.md` in the repo root / M3_Group8):
  - 0p: room empty, door closed
  - 1p: one person walks slowly across the link for the whole capture
  - 2p: two people walk slowly, independently, whole capture
- Captures with SUSPECT/REJECT QC verdicts were re-recorded, never relabeled
  (enforced by `../Code/tools/check_captures.py`).

## Range / reliability captures

Additional captures supporting the M4 evaluation are described alongside their
result files in `../Results/`:

- `roomA_*_d1m/d3m/d5m` — link-range variants (RSSI + accuracy vs distance)
