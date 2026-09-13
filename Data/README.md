# Data

No raw CSI data is checked into this repository (too large, and mostly not
ours to redistribute). We are using a two-stage data plan, as in the proposal:

## Stage 1 — public dataset (in progress for M2)

Shortlisted:
- **Widar 3.0** (Tsinghua University) — multi-person, multi-room, multi-activity
  Wi-Fi CSI dataset, including fall-like and everyday activities.
- **UT-HAR** — smaller CSI-based human activity recognition dataset, useful as
  a faster first target while the full pipeline is being validated.

Once we finalize which one (or both) we use, download instructions and the
exact directory layout expected by `Code/src/load_dataset.py` will be added
here.

## Stage 2 — our own hardware capture (planned for M3)

Two ESP32-S3 boards (transmitter + receiver) capturing CSI (I/Q across 52
subcarriers, RSSI, timestamp) at roughly 100 Hz, using the ESP32 CSI Tool /
Espressif CSI stack. Collected data and a description of the collection
protocol (rooms, activities, number of trials) will be added here once
hardware bring-up is complete.
