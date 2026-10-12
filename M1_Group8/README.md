# M1_Group8 — ElderGuard: Wi-Fi CSI Sensing for Fall Detection

ECE 310 (Wireless Communications), Monsoon Semester 2026 — Group 8

This folder contains our Milestone 1 submission for **ElderGuard**, a low-cost
Wi-Fi CSI sensing system for fall detection in elderly care.

## What's in here

- `Report/M1_Group8_Report.docx` — the M1 report: problem statement, target user, product concept, value
  proposition, functional/non-functional requirements, SOTA anchor, wireless
  concepts mapping, evaluation metrics, technical progress, and timeline
  (with the official M1–M4 dates for the MVP category).
- `Video/` — link to the recorded milestone walkthrough.
- `Code/src/` — first working version of the CSI signal-processing pipeline
  (phase sanitization, bandpass filtering, Doppler spectrogram) plus a dataset
  loader stub. See `Code/src/README.md` for how to run it.
- `Data/` — no raw data is checked in here; see `Data/README.md` for the
  public datasets we're using and where to get them.
- `Results/` — output plot from the pipeline demo (`synthetic_fall_spectrogram.png`).

## Project summary

ElderGuard detects falls using ordinary Wi-Fi Channel State Information (CSI)
instead of a wearable or a camera. Two ESP32-S3 boards act as a low-cost
transmitter/receiver pair; the CSI they capture is cleaned up, filtered to the
0.5–10 Hz band where human movement lives, and converted into a Doppler
spectrogram. A 1D-CNN classifies the result as a fall or normal activity,
alongside a live confidence score based on measured SNR and packet delivery
rate. We are building and validating the pipeline on a public CSI dataset
first (Widar 3.0 / UT-HAR), then extending it to our own hardware.

## Status at M1

- Related-work survey and SOTA position: done (see report).
- Signal-processing pipeline (phase sanitization, bandpass filter, STFT
  spectrogram): implemented and validated on synthetic data.
- Public dataset loader: interface defined, parsing in progress for M2.
- ESP32-S3 hardware bring-up: planned for M3.
