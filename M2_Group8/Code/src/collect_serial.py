#!/usr/bin/env python3
"""
ElderGuard - serial CSI logger (M2)
ECE 310 Group 8

Logs ESP32-CSI-Tool CSV rows coming off the receiver board over USB serial to
a timestamped CSV file for offline processing.

    python collect_serial.py --port /dev/ttyUSB0 --baud 921600 \
        --out ../../Data/captures/roomA_1p_session01.csv

Press Ctrl-C to stop; the file is flushed and closed cleanly.
"""

import argparse
import time

import serial  # pyserial


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Seconds to capture (0 = until Ctrl-C)")
    args = ap.parse_args()

    print(f"Opening {args.port} @ {args.baud} ...")
    with serial.Serial(args.port, args.baud, timeout=1) as ser, \
            open(args.out, "w", newline="") as fh:
        header_written = False
        start = time.time()
        n = 0
        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("type,"):
                    if not header_written:
                        fh.write(line + "\n")
                        header_written = True
                    continue
                if header_written:
                    fh.write(line + "\n")
                    n += 1
                if n % 1000 == 0 and n:
                    print(f"\r{n} CSI frames logged", end="", flush=True)
                if args.duration and (time.time() - start) > args.duration:
                    break
        except KeyboardInterrupt:
            print("\nStopped by user.")
    print(f"\nWrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()
