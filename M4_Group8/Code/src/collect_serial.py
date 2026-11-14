


import argparse
import csv
import os
import time

import serial
from eg_port import resolve_port

RECONNECT_BUDGET = 30.0
MAX_RECONNECTS = 8


def open_port(port, baud):
    ser = serial.Serial(port, baud, timeout=1)


    ser.setDTR(False)
    ser.setRTS(False)

    time.sleep(0.2)
    ser.reset_input_buffer()

    return ser


def reconnect(port, baud):
    deadline = time.time() + RECONNECT_BUDGET

    while time.time() < deadline:
        time.sleep(1)

        try:
            return open_port(port, baud)
        except (serial.SerialException, OSError):
            pass

    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=0)

    args = ap.parse_args()
    args.port = resolve_port(args.port)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Opening {args.port} @ {args.baud} ...")

    try:
        ser = open_port(args.port, args.baud)
    except (serial.SerialException, OSError) as exc:
        raise SystemExit(
            f"Cannot open {args.port}: {exc}\n"
            "Close Arduino Serial Monitor and check the USB connection."
        )

    print("Connected. Waiting for firmware CSV header...")

    header_written = False
    frame_count = 0
    reconnect_count = 0
    gaps = []

    start = time.time()
    last_data = start

    try:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:

            while True:

                if args.duration > 0:
                    if time.time() - start >= args.duration:
                        break

                try:
                    line = (
                        ser.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )

                except (serial.SerialException, OSError) as exc:

                    if reconnect_count >= MAX_RECONNECTS:
                        print(f"\nUSB keeps dropping: {exc!r}")
                        break

                    print(
                        f"\nUSB disconnected. Reconnecting to {args.port}...",
                        flush=True
                    )

                    try:
                        ser.close()
                    except Exception:
                        pass

                    ser = reconnect(args.port, args.baud)

                    if ser is None:
                        print("Could not reconnect. Stopping.")
                        break

                    reconnect_count += 1
                    gaps.append(time.time() - last_data)

                    print("Reconnected. Waiting for CSV header again.")


                    continue

                if not line:
                    continue


                if line.startswith("type,"):

                    if not header_written:
                        header = next(csv.reader([line]))

                        fh.write(",".join(header) + "\n")
                        fh.flush()

                        header_written = True
                        print("CSV header detected.")

                    continue


                if line.startswith("CSI,"):

                    if not header_written:
                        print(
                            "CSI row received before header; skipping."
                        )
                        continue


                    fh.write(line + "\n")
                    frame_count += 1
                    last_data = time.time()

                    if frame_count % 100 == 0:
                        fh.flush()

                    if frame_count % 100 == 0:
                        elapsed = time.time() - start
                        rate = frame_count / max(elapsed, 0.001)

                        print(
                            f"\rCSI frames: {frame_count} | "
                            f"Average rate: {rate:.1f}/s",
                            end="",
                            flush=True
                        )

                    continue


                if line.startswith("#") or "CSI enable" in line:
                    print(f"\nESP32: {line}")

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    finally:
        try:
            ser.close()
        except Exception:
            pass

    elapsed = time.time() - start

    print("\n\n========== CAPTURE SUMMARY ==========")
    print(f"File          : {args.out}")
    print(f"CSI frames    : {frame_count}")
    print(f"Duration      : {elapsed:.1f} seconds")
    print(
        f"Average rate  : "
        f"{frame_count / max(elapsed, 0.001):.2f} frames/sec"
    )
    print(f"Header found  : {header_written}")
    print(f"Reconnects    : {reconnect_count}")

    if gaps:
        print(f"USB gaps      : {len(gaps)}")
        print("This recording may not be continuous. Re-record it.")

    if frame_count == 0:
        print("\nWARNING: No CSI frames captured.")
        print("Check the RX firmware and incoming CSI serial stream.")


if __name__ == "__main__":
    main()
