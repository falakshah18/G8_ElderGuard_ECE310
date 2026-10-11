#!/usr/bin/env python3
"""
ElderGuard - UDP traffic source / fallback transmitter (M2)
ECE 310 Group 8

When the ESP8266 transmitter is unavailable, a laptop on the same network can
flood UDP packets at the ESP32 receiver so the CSI capture path can still be
exercised. The ESP32 receiver captures CSI from any 802.11 frame it receives,
so the packet payload only needs to be non-empty and sent at a steady rate.

    python flood_udp.py --target 192.168.4.1 --port 5000 --rate 100

`--rate` is packets per second; `--duration 0` runs until Ctrl-C.
"""

import argparse
import socket
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, help="Receiver IP address")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--rate", type=float, default=100.0, help="packets/second")
    ap.add_argument("--size", type=int, default=64, help="payload bytes")
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()

    period = 1.0 / args.rate if args.rate > 0 else 0.0
    payload = b"ELDERGUARD" * (args.size // 10 + 1)
    payload = payload[: args.size]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Flooding {args.target}:{args.port} at {args.rate:.0f} pkt/s "
          f"({args.size} B payload)")
    start = time.time()
    n = 0
    try:
        while True:
            sock.sendto(payload, (args.target, args.port))
            n += 1
            if n % 500 == 0:
                print(f"\r{n} packets sent", end="", flush=True)
            if args.duration and (time.time() - start) > args.duration:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()
    print(f"\nSent {n} packets in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
