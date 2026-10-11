#!/usr/bin/env python3
"""
ElderGuard - serial port helper (M2)
ECE 310 Group 8

Lists the serial ports currently available so you can pick the ESP32 receiver
port before running collect_serial.py.

    python list_ports.py
"""

from serial.tools import list_ports


def main():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Is the board plugged in and are the "
              "USB-serial drivers installed?")
        return
    print(f"{len(ports)} serial port(s):")
    for p in ports:
        print(f"  {p.device:20s}  {p.description}")
        if p.hwid:
            print(f"  {'':20s}  {p.hwid}")


if __name__ == "__main__":
    main()
