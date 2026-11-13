"""Find the ElderGuard receiver port without guessing COM numbers.

Every laptop gives the same board a different number (COM3, COM8, COM14...),
so `--port auto` looks the board up by its USB vendor id instead:

    0x303A  Espressif              -> the socket labelled USB  (what we use)
    0x1A86  QinHeng CH340          -> the socket labelled COM
    0x10C4  Silicon Labs CP210x    -> the socket labelled COM

`--port COM8` still works exactly as before; only the word "auto" changes
behaviour.
"""

import sys

from serial.tools import list_ports

VENDORS = {0x303A: "Espressif ESP32-S3 (the socket labelled USB)",
           0x1A86: "QinHeng CH340 (the socket labelled COM)",
           0x10C4: "Silicon Labs CP210x (the socket labelled COM)"}


def candidates():
    return [p for p in list_ports.comports() if p.vid in VENDORS]


def _fmt(p):
    return f"    {p.device}  -  {p.description}"


def resolve_port(spec):
    """Return a port name for open(). Aborts with a readable list if unsure."""
    if spec != "auto":
        return spec
    found = candidates()
    if len(found) == 1:
        print(f"auto-detected the receiver on {found[0].device} "
              f"({VENDORS[found[0].vid]})")
        return found[0].device
    print("Ports that look like an ESP32 board:")
    for p in found:
        print(_fmt(p))
    if not found:
        print("  (none)")
        print("All ports this laptop can see right now:")
        for p in sorted(list_ports.comports(), key=lambda x: x.device):
            print(_fmt(p))
        sys.exit("! 'auto' found no ESP32 board.\n"
                 "  1. Is the receiver board plugged into the socket labelled "
                 "USB?\n"
                 "  2. Is that cable connected to THIS laptop?\n"
                 "  3. Then run it again. If it still says none, send the "
                 "screen above to your teammate - it tells us what your "
                 "laptop can see.")
    sys.exit(f"! 'auto' found {len(found)} possible boards and cannot choose. "
             "Pick one device name from the list above and use it instead of "
             "auto, e.g. --port COM5")
