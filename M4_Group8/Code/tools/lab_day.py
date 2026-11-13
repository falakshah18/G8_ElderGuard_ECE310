#!/usr/bin/env python3
"""ElderGuard LAB-DAY DRIVER - the only thing you need to run tomorrow.

    python tools\\lab_day.py

It guides the whole session: sanity check, every recording (with choreography
instructions and auto-QC after each), then all analysis, then a final
checklist. You only press ENTER when the room is ready and answer y/n.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))   # M4_Group8
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
DATA = os.path.normpath(os.path.join(ROOT, "Data", "classroom"))
RESULTS = os.path.normpath(os.path.join(ROOT, "Results"))
PY = sys.executable

sys.path.insert(0, SRC)

# capture, label, seconds, choreography shown to the operator
CAPTURES = [
    ("roomA_0p_t02.csv",      0, 90, "EMPTY room. Everyone out, door CLOSED. Stand still near the door."),
    ("roomA_1p_t02.csv",      1, 90, "ONE person walks slowly back and forth across the link. WHOLE time, never stops."),
    ("roomA_2p_t02.csv",      2, 90, "TWO people BOTH walk slowly, independently (different speeds, different paths). WHOLE time."),
    ("roomA_0p_t03.csv",      0, 90, "EMPTY room again. Door closed. Nobody touches anything."),
    ("roomA_1p_t03.csv",      1, 90, "ONE person walks slowly, same style as before. WHOLE time."),
    ("roomA_2p_t03.csv",      2, 90, "TWO people walk slowly and independently, same style as before. WHOLE time."),
    ("roomA_1p_still_t01.csv",1, 90, "ONE person sits/stands PERFECTLY STILL (like working at a desk). Minimal movement. This validates breathing detection!"),
    ("roomA_1p_d1m_t01.csv",  1, 60, "Boards now ~1 m APART. ONE person walks slowly. WHOLE time."),
    ("roomA_1p_d3m_t01.csv",  1, 60, "Boards now ~3 m APART. Same walk. WHOLE time. NOTE THE RSSI ON SCREEN."),
    ("roomA_1p_d5m_t01.csv",  1, 60, "Boards now ~5 m APART (or until stream weakens). Same walk. WHOLE time. NOTE THE RSSI."),
    ("roomA_0p_door_t01.csv", 0, 60, "EMPTY room with the DOOR OPEN. Occasional hallway movement is fine. Nobody inside."),
]

RESULTS_NOTE = """
SCREENSHOTS TO TAKE (phone, landscape):
  1. The LOSO score table that train_model.py printed
  2. The M4_METRICS.md summary + ablation lines
  3. The presence table for roomA_1p_still_t01.csv (breathing validation!)
  4. The RSSI values for d1m / d3m / d5m
VIDEOS TO RECORD (after [5]):
  a. live_demo --binary: person walks in -> OCCUPIED, out -> EMPTY
  b. presence_demo live or the still-person replay table
  c. 20 s of the raw console stream
  d. the training score table scrolling
FINALLY: copy the whole M4_Group8 folder to a pen drive AND cloud.
"""


def sh(args_list, **kw):
    return subprocess.run(args_list, **kw)


def resolve_port_interactive():
    from eg_port import resolve_port
    while True:
        try:
            return resolve_port("auto")
        except SystemExit:
            print("No board found. Plug Board B (receiver) into this laptop.")
            time.sleep(2)


def qc_file(path):
    """Run the QC tool, return the verdict for this one file."""
    r = sh([PY, os.path.join(HERE, "check_captures.py"),
            "--data-dir", os.path.dirname(path)],
           capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    name = os.path.basename(path)
    for line in out.splitlines():
        if line.startswith(name):
            return line.split()[-1], line
    return "SKIP", out.strip().splitlines()[-1] if out.strip() else "(no output)"


def sanity(port):
    import serial
    try:
        ser = serial.Serial(port, 921600, timeout=0.5)
    except Exception as exc:
        print(f"!! cannot open {port}: {exc}")
        return False
    ser.setDTR(False); ser.setRTS(False)
    print(f"Listening on {port} for 15 s ...")
    t0, n, rssi = time.time(), 0, None
    while time.time() - t0 < 15:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.startswith("CSI,"):
            n += 1
            parts = line.split(",", 5)
            if len(parts) > 4:
                rssi = parts[3]
    ser.close()
    dt = max(time.time() - t0, 1e-3)
    rate = n / dt
    print(f"CSI rate = {rate:.0f}/s   RSSI = {rssi}")
    if rate < 25:
        print("!! stream too slow (<25/s). Check: Board A powered? Serial monitor closed? Replug Board B.")
        return False
    print(">> Stream is healthy. Safe to record.")
    return True


def record(port, name, label, seconds, choreo):
    path = os.path.join(DATA, name)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        a = input(f"  {name} already exists. Re-record? (y/N): ").strip().lower()
        if a != "y":
            print("  skipped.")
            return True
    print(f"\n=== {name} ({seconds} s) ===")
    print(f"  CHOREOGRAPHY: {choreo}")
    input("  Get ready... press ENTER to START recording >> ")
    print(f"  recording {seconds} s ... do NOT touch the boards.")
    r = sh([PY, os.path.join(HERE, "collect_serial.py"),
            "--port", port, "--baud", "921600",
            "--out", path, "--duration", str(seconds)])
    if r.returncode != 0:
        print("  !! recorder failed. Replug Board B, then choose [3] to retry.")
        return False
    verdict, line = qc_file(path)
    print(f"  QC verdict: {line}")
    if verdict != "OK":
        a = input("  NOT OK. Re-record this capture? (Y/n): ").strip().lower()
        if a != "n":
            return record(port, name, label, seconds, choreo)
        print("  kept anyway - remember this capture is suspect.")
    return True


def analysis():
    print("\n=== [5] generating ALL results (this takes a few minutes) ===")
    steps = [
        ("validate_dataset.py", ["--data-dir", DATA, "--out-dir", RESULTS]),
        ("train_model.py", ["--data-dir", DATA, "--out-dir", RESULTS]),
        ("run_own.py", ["--task", "occupy", "--data-dir", DATA, "--out", RESULTS,
                        "--save-model", os.path.join(RESULTS, "model_occupy.joblib")]),
        ("m4_metrics.py", ["--data-dir", DATA, "--out", RESULTS]),
        ("make_figures.py", ["--metrics", os.path.join(RESULTS, "metrics_occupy.json"),
                             "--out", os.path.join(RESULTS, "occupy")]),
    ]
    for script, extra in steps:
        print(f"\n>>> {script} ...")
        r = sh([PY, os.path.join(HERE, script)] + extra)
        if r.returncode != 0:
            print(f"!! {script} failed - send me its output. Continuing with the rest.")
    still = os.path.join(DATA, "roomA_1p_still_t01.csv")
    if os.path.exists(still):
        print("\n>>> presence validation on the STILL capture (breathing layer):")
        sh([PY, os.path.join(HERE, "presence_demo.py"), "--replay", still])
    print(RESULTS_NOTE)


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    port = None
    while True:
        print("\n" + "=" * 66)
        print("ELDERGUARD LAB DAY  -  what do you want to do?")
        print("  [1] Sanity-check the receiver stream (do this FIRST)")
        print("  [2] Guided recording session (all 11 captures, step by step)")
        print("  [3] Re-record ONE capture")
        print("  [4] QC all captures")
        print("  [5] Generate ALL results (validate + train + metrics + presence)")
        print("  [6] Show the final checklist")
        print("  [0] Quit")
        choice = input("choice: ").strip()
        if choice == "1":
            port = resolve_port_interactive()
            sanity(port)
        elif choice == "2":
            port = port or resolve_port_interactive()
            if not sanity(port):
                continue
            pending = [c for c in CAPTURES
                       if not (os.path.exists(os.path.join(DATA, c[0]))
                               and os.path.getsize(os.path.join(DATA, c[0])) > 100_000)]
            if not pending:
                print("All 11 captures already exist. Use [3] to redo one, or [5].")
                continue
            print(f"{len(pending)} capture(s) to record. Boards are taped? Good.")
            for name, label, secs, choreo in pending:
                ok = record(port, name, label, secs, choreo)
                if not ok:
                    break
            else:
                print("\nAll captures recorded. Now run [4] then [5].")
        elif choice == "3":
            port = port or resolve_port_interactive()
            for i, (name, label, secs, choreo) in enumerate(CAPTURES, 1):
                print(f"  [{i}] {name}")
            k = int(input("which one (number): ").strip())
            record(port, *CAPTURES[k - 1])
        elif choice == "4":
            r = sh([PY, os.path.join(HERE, "check_captures.py"),
                    "--data-dir", DATA])
            print("Every line must end in OK. If not: use [3] to re-record it.")
        elif choice == "5":
            analysis()
        elif choice == "6":
            print(RESULTS_NOTE)
        elif choice == "0":
            return
        else:
            print("choose 0-6")


if __name__ == "__main__":
    main()
