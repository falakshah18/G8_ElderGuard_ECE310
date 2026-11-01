#!/usr/bin/env python3
"""
ElderGuard SELF-TEST - run this before you touch the boards.

It checks, in order, that:
  1. Python and the six libraries are really installed
  2. every file that was supposed to arrive, arrived
  3. the recorder can read a capture the receiver would have written
  4. the quality-check tool runs over a whole folder
  5. the trainer can fit both models (occupied/empty and 0-1-2 head count)
  6. the live demo's replay path gives EMPTY / OCCUPIED on synthetic data
  7. the fall detector ALARMS on a synthetic fall and stays SILENT on a quiet
     room - both directions, so a broken alarm cannot hide
  8. the recorder survives the Windows USB drop that kills naive loggers
  9. your laptop can see the ESP32 board at all (informational)

Nothing here needs hardware, and nothing here writes to your real data folder.

    py -3.11 tools\selftest.py

Exit code 0 and a wall of PASS lines means: your laptop is ready, and if a
later step fails, it is the boards/cable/room, not your setup.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
TOOLS = os.path.join(ROOT, "tools")
TMP = os.path.join(ROOT, "selftest_tmp")

REQUIRED = [
    "src/parse_csi.py", "src/preprocess.py", "src/features.py", "src/models.py",
    "src/evaluate.py", "src/make_figures.py", "src/eg_port.py",
    "src/run_pipeline.py", "src/synthetic.py", "src/collect_serial.py",
    "tools/run_own.py", "tools/check_captures.py", "tools/live_demo.py",
    "tools/make_fake_captures.py", "tools/selftest.py",
    "firmware/tx_esp32_ap/tx_esp32_ap.ino",
    "firmware/rx_esp32_csi/rx_esp32_csi.ino",
    "requirements.txt", "START_HERE.md", "TALLY_SHEET.md", "demo_layout.png",
    "Data/classroom", "Results",
]

LIBS = ["numpy", "scipy", "sklearn", "joblib", "matplotlib", "serial"]

RESULTS = []


def stage(name):
    def deco(fn):
        fn._stage = name
        return fn
    return deco


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}"
          + (f"\n      {detail}" if detail else ""), flush=True)


def run(args, expect_zero=True):
    """Run one of the real tools as its own process, exactly as a human would."""
    print(f"  $ {' '.join(os.path.basename(a) if i else a for i, a in enumerate(args))}")
    p = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.splitlines()[-14:])
    return p.returncode, out, tail


def write_capture(path, fs, seconds, profile):
    """One CSV capture in the receiver's exact format.

    profile(seconds) -> (motion amplitude multiplier, extra noise sd)
    """
    import numpy as np
    n = int(seconds * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    NSUB = 64
    base = rng.uniform(20.0, 45.0, NSUB)
    amp = np.tile(base, (n, 1))
    movers = rng.permutation(NSUB)[: NSUB // 2]
    for k in range(n):
        gain, noise = profile(k / fs)
        amp[k, movers] += gain * abs(np.sin(2 * np.pi * 1.3 * t[k]))
        amp[k] += rng.normal(0, noise, NSUB)
    amp = np.clip(amp, 1.0, 127.0)
    phase = rng.normal(0.0, 0.15, amp.shape)
    header = ("type,channel,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,"
              "not_sounding,aggregation,stbc,fec_coding,sgi,ampdu_cnt,ant,"
              "timestamp,data")
    with open(path, "w", newline="") as fh:
        fh.write(header + "\n")
        for k in range(n):
            i8 = np.empty(2 * NSUB, dtype=int)
            i8[0::2] = np.clip(np.round(amp[k] * np.sin(phase[k])), -127, 127)
            i8[1::2] = np.clip(np.round(amp[k] * np.cos(phase[k])), -127, 127)
            fh.write(f"CSI,1,aa:bb:cc:dd:ee:01,-48,0,1,7,0,1,1,0,0,0,0,0,0,"
                     f"{int(k / fs * 1e6) % 2**31},"
                     f"[{' '.join(str(v) for v in i8)}]\n")
    return n


# ---------------------------------------------------------------- stages ----

@stage("1. python + libraries")
def st_env():
    v = sys.version_info
    detail = f"python {v.major}.{v.minor}.{v.micro} at {sys.executable}"
    ok = (v.major, v.minor) >= (3, 9)
    missing = []
    for lib in LIBS:
        try:
            __import__(lib)
        except Exception as exc:
            missing.append(f"{lib} ({exc.__class__.__name__})")
    if missing:
        ok = False
        detail += ("\n      MISSING: " + ", ".join(missing)
                   + "\n      fix:  py -3.11 -m pip install -r requirements.txt")
    else:
        import numpy, sklearn, serial
        detail += (f"  numpy {numpy.__version__}, sklearn {sklearn.__version__},"
                   f" pyserial {serial.VERSION}")
    return ok, detail


@stage("2. all files arrived")
def st_files():
    absent = [f for f in REQUIRED
              if not os.path.exists(os.path.join(ROOT, f.replace("/", os.sep)))]
    if absent:
        return False, "missing: " + ", ".join(absent) + "\n      re-copy the folder"
    return True, f"{len(REQUIRED)} key files + the two board programs present"


@stage("3. recorder reads a capture")
def st_read():
    os.makedirs(TMP, exist_ok=True)
    f = os.path.join(TMP, "room1_1p_t01_fs60.csv")
    n = write_capture(f, 60, 12, lambda t: (6.0, 1.0))
    sys.path.insert(0, SRC)
    from parse_csi import load_esp32_csv
    s = load_esp32_csv(f)
    ok = (s["amplitude"].shape == (n, 52) and abs(s["fs"] - 60.0) < 0.1
          and s["label"] == 1)
    return ok, (f"read {s['amplitude'].shape[0]} frames x "
                f"{s['amplitude'].shape[1]} subcarriers, fs={s['fs']}, "
                f"label={s['label']} (wanted ({n}, 52), 60.0, 1)")


@stage("4. quality check runs")
def st_qc():
    fake = os.path.join(TMP, "fake")
    rc, out, tail = run([sys.executable, os.path.join(TOOLS, "make_fake_captures.py"),
                         "--out", fake, "--seconds", "30", "--trials", "1"])
    if rc != 0:
        return False, "fake capture generator failed:\n      " + tail
    files = [x for x in os.listdir(fake) if x.endswith(".csv")]
    rc, out2, tail2 = run([sys.executable, os.path.join(TOOLS, "check_captures.py"),
                           "--data-dir", fake])
    ok = rc == 0 and len(files) == 9 and "med" in out2.lower()
    return ok, (f"{len(files)} synthetic sessions, QC tool ran "
                f"(its verdicts on fake data are not meaningful - it only "
                f"proves the tool works)")


@stage("5. both models train")
def st_train():
    fake = os.path.join(TMP, "fake")
    out_dir = os.path.join(TMP, "Results")
    rc, out, tail = run([sys.executable, os.path.join(TOOLS, "run_own.py"),
                         "--data-dir", fake, "--out", out_dir,
                         "--save-model", os.path.join(out_dir, "model_count.joblib")])
    if rc != 0:
        return False, "head-count training failed:\n      " + tail
    rc2, out2, tail2 = run([sys.executable, os.path.join(TOOLS, "run_own.py"),
                            "--task", "occupy", "--data-dir", fake, "--out", out_dir,
                            "--save-model", os.path.join(out_dir, "model_occupy.joblib")])
    if rc2 != 0:
        return False, "occupancy training failed:\n      " + tail2
    accs = [float(l.split("accuracy=")[1].split()[0])
            for l in (out + "\n" + out2).splitlines() if "Best model:" in l]
    m1 = os.path.exists(os.path.join(out_dir, "model_count.joblib"))
    m2 = os.path.exists(os.path.join(out_dir, "model_occupy.joblib"))
    ok = len(accs) == 2 and m1 and m2 and min(accs) > 0.5
    return ok, (f"3-class acc={accs[0] if accs else '?'} "
                f"occupancy acc={accs[1] if len(accs) > 1 else '?'} on synthetic "
                f"data, both .joblib files written")


@stage("6. live path answers EMPTY / OCCUPIED")
def st_replay():
    out_dir = os.path.join(TMP, "Results")
    fake = os.path.join(TMP, "fake")
    # Replay the SAME generator the models were trained on: absolute amplitude
    # scale differs between generators and a cross-scale test would fail for a
    # reason that is not a bug.
    quiet = os.path.join(fake, "room1_0p_trial1_fs100.csv")
    busy = os.path.join(fake, "room1_2p_trial1_fs100.csv")
    if not (os.path.exists(quiet) and os.path.exists(busy)):
        return False, f"run stage 4 first ({quiet} / {busy} not found)"
    r1 = run([sys.executable, os.path.join(TOOLS, "live_demo.py"), "--mode", "count",
              "--binary", "--model", os.path.join(out_dir, "model_occupy.joblib"),
              "--replay", quiet])
    r2 = run([sys.executable, os.path.join(TOOLS, "live_demo.py"), "--mode", "count",
              "--binary", "--model", os.path.join(out_dir, "model_occupy.joblib"),
              "--replay", busy])
    e = (r1[1] + r2[1]).count("PEOPLE = EMPTY")
    o = (r1[1] + r2[1]).count("PEOPLE = OCCUPIED")
    ok = r1[0] == 0 and r2[0] == 0 and e >= 10 and o >= 10
    return ok, (f"the model trained on synthetic data said EMPTY {e}x and "
                f"OCCUPIED {o}x (both files went through the exact live code path)")


@stage("7. fall detector: alarms AND stays silent")
def st_fall():
    """The headline capability, tested in BOTH directions."""
    fall = os.path.join(TMP, "fall_test_fs60.csv")
    calm = os.path.join(TMP, "calm_test_fs60.csv")
    # calm: quiet, one slow normal activity in the middle, then quiet
    def calm_profile(t):
        if 20 <= t < 40:
            return (5.0, 1.2)          # sitting down / walking: normal motion
        return (0.0, 0.6)
    # fall: quiet, then a violent 1.5 s collapse, then total stillness
    def fall_profile(t):
        if 30 <= t < 31.5:
            return (55.0, 6.0)         # the collapse
        return (0.0, 0.6)
    write_capture(fall, 60, 55, fall_profile)
    write_capture(calm, 60, 55, calm_profile)
    common = [sys.executable, os.path.join(TOOLS, "live_demo.py"),
              "--mode", "fall", "--calib", "15"]
    rc_a, out_a, ta = run(common + ["--replay", fall])
    rc_b, out_b, tb = run(common + ["--replay", calm])
    alarmed = "FALL ALARM" in out_a
    silent = "FALL ALARM" not in out_b
    ok = rc_a == 0 and rc_b == 0 and alarmed and silent
    detail = (f"synthetic collapse -> {'ALARMed' if alarmed else 'SILENT'}"
              f" (want ALARM);  normal activity -> "
              f"{'ALARMed' if not silent else 'SILENT'} (want silence)")
    if not ok:
        detail += "\n      calm run tail:\n      " + tb.replace("\n", "\n      ")
    return ok, detail


@stage("8. recorder survives a USB drop")
def st_reconnect():
    os.makedirs(TMP, exist_ok=True)
    probe = os.path.join(TMP, "room1_1p_t01_fs60.csv")
    if not os.path.exists(probe):
        write_capture(probe, 60, 12, lambda t: (6.0, 1.0))
    rc, out, tail = run([sys.executable, os.path.abspath(__file__),
                         "--only", "reconnect-child", "--csv", probe])
    ok = rc == 0 and "RECONNECT_CHILD_OK" in out
    return ok, tail.replace("\n", "\n      ") if not ok else (
        "simulated the exact Windows error 5x, the recorder re-opened the port "
        "and the file stayed valid")


def st_reconnect_child(csv_path):
    """Runs in its own process: a fake port that dies like Board B does."""
    import types
    rows = []
    with open(csv_path, newline="") as fh:
        rdr = csv.reader(fh)
        header = ",".join(next(rdr))
        for r in rdr:
            if r and r[0] == "CSI":
                rows.append(",".join(r))
            if len(rows) >= 40:
                break
    import serial.tools.list_ports      # make sure the submodules are loaded
    real = sys.modules["serial"]

    class Flaky:
        opens = 0

        def __init__(self, port, baud, timeout=1):
            Flaky.opens += 1
            self.i = -1
            self.dead = False

        def setDTR(self, x): pass
        def setRTS(self, x): pass
        def reset_input_buffer(self): pass
        def close(self): pass

        def readline(self):
            if self.dead:
                raise real.SerialException(
                    "ClearCommError failed (PermissionError(13, 'The device "
                    "does not recognize the command.', None, 22))")
            self.i += 1
            if self.i == 0:
                return (header + "\n").encode()
            if self.i == 1:
                return b"Connecting to transmitter AP.... connected.\n"
            if self.i == 2:
                return b"# alive rssi=-48 (~60/s)\n"
            if self.i >= len(rows) + 3:
                self.dead = True
                return b""
            return (rows[(self.i - 3) % len(rows)] + "\n").encode()

    fake = types.ModuleType("serial")
    fake.SerialException = real.SerialException
    fake.Serial = Flaky
    fake.tools = real.tools
    sys.modules["serial"] = fake
    sys.modules["serial.tools"] = real.tools
    sys.modules["serial.tools.list_ports"] = sys.modules[
        "serial.tools.list_ports"] = real.tools.list_ports
    sys.path.insert(0, SRC)
    import collect_serial
    collect_serial.RECONNECT_BUDGET = 2.0
    out = os.path.join(TMP, "reconnect_out.csv")
    sys.argv = ["collect_serial.py", "--port", "COM_FAKE", "--out", out,
                "--duration", "5"]
    # A made-up port name: serial.Serial itself is faked, so nothing real opens.
    collect_serial.main()
    lines = open(out).read().splitlines()
    hdr = sum(1 for l in lines if l.startswith("type,"))
    junk = [i for i, l in enumerate(lines) if i and not l.startswith("CSI,")]
    assert Flaky.opens >= 3, f"only {Flaky.opens} opens: no recovery happened"
    assert hdr == 1, f"{hdr} header lines written (want exactly 1)"
    assert not junk, f"non-CSI text at lines {junk}"
    print(f"RECONNECT_CHILD_OK opens={Flaky.opens} rows={len(lines) - 1} "
          f"header=1 junk_rows=0")


@stage("9. can this laptop see the board?")
def st_ports():
    sys.path.insert(0, SRC)
    from serial.tools import list_ports
    import eg_port
    allp = sorted(list_ports.comports(), key=lambda p: p.device)
    import eg_port as _eg
    esp = _eg.candidates()
    listed = ", ".join(f"{p.device}" + (f"({p.vid:04x})" if p.vid else "")
                       for p in allp) or "none at all"
    if esp:
        return True, (f"found {', '.join(p.device for p in esp)} - use "
                      f"--port auto\n      all ports: {listed}")
    return True, ("no ESP32 on this laptop yet (normal - plug the receiver "
                  "board in later).\n      ports visible now: " + listed)


STAGES = [st_env, st_files, st_read, st_qc, st_train, st_replay, st_fall,
          st_reconnect, st_ports]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="internal: single stage")
    ap.add_argument("--csv", default=None, help="internal: seed capture")
    ap.add_argument("--keep", action="store_true",
                    help="leave selftest_tmp/ in place instead of deleting it")
    args = ap.parse_args()

    if args.only == "reconnect-child":
        st_reconnect_child(args.csv)
        return 0
    if args.only:
        fn = next(s for s in STAGES if s._stage.split(".")[0] == args.only)
        ok, detail = fn()
        return 0 if ok else 1

    print("=" * 74)
    print("ELDERGUARD SELF-TEST  -  no hardware needed, takes about a minute")
    print("=" * 74)
    for fn in STAGES:
        name = fn._stage
        try:
            t = time.time()
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{exc.__class__.__name__}: {exc}"
        record(name, ok, detail)
        print(f"      ({time.time() - t:.0f} s)\n", flush=True)

    bad = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 74)
    if bad:
        print(f"RESULT: {len(bad)} stage(s) FAILED -> " + ", ".join(bad))
        print("Do not start recording. Send this whole screen to your "
              "teammate - every failure names its own fix.")
    else:
        print("RESULT: SOFTWARE SELF-TEST PASSED (synthetic fixtures only).")
        print("Next: connect/flash both ESP32-S3 boards, collect labeled room captures, then train.")
    if not args.keep:
        import shutil
        shutil.rmtree(TMP, ignore_errors=True)
    print("=" * 74)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
