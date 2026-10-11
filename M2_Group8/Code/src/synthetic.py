"""
ElderGuard - synthetic CSI session generator (M2)
ECE 310 Group 8

Purpose: validate the whole pipeline (preprocess -> features -> models ->
evaluate) end-to-end without the public dataset, and give the firmware team a
controlled signal to test against. It is NOT a source of reported results -
those come from the real RadioPoints dataset (see Data/README.md).

The generator builds three classes (0/1/2 people) with physically motivated
structure:
  * 0 people   - near-constant amplitude, small thermal noise.
  * 1 person   - slow body sway, moderate amplitude variance, low Doppler.
  * 2 people   - more independent motion, higher variance and Doppler energy.
Each "session" is one (room, occupancy) capture; sessions are what
leave-one-session-out holds out.
"""

import numpy as np


def generate_session(n_people, fs=100, duration_s=20.0, n_subcarriers=52,
                     room="A", seed=0):
    """Return one session dict."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    base = 20.0 + rng.normal(0, 0.2, size=(n, n_subcarriers))

    if n_people == 0:
        motion = 0.0
        doppler_freqs = []
    elif n_people == 1:
        motion = 0.8
        doppler_freqs = [1.2]
    else:
        motion = 1.6
        doppler_freqs = [1.2, 2.4]

    amplitude = base.copy()
    for f_d in doppler_freqs:
        phase = rng.uniform(0, 2 * np.pi, size=(1, n_subcarriers))
        amplitude += motion * np.sin(2 * np.pi * f_d * t)[:, None] * np.cos(phase)

    raw_phase = rng.normal(0, 0.05, size=(n, n_subcarriers))
    for f_d in doppler_freqs:
        raw_phase += 0.3 * np.sin(2 * np.pi * f_d * t)[:, None]
    # CFO-driven linear drift across subcarriers (what sanitization removes)
    raw_phase += (0.01 * np.arange(n))[:, None]

    # RSSI drops slightly as more bodies absorb the signal; add small noise.
    rssi = -45.0 - 1.5 * n_people + rng.normal(0, 0.4, size=n)

    return {
        "amplitude": amplitude,
        "phase": raw_phase,
        "rssi": rssi,
        "fs": fs,
        "label": n_people,
        "room": room,
        "session_id": f"room{room}_{n_people}p_seed{seed}",
        "pdr": 0.99,
    }


def generate_dataset(rooms=("A", "B", "C"), people=(0, 1, 2), seed=0):
    """Return a list of sessions covering the requested rooms/occupancies."""
    sessions = []
    for r_i, room in enumerate(rooms):
        for p in people:
            sessions.append(
                generate_session(
                    p, room=room, seed=seed + 100 * r_i + p
                )
            )
    return sessions
