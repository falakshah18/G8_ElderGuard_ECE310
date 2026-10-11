# Data

No raw CSI data is checked into this repository — the public dataset is large
and not ours to redistribute, and our own captures are added as they are made.

## Stage 1 — public dataset used for the M2 results

**RadioPoints WiFi-CrowdCounting** dataset (Di Domenico et al., IEEE WiMob 2016)

- Source: `https://github.com/RadioPoints/Device-free_RF_Human_Sensing_Datasets`
- Hardware: Intel 5300 NIC, 30 subcarriers, 3 RX × 2 TX streams
- Rooms: three different rooms; occupancies 0–8 people per room
- M2 scope: the **0 / 1 / 2 person** classes only
- Frame rate: ≈ 55–60 Hz, measured from capture timestamps

### Expected directory layout for `parse_csi.load_radiopoints`

```
Data/radiopoints/
├── roomA/
│   ├── 0_people/   *.dat  (or decoded *.csv)
│   ├── 1_person/   *.dat
│   └── 2_people/   *.dat
├── roomB/ ...
└── roomC/ ...
```

The folder name encodes the occupancy and the parent folder encodes the room.
`.dat` files are parsed with `csiread`; if `csiread` is unavailable, place a
decoded CSV (`amp_0 … amp_N`, optional `rssi`) beside each capture and it will
be used instead.

## Stage 2 — our own hardware capture (immediate M3 task)

Firmware is complete and lives in `Data/arduino/`:

- `transmitter_esp8266/` — ESP8266 NodeMCU v3 soft-AP + UDP flood (~100 pkt/s).
  The ESP8266 has no CSI API, so it is a transmitter only.
- `receiver_csi/` — ESP32-WROOM-32D joins the soft-AP and captures CSI, printing
  one ESP32-CSI-Tool CSV row per frame.
- `self_test/` — single-board bring-up check (Wi-Fi scan + CSI callback).

Captured data (I/Q across 52 subcarriers, RSSI, timestamp at ≈100 Hz) and the
collection protocol (rooms, occupancies, trials) will be added here once
hardware capture is complete.
