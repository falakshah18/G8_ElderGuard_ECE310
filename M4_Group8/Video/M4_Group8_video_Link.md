# M4 Videos

## Video 1 — Demonstration Video

**File:** `Video/M4_Demonstration_Video.mp4` (included in this repository)

The demonstration video shows the ElderGuard two-ESP32-S3 Wi-Fi CSI rig in action:
the transmitter board flooding packets, the receiver streaming CSI to the laptop,
and the live terminal display of room status — `NO PERSON PRESENT` /
`ONE PERSON PRESENT` / `TWO PEOPLE PRESENT` — as people enter and leave the
line between the boards.

## Video 2 — Full Project Walkthrough

**Video:** M4 Walkthrough Video (large file, hosted on Google Drive):
https://drive.google.com/file/d/1G7rk3ea2PLptgkQB5C7VHbpklWIEn_hQ/view?usp=sharing

The walkthrough video covers every file in this repository: the firmware, the
pipeline library, the tools, the hardware enclosure (CAD + 3D-printed parts),
the raw captures, the results files, and a live no-hardware replay demo.

Related material:
- Full analysis: `Report/M4_Group8_Report.pdf`
- Reproduce the live demo: `py -3.12 tools/count_demo.py` (see `Report/M4_Group8_Report.pdf` §9)
