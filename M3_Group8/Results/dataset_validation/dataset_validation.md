# ElderGuard dataset validation

Data folder: `C:\Users\DELL\Downloads\ElderGuard_M3_WiVi32_adapted\ElderGuard_M3\Data\classroom`

CSV files found: 6

| File | Label | Header | Valid CSI | Malformed | Missing | Duplicates | CSI byte lengths | RSSI mean | Timestamp rate Hz | Est. duration s | Large gaps |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| diagnostic.csv | None | OK | 863 | 0 | 0 | 0 | `{'256': 36, '384': 827}` | -56.62224797219003 | 65.74621959237344 | 13.126230000000001 | 0 |
| roomA_0p_t01.csv | 0 | OK | 4578 | 0 | 0 | 0 | `{'128': 51, '256': 397, '384': 4130}` | -43.521625163827 | 62.38303181534623 | 73.38534 | 21 |
| roomA_1p_t01.csv | 1 | OK | 5808 | 0 | 0 | 0 | `{'256': 465, '384': 5343}` | -48.39066804407714 | 66.17257808364214 | 87.77049600000001 | 13 |
| roomA_2p_t01.csv | 2 | OK | 5563 | 0 | 0 | 0 | `{'128': 26, '256': 395, '384': 5142}` | -49.55222002516628 | 61.862047633776676 | 89.925895 | 5 |
| roomA_2p_t01_fs60.csv | 2 | BAD | 0 | 0 | 0 | 0 | `{}` |  |  |  |  |
| roomA_2p_t04_fs60.csv | 2 | BAD | 0 | 0 | 0 | 0 | `{}` |  |  |  |  |

## Expected files

- All expected filenames found.

Labeled capture counts by class: `{'0': 1, '1': 1, '2': 1}`.
CSI vector lengths count signed-byte values as serialized; an even-length vector is required for I/Q pairing.
The validator does not alter or discard source records. Timestamp gap detection uses the device counter and a conservative heuristic; verify suspicious gaps against the raw capture and measured rate.
