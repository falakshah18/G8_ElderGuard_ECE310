#!/usr/bin/env python3
"""Validate ElderGuard CSI capture CSVs without changing the source files.

Writes a Markdown and JSON report to Results/dataset_validation by default.
Uses only the Python standard library; malformed records are counted and
reported, never silently repaired or removed.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter

EXPECTED_HEADER = [
    "type", "channel", "mac", "rssi", "rate", "sig_mode", "mcs",
    "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc",
    "fec_coding", "sgi", "ampdu_cnt", "ant", "timestamp", "data",
]
LABEL_RE = re.compile(r"_(0|1|2)p(?:_|\.)", re.IGNORECASE)
CSI_RE = re.compile(r"^\[\s*(-?\d+(?:\s+-?\d+)*)?\s*\]$")


def finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def inspect_file(path):
    result = {
        "file": os.path.basename(path), "bytes": os.path.getsize(path),
        "label": None, "header_matches": False, "data_rows": 0,
        "valid_csi_rows": 0, "malformed_rows": 0, "missing_values": 0,
        "nonfinite_values": 0, "duplicate_rows": 0,
        "empty_file": os.path.getsize(path) == 0,
        "vector_lengths": {}, "rssi": {}, "timestamp": {}, "gaps": {},
        "errors": [],
    }
    match = LABEL_RE.search(os.path.basename(path))
    if match:
        result["label"] = int(match.group(1))

    vectors = Counter()
    rssis, stamps = [], []
    seen = set()
    try:
        with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                result["errors"].append({"file_error": "empty file: no header or records"})
                return result
            result["header_matches"] = header == EXPECTED_HEADER
            if not result["header_matches"]:
                result["errors"].append({"header": header, "expected": EXPECTED_HEADER})
            for line_num, row in enumerate(reader, start=2):
                if not row or all(not cell.strip() for cell in row):
                    continue
                result["data_rows"] += 1
                if len(row) != len(EXPECTED_HEADER):
                    result["malformed_rows"] += 1
                    result["errors"].append({"line": line_num, "issue": f"column_count={len(row)}"})
                    continue
                record = dict(zip(EXPECTED_HEADER, row))
                signature = tuple(row)
                if signature in seen:
                    result["duplicate_rows"] += 1
                seen.add(signature)
                if any(value.strip() == "" for value in row):
                    result["missing_values"] += 1
                if record["type"] != "CSI":
                    result["malformed_rows"] += 1
                    continue

                vector_match = CSI_RE.fullmatch(record["data"].strip())
                if not vector_match:
                    result["malformed_rows"] += 1
                    result["errors"].append({"line": line_num, "issue": "invalid CSI array syntax"})
                    continue
                values_text = vector_match.group(1) or ""
                values = [int(x) for x in values_text.split()] if values_text else []
                if not values or len(values) % 2:
                    result["malformed_rows"] += 1
                    result["errors"].append({"line": line_num, "issue": f"empty_or_odd_vector_len={len(values)}"})
                    continue
                vectors[len(values)] += 1
                rssi = finite_float(record["rssi"])
                if rssi is None:
                    result["missing_values"] += 1
                else:
                    rssis.append(rssi)
                stamp = finite_float(record["timestamp"])
                if stamp is None:
                    result["missing_values"] += 1
                else:
                    stamps.append(stamp)
                result["valid_csi_rows"] += 1
    except (OSError, csv.Error, StopIteration) as exc:
        result["errors"].append({"file_error": str(exc)})

    if rssis:
        ordered = sorted(rssis)
        result["rssi"] = {
            "min": min(rssis), "median": ordered[len(ordered) // 2],
            "max": max(rssis), "mean": sum(rssis) / len(rssis),
        }
    result["vector_lengths"] = {str(k): v for k, v in sorted(vectors.items())}
    if stamps:
        # Receiver timestamp is a device counter, not wall-clock time.
        backwards = sum(b < a for a, b in zip(stamps, stamps[1:]))
        raw_deltas = [b - a for a, b in zip(stamps, stamps[1:])]
        deltas = [d + 2**32 if d < -(2**31) else d for d in raw_deltas]
        positive = [d for d in deltas if d > 0]
        result["timestamp"] = {
            "first": stamps[0], "last": stamps[-1], "backward_steps": backwards,
            "positive_deltas": len(positive),
        }
        usable = [d for d in positive if d <= 100_000]
        if usable:
            med = sorted(usable)[len(usable) // 2]
            gaps = [d for d in positive if d > max(5 * med, 100_000)]
            result["timestamp"]["estimated_rate_hz"] = 1_000_000.0 / med
            result["timestamp"]["estimated_duration_s_from_frames"] = result["valid_csi_rows"] / (1_000_000.0 / med)
            result["gaps"] = {"median_delta": med, "large_gap_count": len(gaps),
                               "largest_delta": max(positive)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.join("Data", "classroom"))
    ap.add_argument("--out-dir", default=os.path.join("Results", "dataset_validation"))
    ap.add_argument("--expected", nargs="*", default=[
        "roomA_0p_t01.csv", "roomA_1p_t01.csv", "roomA_2p_t01.csv"])
    args = ap.parse_args()

    files = sorted(os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
                   if f.lower().endswith(".csv")) if os.path.isdir(args.data_dir) else []
    report = {"data_dir": os.path.abspath(args.data_dir), "expected_files": args.expected,
              "missing_expected_files": [f for f in args.expected
                                         if not os.path.isfile(os.path.join(args.data_dir, f))],
              "files": [inspect_file(f) for f in files]}
    class_counts = Counter(str(item["label"]) for item in report["files"]
                           if item["label"] is not None and item["valid_csi_rows"] > 0)
    report["capture_count_by_class"] = dict(class_counts)
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "dataset_validation.json")
    md_path = os.path.join(args.out_dir, "dataset_validation.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    lines = ["# ElderGuard dataset validation", "", f"Data folder: `{report['data_dir']}`", "",
             f"CSV files found: {len(files)}", "",
             "| File | Label | Header | Valid CSI | Malformed | Missing | Duplicates | CSI byte lengths | RSSI mean | Timestamp rate Hz | Est. duration s | Large gaps |",
             "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for item in report["files"]:
        rssi_mean = item.get("rssi", {}).get("mean", "")
        lines.append(f"| {item['file']} | {item['label']} | {'OK' if item['header_matches'] else 'BAD'} | "
                     f"{item['valid_csi_rows']} | {item['malformed_rows']} | {item['missing_values']} | "
                     f"{item['duplicate_rows']} | `{item['vector_lengths']}` | {rssi_mean} | "
                     f"{item.get('timestamp', {}).get('estimated_rate_hz', '')} | "
                     f"{item.get('timestamp', {}).get('estimated_duration_s_from_frames', '')} | "
                     f"{item.get('gaps', {}).get('large_gap_count', '')} |")
    lines += ["", "## Expected files", ""]
    lines.extend([f"- Missing: `{f}`" for f in report["missing_expected_files"]] or ["- All expected filenames found."])
    lines += ["", f"Labeled capture counts by class: `{dict(class_counts)}`.",
              "CSI vector lengths count signed-byte values as serialized; an even-length vector is required for I/Q pairing.",
              "The validator does not alter or discard source records. Timestamp gap detection uses the device counter and a conservative heuristic; verify suspicious gaps against the raw capture and measured rate.",
              ""]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        usable_files = [x for x in report["files"] if x["valid_csi_rows"]]
        if usable_files:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            names = [x["file"] for x in usable_files]
            rates = [x.get("timestamp", {}).get("estimated_rate_hz", 0) for x in usable_files]
            axes[0].bar(names, rates)
            axes[0].set_ylabel("Estimated frames/s")
            axes[0].set_title("Median positive CSI timestamp interval")
            axes[0].tick_params(axis="x", rotation=35)
            lengths = sorted({k for x in usable_files for k in x["vector_lengths"]}, key=int)
            bottoms = [0] * len(usable_files)
            for length in lengths:
                vals = [x["vector_lengths"].get(length, 0) for x in usable_files]
                axes[1].bar(names, vals, bottom=bottoms, label=f"{length} bytes")
                bottoms = [a + b for a, b in zip(bottoms, vals)]
            axes[1].set_ylabel("CSI frames")
            axes[1].set_title("Serialized CSI vector length distribution")
            axes[1].tick_params(axis="x", rotation=35)
            axes[1].legend()
            fig.tight_layout()
            plot_path = os.path.join(args.out_dir, "dataset_overview.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            report["plot"] = plot_path
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
    except ImportError:
        print("Plot skipped: matplotlib is not installed.")
    print(f"CSV files found: {len(files)}")
    print(f"Missing expected files: {report['missing_expected_files']}")
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
