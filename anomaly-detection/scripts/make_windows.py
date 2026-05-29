#!/usr/bin/env python3
"""
make_windows.py -- Convert BATADAL attack list CSVs into windows JSON files
                   required by train.py --exclude-windows.

Usage:
    python3 anomaly-detection/scripts/make_windows.py \
        --dataset2-labels data/BATADAL_dataset2_attack_list.csv \
        --test-labels     data/BATADAL_test_attack_list.csv \
        --output-dir      data/

Produces:
    data/dataset2_windows.json
    data/test_windows.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_attack_list(csv_path: str) -> list:
    """
    Read a BATADAL attack list CSV and return a list of window dicts
    in the format expected by train.py --exclude-windows.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    start_col = next(c for c in df.columns if "Starting" in c or "Start" in c)
    end_col   = next(c for c in df.columns if "Ending"   in c or "End"   in c)
    id_col    = next(c for c in df.columns if c.strip() == "ID")
    desc_col  = next((c for c in df.columns if "Description" in c), None)

    windows = []
    for _, row in df.iterrows():
        try:
            start = pd.to_datetime(str(row[start_col]).strip(), dayfirst=True)
            end   = pd.to_datetime(str(row[end_col]).strip(),   dayfirst=True)
            label = f"attack_{int(row[id_col])}"
            if desc_col:
                desc  = str(row[desc_col]).strip()
                label = f"attack_{int(row[id_col])}_{desc[:30].replace(chr(32),chr(95)).replace(chr(44),chr(0))[:-1]}"
            windows.append({
                "start": start.strftime("%Y-%m-%d %H:%M"),
                "end":   end.strftime("%Y-%m-%d %H:%M"),
                "label": label,
            })
            print(f"  Attack {int(row[id_col]):>2}  {start} -> {end}")
        except Exception as e:
            print(f"  Warning: could not parse row: {e}")

    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Convert BATADAL attack list CSVs to windows JSON files."
    )
    parser.add_argument("--dataset2-labels", required=True)
    parser.add_argument("--test-labels",     required=True)
    parser.add_argument("--output-dir",      default="data/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nParsing {args.dataset2_labels} ...")
    d2_windows = parse_attack_list(args.dataset2_labels)
    d2_path = out_dir / "dataset2_windows.json"
    with open(d2_path, "w") as f:
        json.dump(d2_windows, f, indent=2)
    print(f"  -> {d2_path}  ({len(d2_windows)} windows)")

    print(f"\nParsing {args.test_labels} ...")
    test_windows = parse_attack_list(args.test_labels)
    test_path = out_dir / "test_windows.json"
    with open(test_path, "w") as f:
        json.dump(test_windows, f, indent=2)
    print(f"  -> {test_path}  ({len(test_windows)} windows)")

    print(f"\nDone. Both files written to {out_dir}/")


if __name__ == "__main__":
    main()
