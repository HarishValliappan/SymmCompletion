#!/usr/bin/env python3
"""Plot Val F-Score vs Epoch for PCN runs from SymmCompletion logs.

Parses log lines like:
  [Validation] EPOCH: 450  Metrics = ['0.5757', '9.1101', '0.6545']
where Metrics[0] is the overall validation F-Score.

Defaults are tailored to your request:
- scan: logs/*pcn*.err
- epoch range: 0..450
- plot: line-only (no scatter points)
- axes start from (0,0)

Example:
  python plot_pcn_val_fscore.py --out plots/pcn_val_fscore_vs_epoch.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


VAL_LINE_RE = re.compile(
    r"\[Validation\]\s*EPOCH:\s*(?P<epoch>\d+)\s+Metrics\s*=\s*\[(?P<metrics>.*)\]"
)
NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ValPoint:
    epoch: int
    fscore: float
    source: str


def _extract_floats(text: str) -> List[float]:
    return [float(m.group(0)) for m in NUM_RE.finditer(text)]


def parse_validation_points(log_path: str) -> List[ValPoint]:
    points: List[ValPoint] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = VAL_LINE_RE.search(line)
                if not m:
                    continue
                epoch = int(m.group("epoch"))
                metrics_text = m.group("metrics")
                # Skip collapsed/invalid lines containing NaN/Inf.
                if re.search(r"\b(?:nan|inf)\b", metrics_text, flags=re.IGNORECASE):
                    continue
                floats = _extract_floats(metrics_text)
                if not floats:
                    continue
                fscore = floats[0]
                if not (0.0 <= fscore <= 1.0):
                    continue
                points.append(ValPoint(epoch=epoch, fscore=fscore, source=log_path))
    except FileNotFoundError:
        return []
    return points


def aggregate_best_by_epoch(points: Iterable[ValPoint]) -> Dict[int, ValPoint]:
    best: Dict[int, ValPoint] = {}
    for p in points:
        prev = best.get(p.epoch)
        if prev is None or p.fscore > prev.fscore:
            best[p.epoch] = p
    return best


def write_csv(path: str, rows: List[Tuple[int, float, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_fscore", "source"])
        for r in rows:
            w.writerow(r)


def plot_series_line_only(
    epochs: List[int],
    fscores: List[float],
    out_path: str,
    title: str,
    xlim: Tuple[int, int],
    ylim: Tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(epochs, fscores, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F-Score")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-glob",
        default="logs/*pcn*.err",
        help="Glob to PCN log files (relative to --root). Default: logs/*pcn*.err",
    )
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Project root to resolve --logs-glob and output paths (default: script dir).",
    )
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=0,
        help="Keep points with epoch >= MIN_EPOCH (default: 0).",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=450,
        help="Keep points with epoch <= MAX_EPOCH (default: 450).",
    )
    parser.add_argument(
        "--prepend-origin",
        action="store_true",
        default=True,
        help="Add an (0,0) point so the plotted line starts at origin (default: on).",
    )
    parser.add_argument(
        "--no-prepend-origin",
        action="store_false",
        dest="prepend_origin",
        help="Do not add an (0,0) point.",
    )
    parser.add_argument(
        "--out",
        default="plots/pcn_val_fscore_vs_epoch.png",
        help="Output image path (relative to --root).",
    )
    parser.add_argument(
        "--csv",
        default="plots/pcn_val_fscore_vs_epoch.csv",
        help="Output CSV path (relative to --root).",
    )

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    logs_glob = os.path.join(root, args.logs_glob)
    log_paths = sorted(glob.glob(logs_glob))

    if not log_paths:
        print(f"No log files matched: {logs_glob}")
        return 2

    all_points: List[ValPoint] = []
    for p in log_paths:
        all_points.extend(parse_validation_points(p))

    if not all_points:
        print("No validation points found. Expected lines like '[Validation] EPOCH: ... Metrics = [...]'.")
        return 3

    by_epoch = aggregate_best_by_epoch(all_points)
    by_epoch = {
        e: p
        for e, p in by_epoch.items()
        if (args.min_epoch is None or e >= args.min_epoch) and (args.max_epoch is None or e <= args.max_epoch)
    }

    rows = [(e, by_epoch[e].fscore, os.path.relpath(by_epoch[e].source, root)) for e in sorted(by_epoch)]
    csv_path = os.path.join(root, args.csv)
    write_csv(csv_path, rows)

    epochs = [e for e, _, _ in rows]
    fscores = [v for _, v, _ in rows]

    if args.prepend_origin:
        if not epochs or epochs[0] != 0:
            epochs = [0] + epochs
            fscores = [0.0] + fscores

    out_path = os.path.join(root, args.out)
    plot_series_line_only(
        epochs,
        fscores,
        out_path=out_path,
        title="PCN Val F-Score vs Epoch",
        xlim=(0, args.max_epoch if args.max_epoch is not None else max(epochs)),
        ylim=(0.0, 1.0),
    )

    print(f"Parsed {len(all_points)} validation lines from {len(log_paths)} logs")
    print(f"Aggregated epochs in range: {len(by_epoch)}")
    print(f"Wrote plot: {out_path}")
    print(f"Wrote CSV:  {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
