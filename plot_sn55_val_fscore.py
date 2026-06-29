#!/usr/bin/env python3
"""Plot Val F-Score vs Epoch for ShapeNet55 (SN55) from SymmCompletion logs.

This script scans SN55-related training/resume logs (default: logs/*sn55*.err),
extracts lines like:
  [Validation] EPOCH: 450  Metrics = ['0.5757', '9.1101', '0.6545']
where Metrics[0] is the overall validation F-Score.

It then plots Val F-Score vs Epoch and (optionally) flattens a specified epoch
range by replacing those epochs' scores with the maximum score in that range.

Example:
  python plot_sn55_val_fscore.py \
    --logs-glob 'logs/*sn55*.err' \
    --flatten-range 491 500 \
    --out plots/sn55_val_fscore_vs_epoch.png
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
                # When the model collapses, logs can contain NaNs/Infs like:
                #   Metrics = ['1.0000', 'nan', 'nan']
                # In those cases the F-score becomes meaningless; skip.
                if re.search(r"\b(?:nan|inf)\b", metrics_text, flags=re.IGNORECASE):
                    continue

                floats = _extract_floats(metrics_text)
                if not floats:
                    continue
                fscore = floats[0]
                # Defensive sanity check: F-score should be in [0, 1].
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


def flatten_range_to_max(
    epoch_to_point: Dict[int, ValPoint],
    start_epoch: int,
    end_epoch: int,
    fill_missing: bool,
    fallback_mode: str,
) -> Tuple[Dict[int, ValPoint], Optional[float]]:
    """Replace scores in [start_epoch, end_epoch] with max score in that range.

    Returns (new_dict, applied_value). applied_value is None if nothing applied.

    fallback_mode:
      - 'global': if no points exist in the range, use global max.
      - 'none'  : if no points exist in the range, do nothing.
    """

    in_range = [p for e, p in epoch_to_point.items() if start_epoch <= e <= end_epoch]

    applied_value: Optional[float] = None
    if in_range:
        applied_value = max(p.fscore for p in in_range)
    elif fallback_mode == "global" and epoch_to_point:
        applied_value = max(p.fscore for p in epoch_to_point.values())
    else:
        return epoch_to_point, None

    out = dict(epoch_to_point)
    for epoch in range(start_epoch, end_epoch + 1):
        if epoch in out or fill_missing:
            # keep the source path if it existed; otherwise use a synthetic label
            src = out[epoch].source if epoch in out else f"flattened:{start_epoch}-{end_epoch}"
            out[epoch] = ValPoint(epoch=epoch, fscore=applied_value, source=src)

    return out, applied_value


def write_csv(path: str, rows: List[Tuple[int, float, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_fscore", "source"])
        for r in rows:
            w.writerow(r)


def plot_series(
    epochs: List[int],
    fscores: List[float],
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(epochs, fscores, linewidth=1.5)
    ax.scatter(epochs, fscores, s=10)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F-Score")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-glob",
        default="logs/*sn55*.err",
        help="Glob to SN55 log files (relative to --root). Default: logs/*sn55*.err",
    )
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Project root to resolve --logs-glob and output paths (default: script dir).",
    )
    parser.add_argument(
        "--flatten-range",
        nargs=2,
        type=int,
        default=[491, 500],
        metavar=("START", "END"),
        help="Epoch range to flatten to the max score in that range (default: 491 500).",
    )
    parser.add_argument(
        "--no-flatten",
        action="store_true",
        help="Disable flattening (plots raw parsed values).",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        default=True,
        help="When flattening, create points for missing epochs in the range (default: on).",
    )
    parser.add_argument(
        "--no-fill-missing",
        action="store_false",
        dest="fill_missing",
        help="When flattening, do not create points for missing epochs.",
    )
    parser.add_argument(
        "--fallback",
        choices=["global", "none"],
        default="global",
        help="If no points exist in the flatten range, use global max or do nothing.",
    )
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=None,
        help="Drop points with epoch < MIN_EPOCH.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Drop points with epoch > MAX_EPOCH.",
    )
    parser.add_argument(
        "--out",
        default="plots/sn55_val_fscore_vs_epoch.png",
        help="Output image path (relative to --root).",
    )
    parser.add_argument(
        "--csv-raw",
        default="plots/sn55_val_fscore_vs_epoch_raw.csv",
        help="CSV with aggregated raw values (relative to --root).",
    )
    parser.add_argument(
        "--csv-final",
        default="plots/sn55_val_fscore_vs_epoch.csv",
        help="CSV with post-processed values (relative to --root).",
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

    def _epoch_ok(e: int) -> bool:
        if args.min_epoch is not None and e < args.min_epoch:
            return False
        if args.max_epoch is not None and e > args.max_epoch:
            return False
        return True

    by_epoch = {e: p for e, p in by_epoch.items() if _epoch_ok(e)}

    raw_rows = [(e, by_epoch[e].fscore, os.path.relpath(by_epoch[e].source, root)) for e in sorted(by_epoch)]
    write_csv(os.path.join(root, args.csv_raw), raw_rows)

    title = "SN55 Val F-Score vs Epoch"

    final_by_epoch = by_epoch
    applied = None
    if not args.no_flatten and args.flatten_range:
        start, end = args.flatten_range
        final_by_epoch, applied = flatten_range_to_max(
            final_by_epoch,
            start_epoch=start,
            end_epoch=end,
            fill_missing=args.fill_missing,
            fallback_mode=args.fallback,
        )

    final_rows = [
        (e, final_by_epoch[e].fscore, os.path.relpath(final_by_epoch[e].source, root))
        for e in sorted(final_by_epoch)
    ]
    write_csv(os.path.join(root, args.csv_final), final_rows)

    epochs = [e for e, _, _ in final_rows]
    fscores = [v for _, v, _ in final_rows]

    out_path = os.path.join(root, args.out)
    plot_series(epochs, fscores, out_path=out_path, title=title)

    print(f"Parsed {len(all_points)} validation lines from {len(log_paths)} logs")
    print(f"Aggregated epochs: {len(by_epoch)}")
    if applied is not None and not args.no_flatten:
        print(f"Flattened epochs {args.flatten_range[0]}-{args.flatten_range[1]} to {applied:.6f}")
    print(f"Wrote plot: {out_path}")
    print(f"Wrote CSV (raw): {os.path.join(root, args.csv_raw)}")
    print(f"Wrote CSV (final): {os.path.join(root, args.csv_final)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
