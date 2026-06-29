#!/usr/bin/env python3
"""Plot PCN training + validation F-score curves from SymmCompletion logs.

Goal:
    Plot two F-score (F1) curves vs epoch:
        - Train F-score (if present in logs)
        - Val F-score (from '[Validation] ... Metrics[0]')

Important:
    Many runs only log *training losses* and *validation F-score* (no train F-score).
    If train F-score is not found in logs, this script will generate a surrogate
    second curve as the running maximum of Val F-score ("best-so-far").

Defaults:
    - scan: logs/*pcn*.err
    - epoch range: 0..450
    - line-only plots
    - axes and both curves start at (0,0)

Output:
    - plots/pcn_train_and_val_fscore_vs_epoch.png
    - plots/pcn_train_and_val_fscore_vs_epoch.csv
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
TRAIN_F_RE_CANDIDATES = [
    # Examples (if your code ever logs them):
    #   [Training] EPOCH: 12  F-Score: 0.8123
    #   [Train] EPOCH: 12 Metrics = ['0.8123', ...]
    re.compile(r"\[Training\].*?EPOCH:\s*(?P<epoch>\d+).*?F-Score\s*[:=]\s*(?P<f>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    re.compile(r"\[Train\].*?EPOCH:\s*(?P<epoch>\d+).*?F-Score\s*[:=]\s*(?P<f>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    re.compile(r"\[Training\].*?EPOCH:\s*(?P<epoch>\d+).*?Metrics\s*=\s*\[(?P<metrics>.*)\]"),
    re.compile(r"\[Train\].*?EPOCH:\s*(?P<epoch>\d+).*?Metrics\s*=\s*\[(?P<metrics>.*)\]"),
]
NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Point:
    epoch: int
    value: float
    source: str


def _extract_floats(text: str) -> List[float]:
    return [float(m.group(0)) for m in NUM_RE.finditer(text)]


def _has_nan_inf(text: str) -> bool:
    return re.search(r"\b(?:nan|inf)\b", text, flags=re.IGNORECASE) is not None


def parse_val_fscore_points(log_path: str) -> List[Point]:
    points: List[Point] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = VAL_LINE_RE.search(line)
            if not m:
                continue
            metrics_text = m.group("metrics")
            if _has_nan_inf(metrics_text):
                continue
            floats = _extract_floats(metrics_text)
            if not floats:
                continue
            epoch = int(m.group("epoch"))
            fscore = floats[0]
            if not (0.0 <= fscore <= 1.0):
                continue
            points.append(Point(epoch=epoch, value=fscore, source=log_path))
    return points


def parse_train_fscore_points(log_path: str) -> List[Point]:
    points: List[Point] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for rx in TRAIN_F_RE_CANDIDATES:
                m = rx.search(line)
                if not m:
                    continue

                epoch = int(m.group("epoch"))
                if "f" in m.groupdict() and m.group("f") is not None:
                    fscore = float(m.group("f"))
                else:
                    metrics_text = m.groupdict().get("metrics", "")
                    if not metrics_text or _has_nan_inf(metrics_text):
                        continue
                    floats = _extract_floats(metrics_text)
                    if not floats:
                        continue
                    fscore = floats[0]

                if not (0.0 <= fscore <= 1.0):
                    continue
                points.append(Point(epoch=epoch, value=fscore, source=log_path))
                break
    return points


def aggregate_by_epoch(points: Iterable[Point], mode: str) -> Dict[int, Point]:
    """Aggregate duplicates per epoch.

    mode:
      - 'min': keep the smallest value
      - 'max': keep the largest value
      - 'first': keep first seen
    """

    out: Dict[int, Point] = {}
    for p in points:
        prev = out.get(p.epoch)
        if prev is None:
            out[p.epoch] = p
            continue
        if mode == "min" and p.value < prev.value:
            out[p.epoch] = p
        elif mode == "max" and p.value > prev.value:
            out[p.epoch] = p
        elif mode == "first":
            pass
    return out


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_fscore",
                "train_source",
                "val_fscore",
                "val_source",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_fscore_two_curves(
    epochs_train: List[int],
    train_vals: List[float],
    epochs_val: List[int],
    val_vals: List[float],
    out_path: str,
    xlim: Tuple[int, int],
    train_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(epochs_train, train_vals, linewidth=1.6, label=train_label)
    ax.plot(epochs_val, val_vals, linewidth=1.6, label="Val F-Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F-Score")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-glob",
        default="logs/*pcn*.err",
        help="Glob to PCN log files (relative to --root).",
    )
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Project root to resolve --logs-glob and output paths (default: script dir).",
    )
    parser.add_argument("--min-epoch", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=450)
    parser.add_argument(
        "--out",
        default="plots/pcn_train_and_val_fscore_vs_epoch.png",
        help="Output image path (relative to --root).",
    )
    parser.add_argument(
        "--csv",
        default="plots/pcn_train_and_val_fscore_vs_epoch.csv",
        help="Output CSV path (relative to --root).",
    )

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    log_paths = sorted(glob.glob(os.path.join(root, args.logs_glob)))
    if not log_paths:
        print(f"No log files matched: {os.path.join(root, args.logs_glob)}")
        return 2

    all_val: List[Point] = []
    all_train_f: List[Point] = []
    for path in log_paths:
        all_val.extend(parse_val_fscore_points(path))
        all_train_f.extend(parse_train_fscore_points(path))

    # Aggregate duplicates: best Val (max), best Train F (max)
    val_by_epoch = aggregate_by_epoch(all_val, mode="max")
    train_by_epoch = aggregate_by_epoch(all_train_f, mode="max")

    def in_range(epoch: int) -> bool:
        return args.min_epoch <= epoch <= args.max_epoch

    val_by_epoch = {e: p for e, p in val_by_epoch.items() if in_range(e)}
    train_by_epoch = {e: p for e, p in train_by_epoch.items() if in_range(e)}

    if not val_by_epoch:
        print("No validation points found in epoch range.")
        return 3
    using_surrogate = False
    if not train_by_epoch:
        # No train F-score in logs: derive a surrogate curve from validation.
        using_surrogate = True
        running: Dict[int, Point] = {}
        best_so_far = 0.0
        for e in sorted(val_by_epoch):
            best_so_far = max(best_so_far, val_by_epoch[e].value)
            running[e] = Point(epoch=e, value=best_so_far, source="derived:running_max_val")
        train_by_epoch = running

    # Build merged rows for CSV (+ optional origin)
    epochs = sorted(set(val_by_epoch.keys()) | set(train_by_epoch.keys()) | {0})
    rows: List[dict] = []
    for e in epochs:
        t = train_by_epoch.get(e)
        v = val_by_epoch.get(e)
        if e == 0 and t is None and v is None:
            # Force both curves to start at (0,0)
            t = Point(epoch=0, value=0.0, source="forced_origin")
            v = Point(epoch=0, value=0.0, source="forced_origin")
        rows.append(
            {
                "epoch": e,
                "train_fscore": "" if t is None else f"{t.value:.6f}",
                "train_source": "" if t is None else (t.source if t.source.startswith("derived:") or t.source in {"forced_origin"} else os.path.relpath(t.source, root)),
                "val_fscore": "" if v is None else f"{v.value:.6f}",
                "val_source": "" if v is None else (v.source if v.source == "forced_origin" else os.path.relpath(v.source, root)),
            }
        )

    csv_path = os.path.join(root, args.csv)
    write_csv(csv_path, rows)

    # Series for plotting (include origin at 0)
    epochs_train = sorted(set(train_by_epoch.keys()) | {0})
    epochs_val = sorted(set(val_by_epoch.keys()) | {0})

    def _value_or_zero(mapping: Dict[int, Point], epoch: int) -> float:
        p = mapping.get(epoch)
        return 0.0 if p is None else p.value

    train_vals = [_value_or_zero(train_by_epoch, e) for e in epochs_train]
    val_vals = [_value_or_zero(val_by_epoch, e) for e in epochs_val]

    out_path = os.path.join(root, args.out)
    train_label = "Train F-Score" if not using_surrogate else "Best-So-Far F-Score (from Val)"
    plot_fscore_two_curves(
        epochs_train,
        train_vals,
        epochs_val,
        val_vals,
        out_path=out_path,
        xlim=(0, args.max_epoch),
        train_label=train_label,
    )

    print(f"Scanned {len(log_paths)} logs")
    print(f"Val epochs: {len(val_by_epoch)}")
    if using_surrogate:
        print("WARNING: Train F-score not found in logs; using running max of Val F-score as the second curve.")
    else:
        print(f"Train epochs: {len(train_by_epoch)}")
    print(f"Wrote plot: {out_path}")
    print(f"Wrote CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
