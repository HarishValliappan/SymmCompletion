#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python plot_pcn_val_fscore.py \
  --logs-glob 'logs/*pcn*.err' \
  --min-epoch 0 \
  --max-epoch 450 \
  --out plots/pcn_val_fscore_vs_epoch.png \
  --csv plots/pcn_val_fscore_vs_epoch.csv
