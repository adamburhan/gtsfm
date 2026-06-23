#!/bin/bash
# Aggregate a finished depth-factor sweep into final tables.
#
# Runs on a login node (CPU-only, seconds): discovers <root>/<seq>/<mode> runs, reads the
# small metric JSONs, and writes all_metrics.csv + paper_table.tex under <root>/tables.
# Sync only that tables dir back — not the per-run artifacts.
#
# Usage: scripts/aggregate_sweep.sh <sweep_root> [out_dir]
#   e.g. scripts/aggregate_sweep.sh $SCRATCH/logs/sweeps/eth3d_g0.10

set -eo pipefail

ROOT=${1:?usage: aggregate_sweep.sh <sweep_root> [out_dir]}
NAME=$(basename "$ROOT")
OUT_DIR=${2:-$ROOT/tables}

uv run python scripts/aggregate_modes.py \
    --root "$ROOT" \
    --latex \
    --out_dir "$OUT_DIR" \
    --caption "Depth-factor sweep (${NAME})." \
    --label "tab:${NAME}"

echo
echo "Tables written to $OUT_DIR"
echo "Sync only the tables back, e.g.:"
echo "  rsync -a mila:'${OUT_DIR}/' ./tables/${NAME}/"
