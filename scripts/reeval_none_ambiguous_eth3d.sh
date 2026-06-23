#!/bin/bash
# Add ambiguous-subset + mode-correctness metrics to the `none` baseline of a finished ETH3D
# sweep, WITHOUT re-running VGGT/BA. The `none` runs have no depth.npz of their own, so we reuse
# a sibling mode's depth.npz (VGGT depth is identical across depth modes on the same GPU arch).
# Eval-only: geometry/Open3D on CPU, fast — run on a login node.
#
# Non-`none` single-cluster runs already carry these metrics. Scenes that partitioned into >1
# cluster (results/merged, no results/vggt) are skipped: ambiguous-subset is single-cluster only.
#
# Matches the sweep's eval call: same --gap_thresh, same (default) depth bounds/patch params.
#
# Usage: scripts/reeval_none_ambiguous_eth3d.sh [sweep_root] [gap_thresh]
set -eo pipefail

ROOT=${1:-$SCRATCH/logs/sweeps/eth3d_g0.10}
GAP=${2:-0.10}
ETH3D=$SCRATCH/datasets/eth3d

for SEQDIR in "$ROOT"/*/; do
    SEQ=$(basename "$SEQDIR")
    NONE="$SEQDIR/none/results/vggt"
    if [ ! -d "$NONE" ]; then
        echo "skip $SEQ: no single-cluster none/results/vggt (merged or missing)"
        continue
    fi
    DEPTH=$(ls "$SEQDIR"/{bimodal,unimodal,drop_ambiguous}/results/depth.npz 2>/dev/null | head -1)
    if [ -z "$DEPTH" ]; then
        echo "skip $SEQ: no sibling depth.npz to borrow"
        continue
    fi
    echo "=== re-eval none: $SEQ (depth from $(basename "$(dirname "$(dirname "$DEPTH")")")) ==="
    uv run python gtsfm/evaluation/eval_geometry.py \
        --sfm_output "$NONE" \
        --depth_npz "$DEPTH" --gap_thresh "$GAP" \
        --align_mode eth3d \
        --align_ref "$ETH3D/$SEQ/dslr_calibration_undistorted" \
        --gt_ply "$ETH3D/$SEQ/${SEQ}_gt.ply" \
        --tau 0.01 0.02 0.05 0.1 0.2 0.5 \
        --out "$SEQDIR/none/geometry_metrics.json"
done

echo
echo "Done. Re-aggregate with:  scripts/aggregate_sweep.sh $ROOT"
