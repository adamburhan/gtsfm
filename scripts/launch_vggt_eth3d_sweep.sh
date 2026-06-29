#!/bin/bash
# Submit the full VGGT ETH3D depth-factor sweep: 13 sequences x 4 depth modes x gap thresholds.
# Sequences are the high-res multi-view 'training' split (public GT), matching eth3d_benchmark.sh.
set -e

GS_STEPS=7000
GAP="0.10"

# Four-way comparison under one root, so `aggregate_modes.py --root` tables them side by side:
#   none          - baseline BA, no depth factors
#   bimodal_gap   - VGGT depth + patch, largest-gap ambiguity analysis
#   bimodal_gmm   - VGGT depth + patch, 2-component GMM ambiguity analysis
#   bimodal_mda   - MDA mixture modes (precomputed at $SCRATCH/mda_mixture/<seq>_mda)
# Single-cluster scenes that already have MDA mixtures. Conditions differ only in the depth
# source / ambiguity analyzer; everything else is held fixed.
export SWEEP_NAME=${SWEEP_NAME:-eth3d_g${GAP}_compare}
sequences="kicker office pipes relief terrace delivery_area"

# Root: $SCRATCH/logs/sweeps/$SWEEP_NAME/<seq>/<mode>
for SEQ in $sequences; do
    for MODE in none bimodal_gap bimodal_gmm bimodal_mda; do
        cluv submit mila scripts/vggt_eth3d_pred_sweep.sh -- $SEQ $MODE $GS_STEPS $GAP
    done
done
