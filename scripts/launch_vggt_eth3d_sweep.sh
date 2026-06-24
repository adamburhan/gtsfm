#!/bin/bash
# Submit the full VGGT ETH3D depth-factor sweep: 13 sequences x 4 depth modes x gap thresholds.
# Sequences are the high-res multi-view 'training' split (public GT), matching eth3d_benchmark.sh.
set -e

GS_STEPS=7000
GAP="0.10"

# v1 MDA comparison: only the single-cluster scenes that already have precomputed MDA mixtures
# at $SCRATCH/mda_mixture/<seq>_mda. Conditions: none vs bimodal (VGGT depth + patch) vs
# bimodal_mda (MDA mixture modes). Same settings; differ only in the depth source.
sequences="kicker office pipes relief terrace delivery_area"

# Root: $SCRATCH/logs/sweeps/eth3d_g0.10/<seq>/<mode>
for SEQ in $sequences; do
    for MODE in none bimodal bimodal_mda; do
        cluv submit mila scripts/vggt_eth3d_pred_sweep.sh -- $SEQ $MODE $GS_STEPS $GAP
    done
done
