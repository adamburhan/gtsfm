#!/bin/bash
# Submit the full VGGT ETH3D depth-factor sweep: 13 sequences x 4 depth modes x gap thresholds.
# Sequences are the high-res multi-view 'training' split (public GT), matching eth3d_benchmark.sh.
set -e

GS_STEPS=7000
GAP_THRESHOLDS="0.10"

sequences="courtyard delivery_area electro facade kicker meadow office pipes playground relief_2 relief terrace terrains"

# Each gap is its own aggregatable root ($SCRATCH/logs/sweeps/eth3d_g<gap>/<seq>/<mode>)
for SEQ in $sequences; do
    for GAP in $GAP_THRESHOLDS; do
        for MODE in none unimodal drop_ambiguous bimodal; do
            cluv submit mila scripts/vggt_eth3d_pred_sweep.sh -- $SEQ $MODE $GS_STEPS $GAP
        done
    done
done
