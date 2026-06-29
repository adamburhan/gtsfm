#!/bin/bash
# Submit the classical (unified) ETH3D + Depth Pro sweep: none / bimodal_gap / bimodal_gmm /
# bimodal_gmm_null, under one root so aggregate_modes.py --root tables them together.
set -e

GAP="0.10"
export SWEEP_NAME=${SWEEP_NAME:-eth3d_unified_depthpro_g${GAP}}
export MAX_RES=${MAX_RES:-760}     # must match the resolution the Depth Pro maps were produced at
export NULL_NSIGMA=${NULL_NSIGMA:-5}

sequences="kicker office pipes relief terrace delivery_area"

for SEQ in $sequences; do
    for MODE in none bimodal_gap bimodal_gmm bimodal_gmm_null; do
        cluv submit mila scripts/unified_eth3d_depthpro_sweep.sh -- $SEQ $MODE $GAP
    done
done
