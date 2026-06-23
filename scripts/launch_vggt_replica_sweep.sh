#!/bin/bash
# Submit the full VGGT Replica depth-factor sweep: 8 sequences x 4 depth modes x gap thresholds.
set -e

STRIDE=10
GS_STEPS=7000
GAP_THRESHOLDS="0.10"

sequences="office0 office1 office2 office3 office4 room0 room1 room2"

# Each gap is its own aggregatable root ($SCRATCH/logs/sweeps/replica_g<gap>/<seq>/<mode>)
for SEQ in $sequences; do
    for GAP in $GAP_THRESHOLDS; do
        for MODE in none unimodal drop_ambiguous bimodal; do
            cluv submit mila scripts/vggt_replica_sweep.sh -- $SEQ $MODE $STRIDE $GS_STEPS $GAP
        done
    done
done
