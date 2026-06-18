#!/bin/bash
# Submit the full Replica depth-factor sweep: 8 sequences x 4 depth modes x gap thresholds.
# Per-job: stride 10 (200 frames), GS 7000 steps (replica_sweep.sh defaults).
set -e

STRIDE=10
GS_STEPS=7000
GAP_THRESHOLDS="0.05 0.10 0.15"

for SEQ in office0 office1 office2 office3 office4 room0 room1 room2; do
    for MODE in none unimodal drop_ambiguous bimodal; do
        if [ "$MODE" = "none" ] || [ "$MODE" = "unimodal" ]; then
            cluv submit mila scripts/replica_sweep.sh -- $SEQ $MODE $STRIDE $GS_STEPS 0.0
        else
            for GAP in $GAP_THRESHOLDS; do
                cluv submit mila scripts/replica_sweep.sh -- $SEQ $MODE $STRIDE $GS_STEPS $GAP
            done
        fi
    done
done
