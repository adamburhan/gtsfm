#!/bin/bash
# Submit the full Replica depth-factor sweep: 8 sequences x 4 depth modes = 32 jobs.
# Per-job: stride 10 (200 frames), GS 7000 steps (replica_sweep.sh defaults).
set -e

for SEQ in office0 office1 office2 office3 office4 room0 room1 room2; do
    for MODE in none unimodal drop_ambiguous bimodal; do
        cluv submit mila scripts/replica_sweep.sh -- $SEQ $MODE
    done
done
