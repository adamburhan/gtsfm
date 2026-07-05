#!/bin/bash
# Launch the per-image profiled scale (_rs_pis) + relative-sigma (_rs) sweep (spec §7-8).
# Knobs are POSITIONAL args to the sweep (cluv submit does not forward env vars):
#   <seq> <mode> <gap> <sweep_name> <depth_subdir> <gt_scale> <gt_gate> <auto_scale>
# Writes under the SAME table roots as the existing conditions (TABLE/<source>/<seq>/<mode>), so
# one aggregate --root covers old + new rows. `none` baselines are NOT relaunched (must be unaffected).
#
# REVIEW, then run. Every line is a cluv submit; nothing runs until you invoke it.
# GATE: run the SMOKE block first; launch P0/P1 only after
#   python scripts/check_pis_smoke.py <kicker run>/depth_per_image_scale_log.json --bias_fit ...
# passes and the none re-run diffs clean against the existing kicker none baseline.
set -e

GAP=0.10
S=scripts/unified_eth3d_depthpro_sweep.sh
TABLE=eth3d_table        # shared parent root for all conditions (aggregate over this)

SEQUENCES="kicker delivery_area pipes relief relief_2 facade terrace terrains"
DEPTHPRO=depth_pro_760
UNIDEPTH=unidepthv2_760
GT=gt_depth_mesh_760

# ── SMOKE (spec §8, DONE 2026-07-05): kicker rs_pis + none re-run + attribution/ablation cells ──
# cluv submit mila $S -- kicker unimodal_rs_pis $GAP $TABLE/depthpro $DEPTHPRO true false false
# cluv submit mila $S -- kicker none $GAP $TABLE/none_recheck $DEPTHPRO true false false
# cluv submit mila $S -- kicker bimodal_gmm_rs_pis $GAP $TABLE/depthpro $DEPTHPRO true false false
# cluv submit mila $S -- kicker unimodal_rs $GAP $TABLE/depthpro $DEPTHPRO true false false
# cluv submit mila $S -- kicker unimodal_rs_pis_huber $GAP $TABLE/depthpro $DEPTHPRO true false false
# cluv submit mila $S -- pipes unimodal_rs_pis_huber $GAP $TABLE/depthpro $DEPTHPRO true false false
# cluv submit mila $S -- kicker unimodal_rs_pis $GAP $TABLE/depthpro_oracle_gate $DEPTHPRO false true false
# cluv submit mila $S -- kicker unimodal_rs_pis_r10 $GAP $TABLE/depthpro $DEPTHPRO true false false

# ── MAIN (2026-07-05): rs_pis_huber = the keeper config. 8 scenes x {depthpro, unidepthv2} x
# {plain, oracle_gate} x {unimodal, bimodal_gmm}. Skips the 2 cells already run above. ──
for SEQ in $SEQUENCES; do
    for M in unimodal_rs_pis_huber bimodal_gmm_rs_pis_huber; do
        # already-run smoke cells (identical config) — don't resubmit
        if [ "$M" = "unimodal_rs_pis_huber" ] && { [ "$SEQ" = "kicker" ] || [ "$SEQ" = "pipes" ]; }; then :; else
            cluv submit mila $S -- $SEQ $M $GAP $TABLE/depthpro $DEPTHPRO true false false
        fi
        cluv submit mila $S -- $SEQ $M $GAP $TABLE/unidepthv2 $UNIDEPTH true false false
        cluv submit mila $S -- $SEQ $M $GAP $TABLE/depthpro_oracle_gate $DEPTHPRO false true false
        cluv submit mila $S -- $SEQ $M $GAP $TABLE/unidepthv2_oracle_gate $UNIDEPTH false true false
    done
    # metric-vs-shape control (GT depth through the same machinery) — enable if wanted:
    # cluv submit mila $S -- $SEQ unimodal_rs_pis_huber $GAP $TABLE/gt $GT true false false
done

# Aggregate (pis_* columns land in all_metrics.csv):
#   python scripts/aggregate_modes.py --root "$SCRATCH/logs/sweeps/$TABLE" --latex --out_dir tables/eth3d
