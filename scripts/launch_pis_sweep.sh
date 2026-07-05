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

# ── SMOKE (spec §8): one kicker job + a none re-run for the bit-identity check ──
cluv submit mila $S -- kicker unimodal_rs_pis $GAP $TABLE/depthpro $DEPTHPRO true false false
cluv submit mila $S -- kicker none $GAP $TABLE/none_recheck $DEPTHPRO true false false

# ── P0 (after smoke passes): {depthpro, unidepthv2} x {unimodal, bimodal_gmm} x _rs_pis, 8 scenes ──
# GT Sim(3) scale, no gate — same config as the existing depthpro/unidepthv2 table blocks.
# for SEQ in $SEQUENCES; do
#     # seq mode gap sweep_name depth_subdir gt_scale gt_gate auto_scale
#     cluv submit mila $S -- $SEQ unimodal_rs_pis $GAP $TABLE/depthpro $DEPTHPRO true false false
#     cluv submit mila $S -- $SEQ bimodal_gmm_rs_pis $GAP $TABLE/depthpro $DEPTHPRO true false false
#     cluv submit mila $S -- $SEQ unimodal_rs_pis $GAP $TABLE/unidepthv2 $UNIDEPTH true false false
#     cluv submit mila $S -- $SEQ bimodal_gmm_rs_pis $GAP $TABLE/unidepthv2 $UNIDEPTH true false false
#     # metric-vs-shape control: GT depth through the same machinery
#     cluv submit mila $S -- $SEQ unimodal_rs_pis $GAP $TABLE/gt $GT true false false
# done

# ── P1 (attribution: relative sigma alone): depthpro x {unimodal, bimodal_gmm} x _rs, kicker+pipes ──
# for SEQ in kicker pipes; do
#     cluv submit mila $S -- $SEQ unimodal_rs $GAP $TABLE/depthpro $DEPTHPRO true false false
#     cluv submit mila $S -- $SEQ bimodal_gmm_rs $GAP $TABLE/depthpro $DEPTHPRO true false false
# done

# Aggregate (pis_* columns land in all_metrics.csv):
#   python scripts/aggregate_modes.py --root "$SCRATCH/logs/sweeps/$TABLE" --latex --out_dir tables/eth3d
