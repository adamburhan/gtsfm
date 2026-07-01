#!/bin/bash
# Launch the classical ETH3D depth-factor sweep for the diagnostic table (GT + DepthPro backbone).
# Knobs are POSITIONAL args to the sweep (cluv submit does not forward env vars):
#   <seq> <mode> <gap> <sweep_name> <depth_subdir> <gt_scale> <gt_gate> <auto_scale>
# Each condition writes under TABLE/<condition>/<seq>/<mode>, so a single aggregate --root TABLE
# tables them all together (source column = the condition).
#
# REVIEW, then run. Every line is a cluv submit; nothing runs until you invoke it.
# Prereqs per scene under $SCRATCH/datasets/eth3d/<seq>/ :
#   depth_pro_760/   gt_depth_mesh_760/ (render_gt_depth_mesh.py)   occlusion/surface_mesh.ply
set -e

GAP=0.10
S=scripts/unified_eth3d_depthpro_sweep.sh
TABLE=eth3d_table        # shared parent root for all conditions (aggregate over this)
SEQUENCES="kicker courtyard delivery_area electro facade meadow office terrace pipes playground relief relief_2 terrains"
DEPTHPRO=depth_pro_760
GT=gt_depth_mesh_760

for SEQ in $SEQUENCES; do
    # seq mode gap sweep_name depth_subdir gt_scale gt_gate auto_scale
    # 1) GT oracle (perfect-depth upper bound): GT Sim(3) scale, no gate
    cluv submit mila $S -- $SEQ none $GAP $TABLE/gt $GT true false false
    cluv submit mila $S -- $SEQ unimodal $GAP $TABLE/gt $GT true false false
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP $TABLE/gt $GT true false false

    # 2) DepthPro: GT Sim(3) scale, no gate (scale-controlled real depth)
    cluv submit mila $S -- $SEQ none $GAP $TABLE/depthpro $DEPTHPRO true false false
    cluv submit mila $S -- $SEQ unimodal $GAP $TABLE/depthpro $DEPTHPRO true false false
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP $TABLE/depthpro $DEPTHPRO true false false

    # 3) DepthPro + oracle GT gate (gate supplies the scale)
    cluv submit mila $S -- $SEQ unimodal $GAP $TABLE/depthpro_oracle_gate $DEPTHPRO false true false
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP $TABLE/depthpro_oracle_gate $DEPTHPRO false true false

    # 4) DepthPro deployable: point-ratio auto_scale, no gate/GT (isolates the scale-collapse finding)
    cluv submit mila $S -- $SEQ unimodal $GAP $TABLE/depthpro_deploy $DEPTHPRO false false true
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP $TABLE/depthpro_deploy $DEPTHPRO false false true
done

# 20 jobs: 10 conditions x 2 scenes. Aggregate ALL into one table after they finish:
#   python scripts/aggregate_modes.py --root "$SCRATCH/logs/sweeps/$TABLE" --latex --out_dir tables/eth3d
