#!/bin/bash
# Launch the classical ETH3D depth-factor sweep for the diagnostic table (GT + DepthPro backbone).
# Knobs are POSITIONAL args to the sweep (cluv submit does not forward env vars):
#   <seq> <mode> <gap> <sweep_name> <depth_subdir> <gt_scale> <gt_gate> <auto_scale>
# Each source/gate condition uses its own <sweep_name> root (the sweep output is keyed only by
# <seq>/<mode>); aggregate each root separately (see the tail).
#
# REVIEW, then run. Every line is a cluv submit; nothing runs until you invoke it.
# Prereqs per scene under $SCRATCH/datasets/eth3d/<seq>/ :
#   depth_pro_760/   gt_depth_mesh_760/ (render_gt_depth_mesh.py)   occlusion/surface_mesh.ply
set -e

GAP=0.10
S=scripts/unified_eth3d_depthpro_sweep.sh
SEQUENCES="kicker delivery_area"
DEPTHPRO=depth_pro_760
GT=gt_depth_mesh_760

for SEQ in $SEQUENCES; do
    # seq mode gap sweep_name depth_subdir gt_scale gt_gate auto_scale
    # 1) GT oracle (perfect-depth upper bound): GT Sim(3) scale, no gate
    cluv submit mila $S -- $SEQ none $GAP gt $GT true false false
    cluv submit mila $S -- $SEQ unimodal $GAP gt $GT true false false
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP gt $GT true false false

    # 2) DepthPro: GT Sim(3) scale, no gate (scale-controlled real depth)
    cluv submit mila $S -- $SEQ none $GAP depthpro $DEPTHPRO true false false
    cluv submit mila $S -- $SEQ unimodal $GAP depthpro $DEPTHPRO true false false
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP depthpro $DEPTHPRO true false false

    # 3) DepthPro + oracle GT gate (gate supplies the scale)
    cluv submit mila $S -- $SEQ unimodal $GAP depthpro_oracle_gate $DEPTHPRO false true false
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP depthpro_oracle_gate $DEPTHPRO false true false

    # 4) DepthPro deployable: point-ratio auto_scale, no gate/GT (isolates the scale-collapse finding)
    cluv submit mila $S -- $SEQ unimodal $GAP depthpro_deploy $DEPTHPRO false false true
    cluv submit mila $S -- $SEQ bimodal_gmm $GAP depthpro_deploy $DEPTHPRO false false true
done

# 20 jobs: 10 conditions x 2 scenes. Aggregate after they finish (one table per source root):
#   for R in mtg_gt mtg_depthpro mtg_depthpro_gate mtg_depthpro_deploy; do
#     python scripts/aggregate_modes.py --root "$SCRATCH/logs/sweeps/$R" --out_dir "tables/$R"
#   done
