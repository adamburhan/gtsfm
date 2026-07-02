#!/bin/bash
# Sparse-view crossover experiment on ONE sequence (kicker).
# Hypothesis: as views decrease, no-depth BA degrades faster than depth-aware BA; if the curves cross
# at 8/5/3 views, the depth prior is useful in the sparse-view / weak-triangulation regime.
#
# 4 methods x 5 view counts = 20 jobs. Views are a nested, evenly-spread subset (loader.num_views,
# farthest-point order) so smaller counts are subsets of larger ones and share views across methods.
# Output: <TABLE>/<method>/v<NN>/kicker/<mode>  (plot_sparse_view.py reads method + count from the path).
#
# REVIEW, then run. Every line is a cluv submit; nothing runs until you invoke it.
set -e

GAP=0.10
S=scripts/unified_eth3d_depthpro_sweep.sh
TABLE=kicker_sparse           # parent root for this experiment
SEQ=kicker
GT=gt_depth_mesh_760
UNI=unidepthv2_760
COUNTS="full 16 8 5 3"        # full = all registered views (loader.num_views unset)

for C in $COUNTS; do
    NV=$([ "$C" = "full" ] && echo "" || echo "$C")   # empty -> all views
    V=$([ "$C" = "full" ] && echo "vfull" || printf "v%02d" "$C")

    # method            seq   mode         gap   sweep_name         depth_subdir gt_scale gt_gate auto  num_views
    # 1) none (no depth prior; scale flags irrelevant in none mode)
    cluv submit mila $S -- $SEQ none         $GAP $TABLE/none/$V           $GT  true  false false "$NV"
    # 2) GT depth (perfect-depth reference)
    cluv submit mila $S -- $SEQ bimodal_gmm  $GAP $TABLE/gt/$V             $GT  true  false false "$NV"
    # 3) UniDepthV2 depth, GT Sim(3) scale, no gate
    cluv submit mila $S -- $SEQ bimodal_gmm  $GAP $TABLE/unidepthv2/$V     $UNI true  false false "$NV"
    # 4) UniDepthV2 + oracle GT gate (gate supplies the scale)
    cluv submit mila $S -- $SEQ bimodal_gmm  $GAP $TABLE/unidepthv2_oracle_gate/$V $UNI false true  false "$NV"
done

# After all finish, plot acc50 / acc95 / AUC@5 vs view count:
#   python scripts/plot_sparse_view.py --root "$SCRATCH/logs/sweeps/$TABLE" --out_dir tables/kicker_sparse
