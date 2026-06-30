#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=06:00:00

# Classical (unified) GTSfM on ETH3D with Depth Pro metric depth factors.
#   1) GTSfM: SuperPoint+LightGlue front end, COLMAP loader, depth-aware cluster BA.
#   2) Geometry eval vs GT laser scan.
#
# Depth maps are Depth Pro per-image metric depth (.npy, meters), named <image_stem>.npy. They MUST
# be at the loader's working resolution (short side <= max_resolution); the depth factor is brought
# into the gauge-arbitrary classical-SfM scale at BA time via depth_auto_scale (robust global scale).
#
# Usage: cluv submit mila scripts/unified_eth3d_depthpro_sweep.sh -- <sequence> <mode> [gap_thresh]
#   e.g. cluv submit mila scripts/unified_eth3d_depthpro_sweep.sh -- kicker bimodal_gmm
#   mode: none | bimodal_gap | bimodal_gmm | bimodal_gmm_null

set -eo pipefail
module load cuda/12.6.0

SEQ=${1:?usage: unified_eth3d_depthpro_sweep.sh <sequence> <mode> [gap_thresh]}
MODE=${2:?mode: none | unimodal | bimodal_gap | bimodal_gmm | bimodal_gmm_null}
GAPTHRESH=${3:-0.10}
MAX_RES=${MAX_RES:-760}          # loader short-side cap; depth .npy must match this resolution
NULL_NSIGMA=${NULL_NSIGMA:-5}    # bimodal_gmm_null: opt out when best mode > N sigmas off
GT_POSES=${GT_POSES:-false}      # false = real from-scratch SfM (gauge-free); true = GT-anchored metric poses
AUTO_SCALE=${AUTO_SCALE:-true}   # reconcile metric depth with recon scale; needed iff GT_POSES=false
GT_GATE=${GT_GATE:-false}        # oracle diagnostic: drop depth factors whose modes all miss the GT surface
GT_TAU=${GT_TAU:-0.1}           # gate band (m): a mode this close to GT counts as valid
GT_ORACLE=${GT_ORACLE:-false}    # also collapse to the GT-closest mode (mode-selection ceiling)
PATCH_RADIUS=${PATCH_RADIUS:-3}  # half-size of the patch for gap/GMM ambiguity analysis

project_name="gtsfm"
project_root="$HOME/repos/$project_name"
DATA="$SCRATCH/datasets/eth3d"
SWEEP=${SWEEP_NAME:-eth3d_unified_depthpro_g${GAPTHRESH}}
OUT="$SCRATCH/logs/sweeps/${SWEEP}/${SEQ}/${MODE}"
GT="$DATA/$SEQ/${SEQ}_gt.ply"
COLMAP_DIR="$DATA/$SEQ/dslr_calibration_undistorted"
IMAGES_DIR="$DATA/$SEQ/images"
DEPTH_DIR="$DATA/$SEQ/${DEPTH_SUBDIR:-gt_depth_760}"   # set DEPTH_SUBDIR=gt_depth_760 for the GT-depth-as-source oracle

echo "GIT_COMMIT=${GIT_COMMIT:?GIT_COMMIT is not set. Use 'cluv submit' to submit this job script.}"
cd $SLURM_TMPDIR
git clone $project_root
cd $project_name
git checkout --detach $GIT_COMMIT
cp -r $project_root/.venv .venv
git submodule update --init thirdparty/LightGlue

# Copy model weights (untracked by git): SuperPoint/SuperGlue, hloc (NetVLAD), LightGlue.
for w in SuperGluePretrainedNetwork/models/weights hloc/weights; do
    rsync -a $project_root/thirdparty/$w/ thirdparty/$w/ 2>/dev/null || true
done
uv sync
uv pip install -e thirdparty/LightGlue/ --no-deps

mkdir -p $OUT

# Build the GT point cloud once per scene (shared on $SCRATCH); atomic to avoid concurrent clobber.
if [ ! -f "$GT" ]; then
    echo "=== [0] Building GT cloud from scans -> $GT ==="
    TMP="$DATA/$SEQ/.${SEQ}_gt.tmp.$SLURM_JOB_ID.ply"
    uv run python scripts/eth3d_merge_scans.py \
        "$DATA/$SEQ/scan_clean/scan_alignment.mlp" --out "$TMP" --voxel 0.01
    mv -n "$TMP" "$GT" || true
    rm -f "$TMP"
fi

BA="cluster_optimizer.multiview_optimizer.bundle_adjustment_module"

# Depth source shared by all bimodal modes. template=null mirrors the exact image stem with depth_ext
# (DSC_0675.JPG -> DSC_0675.npy), preserving leading zeros (an integer template would drop them).
# depth_scale=1 (float meters); depth_auto_scale reconciles metric depth with the classical-SfM scale.
DEPTH_COMMON="$BA.depth_map_dir=$DEPTH_DIR \
    $BA.depth_filename_template=null $BA.depth_ext=.npy \
    $BA.depth_scale=1.0 $BA.depth_auto_scale=$AUTO_SCALE \
    $BA.depth_min=0.1 $BA.depth_max=100.0 $BA.depth_gap_thresh=$GAPTHRESH \
    $BA.depth_patch_radius=$PATCH_RADIUS \
    $BA.depth_gt_gate=$GT_GATE $BA.depth_gt_ply=$GT $BA.depth_gt_align_ref=$COLMAP_DIR \
    $BA.depth_gt_tau=$GT_TAU $BA.depth_gt_oracle_select=$GT_ORACLE"

DEPTH_ARGS=""
case "$MODE" in
    none) DEPTH_ARGS="$BA.depth_model=none" ;;
    unimodal) DEPTH_ARGS="$BA.depth_model=unimodal $DEPTH_COMMON" ;;  # single depth factor, no patch/GMM (use for clean GT-depth oracle)
    bimodal_gap) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gap $DEPTH_COMMON" ;;
    bimodal_gmm) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gmm $DEPTH_COMMON" ;;
    bimodal_gmm_null) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gmm \
        $BA.depth_null_nsigma=$NULL_NSIGMA $DEPTH_COMMON" ;;
    *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

echo "=== [1/2] GTSfM (unified/classical): seq=$SEQ mode=$MODE gap=$GAPTHRESH max_res=$MAX_RES gt_poses=$GT_POSES auto_scale=$AUTO_SCALE gt_gate=$GT_GATE oracle=$GT_ORACLE ==="
uv run python -m gtsfm.runner \
    --config_name unified.yaml \
    --loader colmap \
    --dataset_dir $COLMAP_DIR \
    --images_dir $IMAGES_DIR \
    --max_resolution $MAX_RES \
    --output_root $OUT \
    --dask_tmpdir $SLURM_TMPDIR \
    loader.use_gt_extrinsics=$GT_POSES \
    $DEPTH_ARGS

# The single-cluster classical reconstruction is written under results/. Prefer the merged scene if
# the scene partitioned, else the per-cluster output.
if [ -d "$OUT/results/merged" ]; then
    SFM="$OUT/results/merged"
else
    SFM="$OUT/results/ba_output"
fi

echo "=== [2/2] Geometry eval vs GT scan ($SFM) ==="
uv run python gtsfm/evaluation/eval_geometry.py \
    --sfm_output $SFM \
    --align_mode eth3d \
    --align_ref $COLMAP_DIR \
    --gt_ply $GT \
    --tau 0.01 0.02 0.05 0.1 0.2 0.5 \
    --out $OUT/geometry_metrics.json

echo "Done. Results in $OUT"
