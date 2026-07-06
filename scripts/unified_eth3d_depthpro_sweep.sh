#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=06:00:00


set -eo pipefail
module load cuda/12.6.0
export PYTHONHASHSEED=0   # deterministic set/dict-hash ordering (must be set before python starts)

SEQ=${1:?usage: unified_eth3d_depthpro_sweep.sh <seq> <mode> <gap> <sweep_name> <depth_subdir> <gt_scale> <gt_gate> <auto_scale>}
MODE=${2:?mode: none | unimodal | bimodal_gap | bimodal_gmm | bimodal_gmm_null | unimodal_log | bimodal_log}
GAPTHRESH=${3:-0.10}
# Args 4-8 are the knobs that vary across the table. cluv submit does not forward env vars to the
# job, so they are passed positionally. The rest are fixed defaults below.
SWEEP_NAME=${4:-eth3d_unified_depthpro_g${GAPTHRESH}}
DEPTH_SUBDIR=${5:-depth_pro_760}   # depth_pro_760 | gt_depth_mesh_760 (GT-depth-as-source oracle)
GT_SCALE=${6:-false}               # fix sf to the GT Sim(3) scale (removes the auto_scale confound)
GT_GATE=${7:-false}                # oracle diagnostic: drop factors whose modes all miss the GT surface
AUTO_SCALE=${8:-true}              # point-ratio metric<->recon scale; used iff GT_SCALE/GT_GATE off
MAX_RES=760                        # loader short-side cap; depth .npy must match this resolution
NULL_NSIGMA=5                      # bimodal_gmm_null: opt out when best mode > N sigmas off
GT_POSES=false                     # false = real from-scratch SfM (gauge-free); true = GT-anchored poses
GT_TAU=0.1                         # gate band (m): a mode this close to GT counts as valid
GT_ORACLE=false                    # also collapse to the GT-closest mode (mode-selection ceiling)
PATCH_RADIUS=3                     # half-size of the patch for gap/GMM ambiguity analysis
ALPHA_SIGMA=1.0                    # *_log modes: prior sigma on alpha_i about the shared init scale
GS_STEPS=7000                      # Gaussian-splatting training steps for the NVS eval

project_name="gtsfm"
project_root="$HOME/repos/$project_name"
DATA="$SCRATCH/datasets/eth3d"
OUT="$SCRATCH/logs/sweeps/${SWEEP_NAME}/${SEQ}/${MODE}"
GT="$DATA/$SEQ/${SEQ}_gt.ply"
GT_MESH="$DATA/$SEQ/occlusion/surface_mesh.ply"   # ETH3D occlusion surface mesh (true point-to-surface gating)
COLMAP_DIR="$DATA/$SEQ/dslr_calibration_undistorted"
IMAGES_DIR="$DATA/$SEQ/images"
DEPTH_DIR="$DATA/$SEQ/${DEPTH_SUBDIR:-depth_pro_760}"   # set DEPTH_SUBDIR=gt_depth_760 for the GT-depth-as-source oracle

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
    $BA.depth_factor_robust_loss=true \
    $BA.depth_scale=1.0 $BA.depth_auto_scale=$AUTO_SCALE \
    $BA.depth_min=0.1 $BA.depth_max=100.0 $BA.depth_gap_thresh=$GAPTHRESH \
    $BA.depth_patch_radius=$PATCH_RADIUS \
    $BA.depth_gt_gate=$GT_GATE $BA.depth_gt_ply=$GT_MESH $BA.depth_gt_align_ref=$COLMAP_DIR \
    $BA.depth_gt_tau=$GT_TAU $BA.depth_gt_oracle_select=$GT_ORACLE $BA.depth_gt_scale=$GT_SCALE"

DEPTH_ARGS=""
case "$MODE" in
    none) DEPTH_ARGS="$BA.depth_model=none" ;;
    unimodal) DEPTH_ARGS="$BA.depth_model=unimodal $DEPTH_COMMON" ;;  # single depth factor, no patch/GMM (use for clean GT-depth oracle)
    bimodal_gap) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gap $DEPTH_COMMON" ;;
    bimodal_gmm) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gmm $DEPTH_COMMON" ;;
    bimodal_gmm_null) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gmm \
        $BA.depth_null_nsigma=$NULL_NSIGMA $DEPTH_COMMON" ;;
    # *_log: log-depth residuals with a free per-image scale offset alpha_i (structure-only depth).
    # Same metric depth_factor_sigma, converted per-measurement to relative (sigma/d). alpha init
    # subsumes auto_scale, so AUTO_SCALE only shifts the alphas (harmless).
    unimodal_log) DEPTH_ARGS="$BA.depth_model=unimodal $BA.depth_log_alpha=true \
        $BA.depth_alpha_sigma=$ALPHA_SIGMA $DEPTH_COMMON" ;;
    bimodal_log) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gmm $BA.depth_log_alpha=true \
        $BA.depth_alpha_sigma=$ALPHA_SIGMA $DEPTH_COMMON" ;;
    *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

echo "=== [1/3] GTSfM (unified/classical): seq=$SEQ mode=$MODE gap=$GAPTHRESH max_res=$MAX_RES gt_poses=$GT_POSES auto_scale=$AUTO_SCALE gt_gate=$GT_GATE oracle=$GT_ORACLE gt_scale=$GT_SCALE ==="
uv run python -m gtsfm.runner \
    --config_name unified.yaml \
    --correspondence_generator_config_name sift \
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

echo "=== [2/3] Geometry eval vs GT scan ($SFM) ==="
# GT scan point cloud (point-to-point), not the occlusion mesh (point-to-surface): the mesh is a
# reconstruction itself and its interpolated surface biases the accuracy numbers.
uv run python gtsfm/evaluation/eval_geometry.py \
    --sfm_output $SFM \
    --align_mode eth3d \
    --align_ref $COLMAP_DIR \
    --gt_ply "$GT" \
    --tau 0.01 0.02 0.05 0.1 0.2 0.5 \
    --out $OUT/geometry_metrics.json

echo "=== [3/3] Gaussian splatting + NVS eval (PSNR/SSIM/LPIPS on held-out views) ==="
uv run python scripts/gaussian_splatting/custom_trainer.py default \
    --data_dir $SFM \
    --images_dir $IMAGES_DIR \
    --init_type sfm \
    --max_steps $GS_STEPS \
    --result_dir $OUT/gs

# Keep only the NVS stats JSON (aggregate_modes reads gs/stats/val_step*.json); GS checkpoints/ply/
# renders are ~850MB/run and regenerable. Set KEEP_GS_ARTIFACTS=1 (edit here) to retain them.
if [ "${KEEP_GS_ARTIFACTS:-0}" != "1" ]; then
    find "$OUT/gs" -mindepth 1 -maxdepth 1 ! -name stats -exec rm -rf {} + 2>/dev/null || true
fi

echo "Done. Results in $OUT"
