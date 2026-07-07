#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=06:00:00


set -eo pipefail
module load cuda/12.6.0
export PYTHONHASHSEED=0   # deterministic set/dict-hash ordering (must be set before python starts)

SEQ=${1:?usage: unified_tt_sweep.sh <seq> <mode> <gap> <sweep_name> <depth_subdir> <auto_scale>}
MODE=${2:?mode: none | unimodal | bimodal_gap | bimodal_gmm | unimodal_log | bimodal_log | mda_unimodal_log | mda_multimodal4_log}
GAPTHRESH=${3:-0.10}
# Args 4-6 are the knobs that vary across the table. cluv submit does not forward env vars to the
# job, so they are passed positionally. The rest are fixed defaults below.
SWEEP_NAME=${4:-tt_unified_g${GAPTHRESH}}
DEPTH_SUBDIR=${5:-mda_depth_760}   # mda_depth_760 (npz mixtures) | depth_pro_760 (.npy, for non-mda modes)
AUTO_SCALE=${6:-true}              # point-ratio metric<->recon scale reconciliation
MAX_RES=760                        # loader short-side cap; depth npz/npy must match this resolution
PATCH_RADIUS=3                     # half-size of the patch for gap/GMM ambiguity analysis
ALPHA_SIGMA=1.0                    # *_log modes: prior sigma on alpha_i about the shared init scale
SIGMA_LOG=0.05                     # mda_* modes: shared log-space (relative) sigma for all components
GS_STEPS=7000                      # Gaussian-splatting training steps for the NVS eval

project_name="gtsfm"
project_root="$HOME/repos/$project_name"
DATA="$SCRATCH/datasets/tanks_and_temples"
OUT="$SCRATCH/logs/sweeps/${SWEEP_NAME}/${SEQ}/${MODE}"
SCENE="$DATA/$SEQ"
GT="$SCENE/${SEQ}.ply"             # LiDAR scan (metric GT geometry, in the LiDAR frame)
IMAGES_DIR="$SCENE/images"
DEPTH_DIR="$SCENE/$DEPTH_SUBDIR"

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

BA="cluster_optimizer.multiview_optimizer.bundle_adjustment_module"

# Depth source shared by the non-mda modes. template=null mirrors the exact image stem with depth_ext
# (000001.jpg -> 000001.npy). No GT gate/oracle knobs on T&T: they need a COLMAP GT dir + mesh.
DEPTH_COMMON="$BA.depth_map_dir=$DEPTH_DIR \
    $BA.depth_filename_template=null $BA.depth_ext=.npy \
    $BA.depth_factor_robust_loss=true \
    $BA.depth_scale=1.0 $BA.depth_auto_scale=$AUTO_SCALE \
    $BA.depth_min=0.1 $BA.depth_max=100.0 $BA.depth_gap_thresh=$GAPTHRESH \
    $BA.depth_patch_radius=$PATCH_RADIUS"

MDA_COMMON="$BA.depth_mda_npz_dir=$DEPTH_DIR \
    $BA.depth_log_alpha=true $BA.depth_alpha_sigma=$ALPHA_SIGMA \
    $BA.depth_factor_sigma_log=$SIGMA_LOG \
    $BA.depth_factor_robust_loss=true \
    $BA.depth_auto_scale=$AUTO_SCALE"

DEPTH_ARGS=""
case "$MODE" in
    none) DEPTH_ARGS="$BA.depth_model=none" ;;
    unimodal) DEPTH_ARGS="$BA.depth_model=unimodal $DEPTH_COMMON" ;;
    bimodal_gap) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gap $DEPTH_COMMON" ;;
    bimodal_gmm) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gmm $DEPTH_COMMON" ;;
    # *_log: log-depth residuals with a free per-image scale offset alpha_i (structure-only depth).
    unimodal_log) DEPTH_ARGS="$BA.depth_model=unimodal $BA.depth_log_alpha=true \
        $BA.depth_alpha_sigma=$ALPHA_SIGMA $DEPTH_COMMON" ;;
    bimodal_log) DEPTH_ARGS="$BA.depth_model=bimodal $BA.depth_hypothesis_method=gmm $BA.depth_log_alpha=true \
        $BA.depth_alpha_sigma=$ALPHA_SIGMA $DEPTH_COMMON" ;;
    # mda_*: raw MDA npz mixtures (DEPTH_SUBDIR = npz dir). Relative depth: no depth_map_dir, shared
    # log-space sigma for both conditions (uni-vs-multi differ only in the mode set).
    mda_unimodal_log) DEPTH_ARGS="$BA.depth_model=unimodal $MDA_COMMON" ;;
    mda_multimodal4_log) DEPTH_ARGS="$BA.depth_model=bimodal $MDA_COMMON" ;;
    *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

echo "=== [1/3] GTSfM (unified/classical): seq=$SEQ mode=$MODE gap=$GAPTHRESH max_res=$MAX_RES auto_scale=$AUTO_SCALE ==="
uv run python -m gtsfm.runner \
    --config_name unified.yaml \
    --correspondence_generator_config_name sift \
    --loader tanks_and_temples \
    --dataset_dir $SCENE \
    --max_resolution $MAX_RES \
    --share_intrinsics \
    --output_root $OUT \
    --dask_tmpdir $SLURM_TMPDIR \
    loader.poses_fpath=$SCENE/${SEQ}_COLMAP_SfM.log \
    loader.bounding_polyhedron_json_fpath=$SCENE/${SEQ}.json \
    loader.ply_alignment_fpath=$SCENE/${SEQ}_trans.txt \
    $DEPTH_ARGS

# The single-cluster classical reconstruction is written under results/. Prefer the merged scene if
# the scene partitioned, else the per-cluster output.
if [ -d "$OUT/results/merged" ]; then
    SFM="$OUT/results/merged"
else
    SFM="$OUT/results/ba_output"
fi

echo "=== [2/3] Geometry eval vs GT LiDAR scan ($SFM) ==="
# tnt mode: Sim(3)-fit recon cameras to the *_COLMAP_SfM.log GT poses, then *_trans.txt maps the
# COLMAP frame to the LiDAR (GT ply) frame.
uv run python gtsfm/evaluation/eval_geometry.py \
    --sfm_output $SFM \
    --align_mode tnt \
    --align_ref $SCENE \
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
