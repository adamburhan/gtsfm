#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --time=06:00:00

# VGGT ETH3D depth-factor sweep: one (sequence, depth_model) per job.
#   1) GTSfM (VGGT front-end, depth-aware cluster BA; GNC disabled for depth CustomFactors)
#   2) Geometry eval vs GT laser scan (accuracy / precision; + mode/ambiguous when depth exists)
#   3) Gaussian splatting + NVS eval (PSNR/SSIM/LPIPS on held-out views)
#
# Usage: cluv submit mila scripts/vggt_eth3d_pred_sweep.sh -- <sequence> <depth_model> [gs_max_steps] [gap_thresh]
#   e.g. cluv submit mila scripts/vggt_eth3d_pred_sweep.sh -- kicker bimodal

set -eo pipefail

module load cuda/12.6.0

SEQ=${1:?usage: vggt_eth3d_pred_sweep.sh <sequence> <depth_model> [gs_max_steps] [gap_thresh]}
MODE=${2:?depth_model: none | unimodal | drop_ambiguous | bimodal}
GS_STEPS=${3:-7000}
GAPTHRESH=${4:-0.10}

project_name="gtsfm"
project_root="$HOME/repos/$project_name"
DATA="$SCRATCH/datasets/eth3d"
OUT="$SCRATCH/logs/cluv/$SLURM_JOB_ID/${SEQ}_${MODE}_${GAPTHRESH}"
GT="$DATA/$SEQ/${SEQ}_gt.ply"

echo "GIT_COMMIT=${GIT_COMMIT:?GIT_COMMIT is not set. Use 'cluv submit' to submit this job script.}"
cd $SLURM_TMPDIR
git clone $project_root
cd $project_name
git checkout --detach $GIT_COMMIT
cp -r $project_root/.venv .venv
git submodule update --init thirdparty/vggt thirdparty/LightGlue

# Copy model weights (untracked by git).
for w in SuperGluePretrainedNetwork/models/weights hloc/weights vggt/weights; do
    rsync -a $project_root/thirdparty/$w/ thirdparty/$w/
done
uv sync
uv pip install -e thirdparty/vggt/ --no-deps
uv pip install -e thirdparty/LightGlue/ --no-deps

mkdir -p $OUT

# Build the GT point cloud once per scene (shared on $SCRATCH). Atomic so concurrent
# mode jobs for the same scene don't clobber each other.
if [ ! -f "$GT" ]; then
    echo "=== [0] Building GT cloud from scans -> $GT ==="
    # Temp must keep a .ply extension (Open3D infers the writer from it); atomic mv so
    # concurrent mode jobs for the same scene don't clobber each other.
    TMP="$DATA/$SEQ/.${SEQ}_gt.tmp.$SLURM_JOB_ID.ply"
    uv run python scripts/eth3d_merge_scans.py \
        "$DATA/$SEQ/scan_clean/scan_alignment.mlp" --out "$TMP" --voxel 0.01
    mv -n "$TMP" "$GT" || true
    rm -f "$TMP"
fi

BA="cluster_optimizer.optimizer.ba_options"

echo "=== [1/3] GTSfM VGGT: seq=$SEQ depth_model=$MODE gap_thresh=$GAPTHRESH ==="
uv run python -m gtsfm.runner \
    --config_name vggt_megaloc_eth3d.yaml \
    --output_root $OUT \
    --dask_tmpdir $SLURM_TMPDIR \
    --dataset_dir=$DATA/$SEQ/dslr_calibration_undistorted \
    --images_dir=$DATA/$SEQ/images \
    $BA.depth_model=$MODE \
    $BA.use_gnc=false \
    $BA.depth_min=0.0 \
    $BA.depth_max=1e9 \
    $BA.depth_gap_thresh=$GAPTHRESH

# Pick the reconstruction to evaluate: merged scene if the scene partitioned
# (>1 cluster), otherwise the single-cluster vggt output. depth.npz (and thus the
# mode/ambiguous metrics) is only well-defined for the single-cluster case.
if [ -d "$OUT/results/merged" ]; then
    SFM="$OUT/results/merged"
    DEPTH=""
    echo "Multi-cluster scene: evaluating merged/ (mode metrics need per-node eval; skipped here)."
else
    SFM="$OUT/results/vggt"
    DEPTH=""
    if [ "$MODE" != "none" ] && [ -f "$OUT/results/depth.npz" ]; then
        DEPTH="--depth_npz $OUT/results/depth.npz --gap_thresh $GAPTHRESH"
    fi
fi

echo "=== [2/3] Geometry eval vs GT scan ($SFM) ==="
uv run python gtsfm/evaluation/eval_geometry.py \
    --sfm_output $SFM \
    $DEPTH \
    --align_mode eth3d \
    --align_ref $DATA/$SEQ/dslr_calibration_undistorted \
    --gt_ply $GT \
    --tau 0.01 0.02 0.05 0.1 0.2 0.5 \
    --out $OUT/geometry_metrics.json

echo "=== [3/3] Gaussian splatting + NVS eval ==="
uv run python scripts/gaussian_splatting/custom_trainer.py default \
    --data_dir $SFM \
    --images_dir $DATA/$SEQ/images \
    --init_type sfm \
    --max_steps $GS_STEPS \
    --result_dir $OUT/gs

echo "Done. Results in $OUT"
