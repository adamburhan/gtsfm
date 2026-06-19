#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --time=06:00:00

# VGGT Replica depth-factor sweep: one (sequence, depth_model) per job.
#   1) GTSfM (VGGT front-end, depth-aware cluster BA)
#   2) Geometry eval vs GT mesh (T&T precision/recall/F-score)
#   3) Gaussian splatting from ba_output + NVS eval (PSNR/SSIM/LPIPS on held-out views)
#
# Usage: cluv submit mila scripts/vggt_replica_sweep.sh -- <sequence> <depth_model> [stride] [gs_max_steps]
#   e.g. cluv submit mila scripts/vggt_replica_sweep.sh -- office0 bimodal

set -eo pipefail

module load cuda/12.6.0

SEQ=${1:?usage: vggt_replica_sweep.sh <sequence> <depth_model> [stride] [gs_max_steps] [gap_thresh]}
MODE=${2:?depth_model: none | unimodal | drop_ambiguous | bimodal}
STRIDE=${3:-10}
GS_STEPS=${4:-7000}
GAPTHRESH=${5:-0.15}

project_name="gtsfm"
project_root="$HOME/repos/$project_name"
DATA="$SCRATCH/datasets/replica/Replica"
OUT="$SCRATCH/logs/cluv/$SLURM_JOB_ID/${SEQ}_${MODE}_${GAPTHRESH}"

echo "GIT_COMMIT=${GIT_COMMIT:?GIT_COMMIT is not set. Use 'cluv submit' to submit this job script.}"
cd $SLURM_TMPDIR
git clone $project_root
cd $project_name
git checkout --detach $GIT_COMMIT
cp -r $project_root/.venv .venv

# Copy model weights (untracked by git).
for w in SuperGluePretrainedNetwork/models/weights hloc/weights vggt/weights; do
    rsync -a $project_root/thirdparty/$w/ thirdparty/$w/
done
uv sync

mkdir -p $OUT

BA="cluster_optimizer.optimizer.ba_options"

echo "=== [1/3] GTSfM VGGT: seq=$SEQ depth_model=$MODE stride=$STRIDE ==="
uv run python -m gtsfm.runner \
    --config_name vggt_megaloc_replica.yaml \
    --output_root $OUT \
    --dask_tmpdir $SLURM_TMPDIR \
    loader.dataset_dir=$DATA \
    loader.sequence=$SEQ \
    loader.stride=$STRIDE \
    $BA.depth_model=$MODE \
    $BA.depth_min=0.0 \
    $BA.depth_max=1e9 \
    $BA.depth_gap_thresh=$GAPTHRESH

echo "=== [2/3] Geometry eval vs GT mesh ==="
uv run python gtsfm/evaluation/eval_geometry_vs_mesh.py \
    --ba_dir $OUT/results/ba_output \
    --gt_traj $DATA/$SEQ/traj.txt \
    --gt_ply $DATA/${SEQ}_mesh.ply \
    --out $OUT/geometry_metrics.json

echo "=== [3/3] Gaussian splatting + NVS eval ==="
uv run python scripts/gaussian_splatting/custom_trainer.py default \
    --data_dir $OUT/results/ba_output \
    --images_dir $DATA/$SEQ/results \
    --init_type sfm \
    --max_steps $GS_STEPS \
    --result_dir $OUT/gs

echo "Done. Results in $OUT"
