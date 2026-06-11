#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Replica depth-factor sweep: one (sequence, depth_model) per job.
#   1) GTSfM (SIFT front-end, depth-aware global BA)
#   2) Geometry eval vs GT mesh (T&T precision/recall/F-score)
#   3) Gaussian splatting from ba_output + NVS eval (PSNR/SSIM/LPIPS on held-out views)
#
# Usage: cluv submit mila scripts/replica_sweep.sh -- <sequence> <depth_model> [stride] [gs_max_steps]
#   e.g. cluv submit mila scripts/replica_sweep.sh -- office0 bimodal

set -eo pipefail

SEQ=${1:?usage: replica_sweep.sh <sequence> <depth_model> [stride] [gs_max_steps]}
MODE=${2:?depth_model: none | unimodal | drop_ambiguous | bimodal}
STRIDE=${3:-10}
GS_STEPS=${4:-7000}

project_name="gtsfm"
project_root="$HOME/repos/$project_name"
DATA="$SCRATCH/datasets/replica/Replica"
OUT="$SCRATCH/logs/cluv/$SLURM_JOB_ID/${SEQ}_${MODE}"

# Pin the code at the submitted commit: the repo in $HOME may change while jobs sit in queue.
echo "GIT_COMMIT=${GIT_COMMIT:?GIT_COMMIT is not set. Use 'cluv submit' to submit this job script.}"
cd $SLURM_TMPDIR
git clone $project_root
cd $project_name
git checkout --detach $GIT_COMMIT
cp -r $project_root/.venv .venv
uv sync

mkdir -p $OUT
BA=cluster_optimizer.multiview_optimizer.bundle_adjustment_module

echo "=== [1/3] GTSfM: seq=$SEQ depth_model=$MODE stride=$STRIDE ==="
uv run python -m gtsfm.runner \
    --config_name unified.yaml \
    --correspondence_generator_config_name sift.yaml \
    --loader replica \
    --dataset_dir $DATA \
    --max_resolution 760 \
    --output_root $OUT \
    loader.sequence=$SEQ \
    loader.stride=$STRIDE \
    $BA.depth_model=$MODE \
    $BA.depth_map_dir=$DATA/$SEQ/results

echo "=== [2/3] Geometry eval vs GT mesh ==="
uv run python gtsfm/evaluation/eval_geometry_vs_mesh.py \
    --ba_dir $OUT/results/ba_output \
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
