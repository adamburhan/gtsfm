#!/bin/bash
# VGGT Replica depth-factor run on a local/remote workstation (no SLURM, no cluv).
# Runs in-place in the current checkout; assumes submodules + model weights are already
# set up and the .venv exists (uv sync done once).
#
#   1) GTSfM (VGGT front-end, depth-aware cluster BA)
#   2) Geometry eval vs GT mesh
#   3) Gaussian splatting + NVS eval
#
# Usage:
#   DATA=/path/to/Replica OUT=/path/to/out \
#     bash scripts/vggt_replica_local.sh <sequence> <depth_model> [stride] [gs_max_steps] [gap_thresh]
#   e.g. DATA=~/data/Replica OUT=~/runs bash scripts/vggt_replica_local.sh office0 bimodal

set -eo pipefail

SEQ=${1:?usage: vggt_replica_local.sh <sequence> <depth_model> [stride] [gs_max_steps] [gap_thresh]}
MODE=${2:?depth_model: none | unimodal | drop_ambiguous | bimodal}
STRIDE=${3:-10}
GS_STEPS=${4:-7000}
GAPTHRESH=${5:-0.15}
HMETHOD=${HMETHOD:-gap}   # ambiguity analysis: gap (largest-gap) | gmm (2-component GMM)

project_name="gtsfm"
project_root="$HOME/repos/$project_name"
DATA="$SCRATCH/datasets/replica/Replica"
OUT="$SCRATCH/logs/cluv/$SLURM_JOB_ID/${SEQ}_${MODE}_${GAPTHRESH}"
mkdir -p $OUT

BA="cluster_optimizer.optimizer.ba_options"

echo "=== [1/3] GTSfM VGGT: seq=$SEQ depth_model=$MODE stride=$STRIDE ==="
uv run python -m gtsfm.runner \
    --config_name vggt_megaloc_replica.yaml \
    --output_root $OUT \
    loader.dataset_dir=$DATA \
    loader.sequence=$SEQ \
    loader.stride=$STRIDE \
    $BA.depth_model=$MODE \
    $BA.depth_hypothesis_method=$HMETHOD \
    $BA.depth_min=0.0 \
    $BA.depth_max=1e9 \
    $BA.depth_gap_thresh=$GAPTHRESH

echo "=== [2/3] Geometry eval vs GT mesh ==="
uv run python gtsfm/evaluation/eval_geometry.py \
    --sfm_output $OUT/results/merged \
    --align_mode replica \
    --align_ref $DATA/${SEQ}/traj.txt \
    --gt_ply $DATA/${SEQ}_mesh.ply \
    --out $OUT/geometry_metrics.json

echo "=== [3/3] Gaussian splatting + NVS eval ==="
uv run python scripts/gaussian_splatting/custom_trainer.py default \
    --data_dir $OUT/results/merged \
    --images_dir $DATA/$SEQ/results \
    --init_type sfm \
    --max_steps $GS_STEPS \
    --result_dir $OUT/gs

echo "Done. Results in $OUT"
