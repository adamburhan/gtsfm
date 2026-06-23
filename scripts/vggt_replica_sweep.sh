#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100l:1
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
# Sweep root groups runs as <root>/<seq>/<mode> so `aggregate_modes.py --root` discovers
# them directly. Gap is folded into the default root name (one gap per aggregatable root,
# since the aggregator has no gap dimension); override SWEEP_NAME to share a root across
# manual submissions.
SWEEP=${SWEEP_NAME:-replica_g${GAPTHRESH}}
OUT="$SCRATCH/logs/sweeps/${SWEEP}/${SEQ}/${MODE}"

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

echo "=== Curating outputs (keep metrics + reconstruction + figures; drop multi-GB artifacts) ==="
# GS checkpoints/ply/renders are ~850MB/run and regenerable; keep only the NVS stats JSON
# that aggregation reads. Set KEEP_GS_ARTIFACTS=1 to retain them (e.g. for paper figures).
if [ "${KEEP_GS_ARTIFACTS:-0}" != "1" ]; then
    find "$OUT/gs" -mindepth 1 -maxdepth 1 ! -name stats -exec rm -rf {} + 2>/dev/null || true
fi
# Debug image dumps (not used downstream).
rm -rf "$OUT/results/processed_images"
# results/depth.npz (~18MB) is kept by default: it lets you re-run eval_geometry at other
# gap_thresh values without re-running VGGT. Set DROP_DEPTH_NPZ=1 to remove it.
[ "${DROP_DEPTH_NPZ:-0}" = "1" ] && rm -f "$OUT/results/depth.npz"

echo "Done. Results in $OUT"
