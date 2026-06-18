"""Aggregate Replica sweep results (sequence x depth_model) into one table.

Walks the cluv results directory for run dirs laid out as <root>/<job_id>/<seq>_<mode>/,
collecting per run:
  - geometry_metrics.json            (T&T mesh accuracy / precision / recall / F-score)
  - gs/stats/val_step*.json          (held-out PSNR / SSIM / LPIPS, latest step)
  - results/metrics/bundle_adjustment_metrics.json
                                     (pose AUC + median rotation / translation errors vs GT)

If a (sequence, mode) pair appears under multiple job ids (resubmissions), the highest
job id wins. Writes a CSV and prints pivot tables (mode x sequence) for the headline metrics.

Example:
    python scripts/aggregate_replica_sweep.py --results_root '$SCRATCH/logs/cluv' --out sweep.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

MODES = ("none", "unimodal", "drop_ambiguous", "bimodal")

GEOMETRY_KEYS = [
    "n_points",
    "accuracy_median_m",
    "accuracy_mean_m",
    "accuracy_p95_m",
    "precision@2.5cm",
    "precision@5cm",
    "recall@5cm",
    "fscore@2.5cm",
    "fscore@5cm",
]
POSE_AUC_KEYS = ["pose_auc_@1.0_deg", "pose_auc_@2.5_deg", "pose_auc_@5.0_deg"]
DEPTH_FACTOR_KEYS = ["unimodal", "bimodal", "dropped_ambiguous", "skipped"]
PIVOT_METRICS = [
    "fscore@5cm",
    "accuracy_median_m",
    "accuracy_p95_m",
    "pose_auc_@1.0_deg",
    "psnr",
    "lpips",
    "num_depth_factors_bimodal",
    "num_depth_factors_skipped",
]


def parse_run_dir_name(name: str):
    """Parse '<seq>_<mode>[_<gap_thresh>]' (seq has no underscore, modes may)."""
    seq, _, rest = name.partition("_")
    for mode in MODES:
        if rest == mode:
            return seq, mode, None
        if rest.startswith(mode + "_"):
            return seq, mode, rest[len(mode) + 1:]
    return None, None, None


def metric_median(value):
    """GtsfmMetric values are scalars or {'summary': {...}} distribution dicts."""
    if isinstance(value, dict):
        return value.get("summary", {}).get("median")
    return value


def collect_run(run_dir: Path) -> dict:
    row = {}

    geom_path = run_dir / "geometry_metrics.json"
    if geom_path.exists():
        geom = json.loads(geom_path.read_text())
        row.update({k: geom.get(k) for k in GEOMETRY_KEYS})

    val_jsons = sorted(
        run_dir.glob("gs/stats/val_step*.json"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    if val_jsons:
        gs = json.loads(val_jsons[-1].read_text())
        row.update({k: gs.get(k) for k in ("psnr", "ssim", "lpips", "num_GS")})

    ba_path = run_dir / "results/metrics/bundle_adjustment_metrics.json"
    if ba_path.exists():
        ba = json.loads(ba_path.read_text())["bundle_adjustment_metrics"]
        row.update({k: ba.get(k) for k in POSE_AUC_KEYS})
        row["rot_error_median_deg"] = metric_median(ba.get("rotation_angle_error_deg"))
        row["trans_error_median"] = metric_median(ba.get("translation_error_distance"))
        for stat in DEPTH_FACTOR_KEYS:
            value = metric_median(ba.get(f"num_depth_factors_{stat}"))
            if value is not None:
                row[f"num_depth_factors_{stat}"] = value

    for depth_path in run_dir.glob("results/**/depth_factor_metrics.json"):
        depth = json.loads(depth_path.read_text())["depth_factor_metrics"]
        for stat in DEPTH_FACTOR_KEYS:
            value = metric_median(depth.get(f"num_depth_factors_{stat}"))
            if value is not None:
                key = f"num_depth_factors_{stat}"
                row[key] = (row.get(key) or 0) + value

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results_root",
        default="$SCRATCH/logs/cluv",
        help="Local cluv results dir containing <job_id>/<seq>_<mode>/ run dirs. The default is the "
        "literal '$SCRATCH/logs/cluv' folder cluv creates in the repo on machines without $SCRATCH.",
    )
    parser.add_argument("--out", default="replica_sweep_results.csv", help="Output CSV path.")
    args = parser.parse_args()

    root = Path(args.results_root)
    runs = {}  # (seq, mode, gap_thresh) -> (job_id, row)
    for run_dir in sorted(root.glob("*/*/")):
        seq, mode, gap_thresh = parse_run_dir_name(run_dir.name)
        if seq is None:
            continue
        job_match = re.search(r"(\d+)", run_dir.parent.name)
        job_id = int(job_match.group(1)) if job_match else -1
        key = (seq, mode, gap_thresh)
        if key in runs and runs[key][0] > job_id:
            continue
        row = {"sequence": seq, "depth_model": mode, "gap_thresh": gap_thresh, "job_id": job_id, **collect_run(run_dir)}
        runs[key] = (job_id, row)

    if not runs:
        raise SystemExit(f"No '<seq>_<mode>[_<gap_thresh>]' run dirs found under {root}.")

    df = pd.DataFrame([row for _, row in runs.values()])
    df = df.sort_values(["sequence", "depth_model", "gap_thresh"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} runs, {df['sequence'].nunique()} sequences)\n")

    for metric in PIVOT_METRICS:
        if metric not in df.columns or df[metric].isna().all():
            continue
        pivot = df.pivot_table(index=["depth_model", "gap_thresh"], columns="sequence", values=metric)
        pivot["mean"] = pivot.mean(axis=1)
        print(f"== {metric} ==")
        print(pivot.round(4).to_string(), "\n")


if __name__ == "__main__":
    main()
