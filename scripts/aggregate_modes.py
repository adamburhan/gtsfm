"""Aggregate depth-factor modes into final per-dataset tables (geometry / NVS / pose).

Two ways to point it at runs:
  - single scene:  `none=DIR unimodal=DIR drop_ambiguous=DIR bimodal=DIR`
  - whole dataset: `--root ROOT`, discovering run dirs named after a mode anywhere under
                   ROOT (e.g. ROOT/<seq>/<mode>/ or ROOT/<seq>/modes/<mode>/); the
                   sequence is the path component above the mode (ignoring a 'modes' level).

Per run dir it locates (shallowest match = the merged/top-level file):
  - geometry_metrics.json            (global geometry + "modes" + "ambiguous_subset")
  - bundle_adjustment_metrics.json   (pose AUC, rotation/translation error)
  - gs/stats/val_step*.json          (latest-step PSNR / SSIM / LPIPS)

Outputs an all-metrics CSV, prints the headline (Geometry|NVS|Pose) and mode-selection
tables, and with --latex writes the paper table (seq x mode, category column groups).

Examples:
    python scripts/aggregate_modes.py --root $ETH3D --latex --out_dir tables/eth3d
    python scripts/aggregate_modes.py none=$B/none bimodal=$B/bimodal --out_dir tables/kicker
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

MODE_ORDER = ["none", "unimodal", "drop_ambiguous", "bimodal", "bimodal_gap", "bimodal_gmm", "bimodal_gmm_null", "bimodal_mda", "bimodal_mda_null"]
POSE_KEYS = ["pose_auc_@1.0_deg", "pose_auc_@2.5_deg", "pose_auc_@5.0_deg"]
# Pose metrics live in different files per pipeline; prefer the merged/final one.
POSE_FILES = [
    ("merging_metrics.json", "merging_metrics"),
    ("cluster_vggt_metrics.json", "cluster_vggt_metrics"),
    ("bundle_adjustment_metrics.json", "bundle_adjustment_metrics"),
]
NVS_KEYS = ["psnr", "ssim", "lpips", "num_GS"]
MODE_KEYS = [
    "n_ambiguous_measurements", "mode2_selected_frac", "mode_correct_frac", "mode2_correct_frac",
    "primary_correct_frac", "bimodal_over_primary", "oracle_within_tau_frac",
    "selection_cost_mean_m", "dist_selected_median_m",
]

# (json_key, latex_header, mm_scale, precision) for the paper table.
GEOM_COLS = [("accuracy_median_m", r"acc$_{50}$", 1000, 1), ("accuracy_mean_m", r"acc$_{\mu}$", 1000, 1),
             ("accuracy_p95_m", r"acc$_{95}$", 1000, 1)]
NVS_COLS = [("psnr", "PSNR", 1, 2), ("ssim", "SSIM", 1, 3), ("lpips", "LPIPS", 1, 3)]
POSE_COLS = [("pose_auc_@1.0_deg", r"@1\degree", 1, 3), ("pose_auc_@2.5_deg", r"@2.5\degree", 1, 3),
             ("pose_auc_@5.0_deg", r"@5\degree", 1, 3)]
PAPER_COLS = GEOM_COLS + NVS_COLS + POSE_COLS


def _find(run_dir: Path, pattern: str):
    hits = sorted(run_dir.rglob(pattern), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _median(value):
    return value.get("summary", {}).get("median") if isinstance(value, dict) else value


def collect(run_dir: Path) -> dict:
    """Flat metric record for one run dir (geometry + ambiguous + modes + pose + nvs)."""
    rec: dict = {}
    g = _find(run_dir, "geometry_metrics.json")
    if g:
        geom = json.loads(g.read_text())
        for k in ["n_points", "accuracy_median_m", "accuracy_mean_m", "accuracy_p95_m"]:
            rec[k] = geom.get(k)
        for k in sorted(geom):
            if k.startswith(("precision@", "fscore@")):
                rec[k] = geom[k]
        amb = geom.get("ambiguous_subset", {})
        rec["amb_n_points"] = amb.get("n_points")
        rec["amb_accuracy_median_m"] = amb.get("accuracy_median_m")
        rec.update({k: geom.get("modes", {}).get(k) for k in MODE_KEYS})
    for fname, wrapper in POSE_FILES:
        b = _find(run_dir, fname)
        if b:
            ba = json.loads(b.read_text())[wrapper]
            rec.update({k: ba.get(k) for k in POSE_KEYS})
            rec["rot_err_median_deg"] = _median(ba.get("rotation_angle_error_deg"))
            rec["trans_err_median"] = _median(ba.get("translation_error_distance"))
            break
    vals = sorted(run_dir.rglob("val_step*.json"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if vals:
        gs = json.loads(vals[-1].read_text())
        rec.update({k: gs.get(k) for k in NVS_KEYS})
    return rec


def discover(root: Path) -> dict:
    """Find {(seq, mode): run_dir} by locating mode-named dirs that contain a run."""
    runs = {}
    for mode in MODE_ORDER:
        for d in root.rglob(mode):
            if not d.is_dir() or not _find(d, "geometry_metrics.json"):
                continue
            parents = d.relative_to(root).parts[:-1]
            seq = next((p for p in reversed(parents) if p != "modes"), "scene")
            runs[(seq, mode)] = d
    return runs


def _fmt(v, scale, prec):
    return "---" if v is None or pd.isna(v) else f"{v * scale:.{prec}f}"


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    seqs = sorted(df["seq"].unique())
    rows = [
        r"\begin{table}[h]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "rrr" * 3 + "}", r"\toprule",
        r" & & \multicolumn{3}{c}{Geometry} & \multicolumn{3}{c}{NVS} & \multicolumn{3}{c}{Pose AUC} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        "Seq. & Mode & " + " & ".join(h for _, h, _, _ in PAPER_COLS) + r" \\", r"\midrule",
    ]
    for si, seq in enumerate(seqs):
        if si:
            rows.append(r"\midrule")
        sub = df[df["seq"] == seq]
        modes = [m for m in MODE_ORDER if m in set(sub["mode"])]
        for mi, mode in enumerate(modes):
            r = sub[sub["mode"] == mode].iloc[0]
            cells = [_fmt(r.get(k), scale, prec) for k, _, scale, prec in PAPER_COLS]
            first = r"\multirow{%d}{*}{%s}" % (len(modes), seq) if mi == 0 else ""
            rows.append(f"{first} & {mode} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="mode=run_dir pairs (single scene). Ignored if --root given.")
    parser.add_argument("--root", default=None, help="Discover <seq>/<mode>/ runs under this dir (whole dataset).")
    parser.add_argument("--out_dir", default=".", help="Where to write CSV / .tex.")
    parser.add_argument("--latex", action="store_true", help="Also write the paper table (seq x mode).")
    parser.add_argument("--caption", default="Sweep results.", help="LaTeX caption.")
    parser.add_argument("--label", default="tab:sweep", help="LaTeX label.")
    args = parser.parse_args()

    if args.root:
        run_map = discover(Path(args.root))
    else:
        run_map = {}
        for spec in args.runs:
            mode, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"Expected mode=run_dir, got '{spec}'")
            run_map[("scene", mode)] = Path(path)
    if not run_map:
        raise SystemExit("No runs found.")

    recs = [{"seq": seq, "mode": mode, **collect(d)} for (seq, mode), d in run_map.items()]
    df = pd.DataFrame(recs)
    df["mode"] = pd.Categorical(df["mode"], [m for m in MODE_ORDER if m in set(df["mode"])])
    df = df.sort_values(["seq", "mode"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_metrics.csv", index=False)

    main_cols = ["seq", "mode"] + [k for k, *_ in PAPER_COLS if k in df.columns]
    print("\n== Geometry | NVS | Pose (headline) ==")
    print(df[main_cols].round(4).to_markdown(index=False))
    mode_cols = ["seq", "mode"] + [k for k in MODE_KEYS if k in df.columns]
    if df[[k for k in MODE_KEYS if k in df.columns]].notna().any().any():
        print("\n== Mode-selection (contribution) ==")
        print(df[mode_cols].round(4).to_markdown(index=False))

    if args.latex:
        tex_path = out_dir / "paper_table.tex"
        tex_path.write_text(latex_table(df, args.caption, args.label))
        print(f"\nWrote {tex_path}")
    print(f"Wrote {out_dir / 'all_metrics.csv'}")


if __name__ == "__main__":
    main()
