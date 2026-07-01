"""Aggregate depth-factor sweep runs into per-dataset tables (geometry / NVS / pose).

Two ways to point it at runs:
  - single scene:  `none=DIR unimodal=DIR drop_ambiguous=DIR bimodal=DIR`
  - whole dataset: `--root ROOT`, discovering run dirs named after a mode anywhere under
                   ROOT (e.g. ROOT/<seq>/<mode>/ or ROOT/<seq>/modes/<mode>/); the
                   sequence is the path component above the mode (ignoring a 'modes' level).

Per run dir it locates (shallowest match = the merged/top-level file):
  - geometry_metrics.json            (geometry accuracy/precision + "alignment": Sim(3) scale, camera RMS)
  - bundle_adjustment_metrics.json   (pose AUC, rotation/translation error)
  - gs/stats/val_step*.json          (latest-step PSNR / SSIM / LPIPS)

Outputs an all-metrics CSV, prints the diagnostic table (geometry cm | pose RMS | Sim(3) scale),
and with --latex writes the paper table (seq x mode, category column groups, best-per-metric bolded).

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

# Diagnostic per-scene table (printed + reference). (json_key, header, scale, precision).
DIAG_COLS = [
    ("n_points", "n", 1, 0),
    ("n_factors", "n_fac", 1, 0),
    ("accuracy_median_m", "median_cm", 100, 2),
    ("accuracy_mean_m", "mean_cm", 100, 2),
    ("accuracy_p95_m", "p95_cm", 100, 2),
    ("precision@1cm", "prec@1cm", 1, 3),
    ("precision@2cm", "prec@2cm", 1, 3),
    ("precision@5cm", "prec@5cm", 1, 3),
    ("camera_rms_m", "poseRMS_cm", 100, 2),
    ("sim3_scale", "scale", 1, 4),
]

# Paper table. (json_key, latex_header, scale, precision, direction) — direction picks best (min/max) to bold.
GEOM_COLS = [
    ("accuracy_median_m", r"acc$_{50}\downarrow$", 1000, 1, "min"),
    ("accuracy_mean_m", r"acc$_{\mu}\downarrow$", 1000, 1, "min"),
    ("accuracy_p95_m", r"acc$_{95}\downarrow$", 1000, 1, "min"),
]
NVS_COLS = [
    ("psnr", r"PSNR$\uparrow$", 1, 2, "max"),
    ("ssim", r"SSIM$\uparrow$", 1, 3, "max"),
    ("lpips", r"LPIPS$\downarrow$", 1, 3, "min"),
]
POSE_COLS = [
    ("pose_auc_@1.0_deg", r"@1$^\circ\uparrow$", 1, 3, "max"),
    ("pose_auc_@2.5_deg", r"@2.5$^\circ\uparrow$", 1, 3, "max"),
    ("pose_auc_@5.0_deg", r"@5$^\circ\uparrow$", 1, 3, "max"),
]
PAPER_COLS = GEOM_COLS + NVS_COLS + POSE_COLS


def _find(run_dir: Path, pattern: str):
    hits = sorted(run_dir.rglob(pattern), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _median(value):
    return value.get("summary", {}).get("median") if isinstance(value, dict) else value


def collect(run_dir: Path) -> dict:
    """Flat metric record for one run dir (geometry + alignment + pose + nvs)."""
    rec: dict = {}
    g = _find(run_dir, "geometry_metrics.json")
    if g:
        geom = json.loads(g.read_text())
        for k in ["n_points", "accuracy_median_m", "accuracy_mean_m", "accuracy_p95_m"]:
            rec[k] = geom.get(k)
        for k in sorted(geom):
            if k.startswith(("precision@", "recall@", "fscore@")):
                rec[k] = geom[k]
        alignment = geom.get("alignment", {})
        rec["sim3_scale"] = alignment.get("sim3_scale")
        rec["camera_rms_m"] = alignment.get("camera_rms_m")
    for fname, wrapper in POSE_FILES:
        b = _find(run_dir, fname)
        if b:
            ba = json.loads(b.read_text()).get(wrapper, {})
            rec.update({k: ba.get(k) for k in POSE_KEYS})
            rec["rot_err_median_deg"] = _median(ba.get("rotation_angle_error_deg"))
            rec["trans_err_median"] = _median(ba.get("translation_error_distance"))
            break
    # Depth factors actually applied (unimodal + bimodal), from the BA metrics. n_factors=0 on a
    # depth row means the map was missing / all samples skipped -> the run is silently just `none`.
    bam = _find(run_dir, "bundle_adjustment_metrics.json")
    if bam:
        ba = json.loads(bam.read_text()).get("bundle_adjustment_metrics", {})
        uni, bim = ba.get("num_depth_factors_unimodal"), ba.get("num_depth_factors_bimodal")
        if uni is not None or bim is not None:
            rec["n_factors"] = int(uni or 0) + int(bim or 0)
    vals = sorted(run_dir.rglob("val_step*.json"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if vals:
        gs = json.loads(vals[-1].read_text())
        rec.update({k: gs.get(k) for k in NVS_KEYS})
    return rec


def discover(root: Path) -> dict:
    """Find {(source, seq, mode): run_dir} by locating mode-named dirs that contain a run.

    source = the top-level condition dir (the sweep_name, e.g. gt / depthpro), seq = the dir just
    above the mode. source is "" for a flat <seq>/<mode> root (single-condition aggregation).
    """
    runs = {}
    for mode in MODE_ORDER:
        for d in root.rglob(mode):
            if not d.is_dir() or not _find(d, "geometry_metrics.json"):
                continue
            parents = d.relative_to(root).parts[:-1]
            seq = next((p for p in reversed(parents) if p != "modes"), "scene")
            source = parents[0] if len(parents) >= 2 else ""
            runs[(source, seq, mode)] = d
    return runs


def _fmt(v, scale, prec):
    return "---" if v is None or pd.isna(v) else f"{v * scale:.{prec}f}"


def _best_value(series, scale, prec, direction):
    """Best display value in a column (min/max over non-null, rounded to display precision), or None."""
    vals = [round(v * scale, prec) for v in series if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return min(vals) if direction == "min" else max(vals)


def _tex(s: str) -> str:
    return s.replace("_", r"\_")


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    seqs = sorted(df["seq"].unique())
    rows = [
        r"\providecommand{\best}[1]{\textbf{#1}}",
        r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{ll" + "rrr" * 3 + "}", r"\toprule",
        r" & & \multicolumn{3}{c}{Geometry (mm) $\downarrow$} & \multicolumn{3}{c}{NVS} "
        r"& \multicolumn{3}{c}{Pose AUC $\uparrow$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        "Seq. & Method & " + " & ".join(h for _, h, _, _, _ in PAPER_COLS) + r" \\", r"\midrule",
    ]
    for si, seq in enumerate(seqs):
        if si:
            rows.append(r"\midrule")
        sub = df[df["seq"] == seq].sort_values(["source", "mode"])
        best = {k: _best_value(sub[k], scale, prec, direction)
                for k, _, scale, prec, direction in PAPER_COLS if k in sub.columns}
        for ri, (_, r) in enumerate(sub.iterrows()):
            cells = []
            for k, _, scale, prec, _ in PAPER_COLS:
                s = _fmt(r.get(k), scale, prec)
                if s != "---" and best.get(k) is not None and round(r[k] * scale, prec) == best[k]:
                    s = r"\best{%s}" % s
                cells.append(s)
            method = f"{r['source']}/{r['mode']}" if r["source"] else str(r["mode"])
            first = r"\multirow{%d}{*}{%s}" % (len(sub), _tex(seq)) if ri == 0 else ""
            rows.append(f"{first} & {_tex(method)} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table*}"]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="mode=run_dir pairs (single scene). Ignored if --root given.")
    parser.add_argument("--root", default=None, help="Discover <seq>/<mode>/ runs under this dir (whole dataset).")
    parser.add_argument("--out_dir", default=".", help="Where to write CSV / .tex.")
    parser.add_argument("--latex", action="store_true", help="Also write the paper table (seq x mode).")
    parser.add_argument(
        "--caption",
        default=(
            "Depth-factor sweep on ETH3D. Geometry in millimetres; arrows show the preferred direction "
            "(lower for geometry/LPIPS, higher for PSNR/SSIM/pose AUC). Best per sequence and metric is bolded."
        ),
        help="LaTeX caption.",
    )
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
            run_map[("", "scene", mode)] = Path(path)
    if not run_map:
        raise SystemExit("No runs found.")

    recs = [{"source": src, "seq": seq, "mode": mode, **collect(d)} for (src, seq, mode), d in run_map.items()]
    df = pd.DataFrame(recs)
    df["mode"] = pd.Categorical(df["mode"], [m for m in MODE_ORDER if m in set(df["mode"])])
    df["source"] = pd.Categorical(df["source"], sorted(set(df["source"])))
    df = df.sort_values(["seq", "source", "mode"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_metrics.csv", index=False)

    disp = pd.DataFrame({"seq": df["seq"], "source": df["source"], "mode": df["mode"]})
    for key, header, scale, prec in DIAG_COLS:
        disp[header] = df[key].map(lambda v: _fmt(v, scale, prec)) if key in df.columns else "---"
    print("\n== Diagnostic table (geometry cm | pose RMS cm | Sim(3) scale) ==")
    print(disp.to_markdown(index=False))

    if args.latex:
        tex_path = out_dir / "paper_table.tex"
        tex_path.write_text(latex_table(df, args.caption, args.label))
        print(f"\nWrote {tex_path}")
    print(f"Wrote {out_dir / 'all_metrics.csv'}")


if __name__ == "__main__":
    main()
