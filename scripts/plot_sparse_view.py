"""Plot the sparse-view crossover experiment: acc50 / acc95 / AUC@5 vs number of views.

Reads the launch_kicker_sparse.sh output tree (<root>/<method>/v<NN>/<seq>/<mode>/geometry_metrics.json)
and draws one line per method across view counts. Tests whether depth-aware BA overtakes no-depth BA
as views decrease (curves crossing in the sparse tail = depth prior useful under weak triangulation).

Usage:
  python scripts/plot_sparse_view.py --root $SCRATCH/logs/sweeps/kicker_sparse --out_dir tables/kicker_sparse
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

# (json key, axis label, lower-is-better, cm-scale). AUC@5 now lives in geometry_metrics.json (eval_geometry).
PANELS = [
    ("accuracy_median_m", "acc50 (cm) ↓", True, 100.0),
    ("accuracy_p95_m", "acc95 (cm) ↓", True, 100.0),
    ("pose_auc_@5.0_deg", "pose AUC@5° ↑", False, 1.0),
]
# Fixed method order + display names (path dir -> label).
METHODS = ["none", "gt", "unidepthv2", "unidepthv2_oracle_gate"]


def _views(vlabel: str, geom: dict) -> int:
    """View count from the vNN dir; 'vfull' falls back to the reconstructed camera count."""
    if vlabel == "vfull":
        return int(geom.get("rotation_angle_error_deg", {}).get("summary", {}).get("len", 0)) or 999
    return int(vlabel.lstrip("v"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Sparse-view sweep root (kicker_sparse).")
    ap.add_argument("--out_dir", default="tables/kicker_sparse", help="Where to write the figure + csv.")
    args = ap.parse_args()
    root, out_dir = Path(args.root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # data[method][views] = geom dict
    data: dict[str, dict[int, dict]] = {m: {} for m in METHODS}
    for gj in root.rglob("geometry_metrics.json"):
        rel = gj.relative_to(root).parts  # <method>/<vNN>/<seq>/<mode>/geometry_metrics.json
        if len(rel) < 4:
            continue
        method, vlabel = rel[0], rel[1]
        if method not in data:
            continue
        geom = json.loads(gj.read_text())
        data[method][_views(vlabel, geom)] = geom

    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 4.2))
    rows = []
    for ax, (key, ylabel, lower_better, scale) in zip(axes, PANELS):
        for method in METHODS:
            pts = sorted(data[method].items())  # by views
            xs = [v for v, _ in pts]
            ys = [(g.get(key) * scale if g.get(key) is not None else float("nan")) for _, g in pts]
            if xs:
                ax.plot(xs, ys, marker="o", label=method)
            for v, g in pts:
                rows.append({"method": method, "views": v, "metric": key, "value": g.get(key)})
        ax.set_xlabel("number of views")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=8, title="method")
    fig.suptitle("Sparse-view crossover (kicker)")
    fig.tight_layout()

    fig_path = out_dir / "sparse_view_crossover.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    with open(out_dir / "sparse_view_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "views", "metric", "value"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["metric"], r["method"], r["views"])))
    print(f"Wrote {fig_path}")
    print(f"Wrote {out_dir / 'sparse_view_metrics.csv'}")


if __name__ == "__main__":
    main()
