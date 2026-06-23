"""Diagnose one depth-factor run: did the pipeline classify ambiguity and apply the factor as expected?

Checks:
  1. Recompute the ambiguous count on vggt_pre_ba (the recon BA factored on) with the BA's
     params; it should match the logged num_depth_factors_bimodal.
  2. Recompute on vggt (post-BA); the difference vs (1) is post-BA track filtering.
  3. Save per-camera overlays of ambiguous measurements on the depth map, to eyeball that
     they sit at depth discontinuities rather than flat regions.

Usage:
    python scripts/diag_run.py <run_dir> [--gap_thresh 0.10] [--patch_radius 5] [--min_valid 10] [--n_vis 6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData


def _local_arrays(npz_path: Path, data: GtsfmData):
    """depth.npz re-keyed to read_colmap's 0-based sorted-filename order (as in eval_geometry)."""
    raw = np.load(npz_path)
    arrays = {int(k): raw[k] for k in raw.keys()}
    gk = sorted(arrays)
    n = len(data.get_valid_camera_indices())
    if n != len(gk):
        print(f"  [warn] {n} recon cams != {len(gk)} depth cams; positional map may be off.")
    return {local: arrays[g] for local, g in enumerate(gk)}


def classify(recon_dir: Path, npz_path: Path, args):
    """Walk measurements, count ambiguous, and record per-camera (u, v, ambiguous)."""
    data = GtsfmData.read_colmap(str(recon_dir))
    arrays = _local_arrays(npz_path, data)
    provider = DepthProvider(
        depth_arrays=arrays, depth_min=args.depth_min, depth_max=args.depth_max, compute_hypotheses=True,
        patch_radius=args.patch_radius, gap_thresh=args.gap_thresh, ambiguity_thresh=0.0, min_valid=args.min_valid,
    )
    n_amb = n_tot = 0
    per_cam: dict[int, list] = {}
    for j in range(data.number_tracks()):
        track = data.get_track(j)
        for m in range(track.numberMeasurements()):
            i, uv = track.measurement(m)
            s = provider.get_depth(i, float(uv[0]), float(uv[1]))
            if s is None:
                continue
            amb = bool(s.ambiguous and s.depth_alt is not None)
            n_tot += 1
            n_amb += amb
            per_cam.setdefault(i, []).append((float(uv[0]), float(uv[1]), amb))
    return arrays, n_amb, n_tot, per_cam


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--gap_thresh", type=float, default=0.10)
    p.add_argument("--patch_radius", type=int, default=5)
    p.add_argument("--min_valid", type=int, default=10)
    p.add_argument("--depth_min", type=float, default=0.0)
    p.add_argument("--depth_max", type=float, default=1e9)
    p.add_argument("--n_vis", type=int, default=6)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    res = Path(args.run_dir) / "results"
    npz = res / "depth.npz"
    logged = json.loads(next(res.rglob("depth_factor_metrics.json")).read_text())["depth_factor_metrics"]
    n_bi = int(logged["num_depth_factors_bimodal"])

    print("== Count cross-check (eval classifier vs BA) ==")
    print(f"  BA logged: bimodal={n_bi}, unimodal={int(logged['num_depth_factors_unimodal'])}, "
          f"dropped={int(logged['num_depth_factors_dropped_ambiguous'])}")
    arrays, n_pre, t_pre, per_cam = classify(res / "vggt_pre_ba", npz, args)
    rel = abs(n_pre - n_bi) / max(n_bi, 1)
    print(f"  recompute on vggt_pre_ba : ambiguous={n_pre} / {t_pre}   vs logged {n_bi}  "
          f"({'MATCH' if rel < 0.05 else 'MISMATCH'}, {100*rel:.1f}% off)")
    _, n_post, t_post, _ = classify(res / "vggt", npz, args)
    print(f"  recompute on vggt (post) : ambiguous={n_post} / {t_post}   (delta {n_pre - n_post} = post-BA filtering)")

    out = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "diag"
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cams = sorted(per_cam, key=lambda c: -sum(a for _, _, a in per_cam[c]))[: args.n_vis]
    for c in cams:
        pts = np.array(per_cam[c])
        amb, non = pts[pts[:, 2] == 1], pts[pts[:, 2] == 0]
        plt.figure(figsize=(8, 5))
        plt.imshow(arrays[c], cmap="turbo")
        plt.scatter(non[:, 0], non[:, 1], s=4, c="white", alpha=0.35, label="measurement")
        if len(amb):
            plt.scatter(amb[:, 0], amb[:, 1], s=16, c="red", edgecolors="k", linewidths=0.3, label="ambiguous")
        plt.title(f"cam {c}: {len(amb)}/{len(pts)} ambiguous")
        plt.legend(loc="upper right", fontsize=7)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"cam_{c:03d}.png", dpi=120)
        plt.close()
    print(f"\nWrote {len(cams)} overlays (most-ambiguous cameras) -> {out}")


if __name__ == "__main__":
    main()
