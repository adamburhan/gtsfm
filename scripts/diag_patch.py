"""Show, on one keypoint, exactly how (d, d_alt) are computed from the depth patch.

Picks the ambiguous measurement with the largest log-depth gap (clearest example) — or a
measurement you specify — reproduces _analyze_patch step by step with every intermediate
printed, and saves a figure: the depth patch with the keypoint marked, and the sorted
log-depths with the largest gap / split / near & far clusters / the two modes annotated.

Usage:
    python scripts/diag_patch.py <run_dir> [--gap_thresh 0.10] [--patch_radius 5] [--min_valid 10]
        [--cam C --u U --v V]   # optional: a specific measurement instead of the max-gap one
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData


def local_arrays(npz_path: Path, data: GtsfmData):
    raw = np.load(npz_path)
    arrays = {int(k): raw[k] for k in raw.keys()}
    return {local: arrays[g] for local, g in enumerate(sorted(arrays))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--gap_thresh", type=float, default=0.10)
    p.add_argument("--patch_radius", type=int, default=5)
    p.add_argument("--min_valid", type=int, default=10)
    p.add_argument("--depth_min", type=float, default=0.0)
    p.add_argument("--depth_max", type=float, default=1e9)
    p.add_argument("--cam", type=int, default=None)
    p.add_argument("--u", type=float, default=None)
    p.add_argument("--v", type=float, default=None)
    args = p.parse_args()

    res = Path(args.run_dir) / "results"
    data = GtsfmData.read_colmap(str(res / "vggt"))
    arrays = local_arrays(res / "depth.npz", data)
    provider = DepthProvider(
        depth_arrays=arrays, depth_min=args.depth_min, depth_max=args.depth_max, compute_hypotheses=True,
        patch_radius=args.patch_radius, gap_thresh=args.gap_thresh, ambiguity_thresh=0.0, min_valid=args.min_valid,
    )

    # Pick the measurement: specified, or the ambiguous one with the largest gap (clearest).
    if args.cam is not None and args.u is not None and args.v is not None:
        cam, u, v = args.cam, args.u, args.v
    else:
        best = (-1.0, None)
        for j in range(data.number_tracks()):
            t = data.get_track(j)
            for m in range(t.numberMeasurements()):
                i, uv = t.measurement(m)
                s = provider.get_depth(i, float(uv[0]), float(uv[1]))
                if s is not None and s.ambiguous and s.score > best[0]:
                    best = (s.score, (i, float(uv[0]), float(uv[1])))
        if best[1] is None:
            raise SystemExit("No ambiguous measurements found.")
        cam, u, v = best[1]

    # ---- reproduce _analyze_patch, exposing every step ----
    dmap = arrays[cam]
    h, w = dmap.shape[:2]
    r = args.patch_radius
    col, row = int(np.clip(round(u), 0, w - 1)), int(np.clip(round(v), 0, h - 1))
    d_center = float(dmap[row, col])
    patch = dmap[max(0, row - r):min(h, row + r + 1), max(0, col - r):min(w, col + r + 1)]
    valid = patch[np.isfinite(patch) & (patch >= args.depth_min) & (patch <= args.depth_max)]
    logs = np.sort(np.log(valid))
    gaps = np.diff(logs)
    k = int(np.argmax(gaps))
    max_gap = float(gaps[k])
    split = 0.5 * (logs[k] + logs[k + 1])
    near, far = valid[np.log(valid) <= split], valid[np.log(valid) > split]
    d_near, d_far = float(np.median(near)), float(np.median(far))
    log_c = np.log(d_center)
    d_alt = d_far if abs(log_c - np.log(d_near)) <= abs(log_c - np.log(d_far)) else d_near
    ambiguous = max_gap >= args.gap_thresh and min(near.size, far.size) >= 3

    print(f"camera {cam}, pixel (u,v)=({u:.1f},{v:.1f}) -> (row,col)=({row},{col})")
    print(f"patch {patch.shape}, {valid.size} valid depths in [{valid.min():.3f}, {valid.max():.3f}] (recon units)")
    print(f"d_center (= mode 1, d) = depth_map[{row},{col}] = {d_center:.4f}")
    print(f"largest log gap: index k={k}, max_gap={max_gap:.4f}  (gap_thresh={args.gap_thresh})  "
          f"=> {'>=' if max_gap >= args.gap_thresh else '<'} thresh")
    print(f"split at log={split:.4f} (depth {np.exp(split):.4f}):  near={near.size} pts (median {d_near:.4f}),  "
          f"far={far.size} pts (median {d_far:.4f})")
    print(f"d_alt (= mode 2) = farther-from-center median = {d_alt:.4f}   "
          f"(|logc-log d_near|={abs(log_c-np.log(d_near)):.3f} vs |logc-log d_far|={abs(log_c-np.log(d_far)):.3f})")
    print(f"AMBIGUOUS = (max_gap>=thresh) and (min(near,far)>=3) = {ambiguous}")
    chk = provider.get_depth(cam, u, v)
    print(f"DepthProvider.get_depth agrees: ambiguous={chk.ambiguous}, depth={chk.depth:.4f}, "
          f"depth_alt={None if chk.depth_alt is None else round(chk.depth_alt,4)}")

    # ---- figure: patch heatmap + the two modes on the log-depth axis ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axp, axd) = plt.subplots(1, 2, figsize=(11, 5))
    cr, cc = row - max(0, row - r), col - max(0, col - r)  # center within patch
    im = axp.imshow(patch, cmap="turbo")
    axp.scatter([cc], [cr], s=120, marker="x", c="black", linewidths=2, label="keypoint")
    axp.set_title(f"depth patch (cam {cam}, {patch.shape[0]}x{patch.shape[1]})")
    axp.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=axp, fraction=0.046, label="depth (recon units)")

    lv = np.log(valid)
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, lv.size)
    axd.scatter(jitter[lv <= split], lv[lv <= split], s=18, c="tab:blue", label=f"near ({near.size})")
    axd.scatter(jitter[lv > split], lv[lv > split], s=18, c="tab:orange", label=f"far ({far.size})")
    axd.axhline(split, ls="--", c="gray", label=f"split (gap={max_gap:.3f})")
    axd.axhline(log_c, c="black", lw=2, label=f"d (mode 1) = {d_center:.3f}")
    axd.axhline(np.log(d_alt), c="red", lw=2, ls=":", label=f"d_alt (mode 2) = {d_alt:.3f}")
    axd.set_ylabel("log depth"); axd.set_xticks([])
    axd.set_title(f"two modes  |  ambiguous={ambiguous}")
    axd.legend(loc="best", fontsize=8)

    out = Path(args.run_dir) / "diag" / "patch_example.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
