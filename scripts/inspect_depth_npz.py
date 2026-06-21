"""Validate persisted VGGT depth (`depth.npz`) for a run.

Tier 1 (structural): each npz loads, keys are camera indices, arrays are finite/positive.
Tier 2 (correctness): sample depth at each track measurement and compare to the point's
camera-frame Z in the sibling reconstruction. Correct keying/scale/orientation -> median
ratio ~1.0 and positive; a scale/sign/keying bug shifts the median (genuine depth ambiguity
only fattens the tail).

Usage:
    python scripts/inspect_depth_npz.py [results_dir]   # default: cwd
"""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gtsfm.common.depth_provider import DepthProvider
from gtsfm.common.gtsfm_data import GtsfmData


def structural(npz_path: Path) -> dict[int, np.ndarray]:
    """Tier 1: load + report shape/finiteness/range. Returns {cam_idx: array}."""
    raw = np.load(npz_path)
    arrays = {int(k): raw[k] for k in raw.keys()}
    vals = np.concatenate([a.ravel() for a in arrays.values()])
    fin = vals[np.isfinite(vals)]
    a0 = next(iter(arrays.values()))
    print(f"  [struct] {len(arrays)} cams  shape={a0.shape} dtype={a0.dtype}  "
          f"finite={100*np.isfinite(vals).mean():.1f}% positive={100*(fin>0).mean():.1f}%  "
          f"min={fin.min():.3f} median={np.median(fin):.3f} max={fin.max():.3f}")
    return arrays


def correctness(arrays: dict[int, np.ndarray], recon_dir: Path, max_meas: int, tol: float) -> None:
    """Tier 2: depth-at-pixel vs point camera-frame Z over the sibling reconstruction."""
    data = GtsfmData.read_colmap(str(recon_dir))
    # read_colmap re-bases cameras + measurements to 0-based in sorted-filename order, dropping
    # the global index that depth arrays are keyed by. Loader index order == filename order, so
    # sorted depth keys align positionally with read_colmap's 0-based indices. Re-key to match.
    global_keys = sorted(arrays)
    cam_idxs = data.get_valid_camera_indices()
    if len(cam_idxs) != len(global_keys):
        print(f"  [correct] {recon_dir.name}: {len(cam_idxs)} recon cams != {len(global_keys)} depth cams; "
              "cannot align indices, skipped")
        return
    arrays = {local: arrays[g] for local, g in enumerate(global_keys)}
    provider = DepthProvider(depth_arrays=arrays, depth_min=0.0, depth_max=1e9, compute_hypotheses=False)

    ratios, n_none, n = [], 0, 0
    for j in range(data.number_tracks()):
        track = data.get_track(j)
        point_w = np.array(track.point3())
        for m in range(track.numberMeasurements()):
            i, uv = track.measurement(m)
            cam = data.get_camera(i)
            if cam is None or i not in arrays:
                continue
            n += 1
            sample = provider.get_depth(i, float(uv[0]), float(uv[1]))
            if sample is None:
                n_none += 1
                continue
            z_geom = float(cam.pose().transformTo(point_w)[2])
            if z_geom > 0:
                ratios.append(sample.depth / z_geom)
        if n >= max_meas:
            break

    if not ratios:
        print(f"  [correct] {recon_dir.name}: no valid measurements sampled")
        return
    r = np.array(ratios)
    within = 100 * (np.abs(r - 1.0) < tol).mean()
    print(f"  [correct] {recon_dir.name}: n={len(r)} sampled, {n_none} out-of-range  "
          f"median(d/Z)={np.median(r):.3f}  within±{tol:.0%}={within:.1f}%  "
          f"neg={100*(r<0).mean():.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=".", help="Run results dir to walk for depth.npz.")
    p.add_argument("--recon", choices=["auto", "vggt", "vggt_pre_ba"], default="auto",
                   help="Sibling reconstruction to cross-check (auto = vggt_pre_ba if present).")
    p.add_argument("--max_meas", type=int, default=20000, help="Cap measurements sampled per node.")
    p.add_argument("--tol", type=float, default=0.05, help="Agreement band on depth/Z ratio.")
    args = p.parse_args()

    npz_paths = sorted(Path(args.results_dir).rglob("depth.npz"))
    if not npz_paths:
        print(f"No depth.npz under {args.results_dir}")
        return

    for npz_path in npz_paths:
        print(f"\n{npz_path.relative_to(args.results_dir)}")
        arrays = structural(npz_path)
        node = npz_path.parent
        candidates = ["vggt_pre_ba", "vggt"] if args.recon == "auto" else [args.recon]
        recon_dir = next((node / c for c in candidates if (node / c / "images.txt").exists()), None)
        if recon_dir is None:
            print("  [correct] no sibling reconstruction found; skipped")
            continue
        correctness(arrays, recon_dir, args.max_meas, args.tol)


if __name__ == "__main__":
    main()
