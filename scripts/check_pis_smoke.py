"""Evaluate the per-image-scale smoke-test gate (spec §8) from depth_per_image_scale_log.json.

Criteria checked:
  (a) cost_pre_refit non-increasing across alternation rounds (round-0 entries have no cost);
  (b) all final a_i in [0.7, 1.5], no clamps (fallbacks are reported, not failed);
  (c) optional: Pearson r > 0.5 between the converged a_i and the offline dense-fit a_i
      (--bias_fit CSV, joined on the image filename stem).
Criterion (d) — the `none` re-run bit-identity — is a separate diff of the two run dirs.

Example:
    python scripts/check_pis_smoke.py RUN_DIR/depth_per_image_scale_log.json \
        --bias_fit bias_fit.csv --image_col image --a_col a
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="depth_per_image_scale_log.json from the smoke run")
    parser.add_argument("--a_lo", type=float, default=0.7)
    parser.add_argument("--a_hi", type=float, default=1.5)
    parser.add_argument("--bias_fit", default=None, help="offline dense-fit CSV (per-image a_i)")
    parser.add_argument("--image_col", default="image", help="bias_fit column with the image filename")
    parser.add_argument("--a_col", default="a", help="bias_fit column with the fitted per-image scale")
    parser.add_argument("--invert_bias_fit", action="store_true",
                        help="compare against 1/a from bias_fit (BA's a_i is the measurement CORRECTION, "
                             "i.e. the inverse of the fitted miscalibration)")
    args = parser.parse_args()

    log = json.loads(Path(args.log).read_text())
    rounds = log["rounds"]
    ok = True

    # (a) cost non-increasing over consecutive alternation rounds.
    costs = [(r["round"], r["cost_pre_refit"]) for r in rounds if r.get("cost_pre_refit") is not None]
    print(f"rounds: {len(rounds)} total, {len(costs)} with cost; sf={log.get('sf')}")
    for (r0, c0), (r1, c1) in zip(costs, costs[1:]):
        marker = "" if c1 <= c0 * (1 + 1e-9) else "  <-- INCREASED"
        if marker:
            ok = False
        print(f"  round {r0} -> {r1}: cost {c0:.6g} -> {c1:.6g}{marker}")

    # (b) final a_i band + clamps.
    final = {e["image_id"]: e for e in rounds[-1]["images"]}
    a = np.array([e["a"] for e in final.values()])
    n_clamped = sum(e["clamped"] for e in final.values())
    n_fallback = sum(e["fallback"] for e in final.values())
    print(f"final a_i: n={len(a)} mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f}")
    print(f"fallbacks={n_fallback} clamped={n_clamped}")
    if a.min() < args.a_lo or a.max() > args.a_hi:
        ok = False
        print(f"  FAIL: a_i outside [{args.a_lo}, {args.a_hi}]")
    if n_clamped > 0:
        ok = False
        print("  FAIL: clamped images present")

    # (c) correlation with the offline dense-fit scales.
    if args.bias_fit:
        import pandas as pd

        df = pd.read_csv(args.bias_fit)
        offline = {Path(str(r[args.image_col])).stem: float(r[args.a_col]) for _, r in df.iterrows()}
        xs, ys = [], []
        for e in final.values():
            stem = Path(e.get("image_fname") or "").stem
            if stem in offline:
                ref = offline[stem]
                xs.append(np.log(e["a"]))
                ys.append(np.log(1.0 / ref if args.invert_bias_fit else ref))
        if len(xs) < 3:
            ok = False
            print(f"  FAIL: only {len(xs)} images matched between log and {args.bias_fit}")
        else:
            r = float(np.corrcoef(xs, ys)[0, 1])
            print(f"pearson r (log a, n={len(xs)}): {r:.3f}")
            if r <= 0.5:
                ok = False
                print("  FAIL: r <= 0.5")

    print("SMOKE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
