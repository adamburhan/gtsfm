"""Global determinism controls: seed every RNG the classical pipeline touches.

Reproducibility across runs requires seeding cv2 (verifier RANSAC), NumPy (triangulation and robust
Sim(3) sampling), Python ``random``, and torch (SuperPoint / LightGlue / NetVLAD), plus forcing cuDNN
to deterministic algorithms. GTSfM runs the front-end and data association on separate Dask worker
*processes*, so ``set_deterministic`` must also run in each worker — register :class:`DeterminismPlugin`
on the client (see ``runner._create_dask_client``).

For RANSAC calls that run in parallel across workers, a per-process seed is not enough: the order in
which a worker processes pairs/tracks varies run-to-run, so each call must reset its own RNG with a
fixed seed (``RANSAC_SEED``) right before it — that makes the result call-order independent. See the
reseed in ``verifier/ransac.py`` and the local generators in ``point3d_initializer`` / ``utils.align``.

Note: ``PYTHONHASHSEED`` fixes set/dict-hash iteration order but only takes effect if set *before* the
interpreter starts — export it in the launch script, not here.
"""

import random as _random

import numpy as np

RANSAC_SEED = 0  # reset before each RANSAC call so estimates are call-order independent


def set_deterministic(seed: int = 0) -> None:
    """Seed Python, NumPy, OpenCV, and torch (incl. cuDNN) in the current process."""
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import cv2

        cv2.setRNGSeed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


try:
    from dask.distributed import WorkerPlugin

    class DeterminismPlugin(WorkerPlugin):
        """Seeds each Dask worker process at startup (front-end nets + verifier RANSAC run there)."""

        def __init__(self, seed: int = 0):
            self.seed = seed

        def setup(self, worker):
            set_deterministic(self.seed)

    def register_determinism(client, seed: int = 0) -> None:
        """Register the plugin on all current + future workers (API name varies across Dask versions)."""
        plugin = DeterminismPlugin(seed)
        for method in ("register_plugin", "register_worker_plugin"):
            fn = getattr(client, method, None)
            if fn is not None:
                try:
                    fn(plugin)
                    return
                except Exception:
                    continue

except Exception:  # dask not importable in this context
    DeterminismPlugin = None

    def register_determinism(client, seed: int = 0) -> None:
        return
