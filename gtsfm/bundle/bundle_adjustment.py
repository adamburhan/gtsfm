"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import dask
import gtsam  # type: ignore
import numpy as np
from dask.delayed import Delayed
from gtsam import BetweenFactorPose3, NonlinearFactorGraph, PriorFactorPoint3, PriorFactorPose3, Values  # type: ignore
from gtsam.noiseModel import Diagonal, Isotropic, Robust, mEstimator  # type: ignore
from gtsam.symbol_shorthand import K, P, X  # type: ignore
from numpy.typing import NDArray

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common import gtsfm_data
from gtsfm.common.depth_provider import DepthProvider, MdaDepthProvider
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationOptions,
    TriangulationSamplingMode,
)
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

METRICS_GROUP = "bundle_adjustment_metrics"

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"
"""In this file, we use the GTSAM's GeneralSFMFactor2 instead of GeneralSFMFactor because Factor2 enables decoupling
of the camera pose and the camera intrinsics, and hence gives an option to share the intrinsics between cameras.
"""


CAM_POSE3_DOF = 6  # 6 dof for pose of camera
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logger = logger_utils.get_logger()


class RobustBAMode(Enum):
    """Robust BA modes."""

    NONE = "NONE"
    HUBER = "HUBER"
    GMC = "GMC"
    TLS = "TLS"


class DepthFactorMode(Enum):
    """Depth-factor variants for depth-aware bundle adjustment."""

    NONE = "none"  # No depth factors (baseline BA).
    UNIMODAL = "unimodal"  # One depth factor per measurement, single hypothesis.
    DROP_AMBIGUOUS = "drop_ambiguous"  # Like UNIMODAL, but skip measurements near depth discontinuities.
    BIMODAL = "bimodal"  # Max-mixture factor over (depth, depth_alt) for ambiguous measurements.


def make_depth_factor(pose_key: int, lm_key: int, d: float, noise) -> "gtsam.CustomFactor":
    """Binary camera-frame-Z depth factor on a camera pose and a landmark.

    Residual is `z_pred - d`, where `z_pred` is the Z coordinate of the landmark
    expressed in the camera frame (planar depth, as measured by a depth sensor),
    and `d` is the measured depth in meters. The analytic Jacobian is the third
    row of the `transformTo` Jacobians w.r.t. the pose (1x6) and the point (1x3).

    Args:
        pose_key: GTSAM key for the camera pose variable X(i) (world-to-camera as wTc).
        lm_key: GTSAM key for the landmark point variable P(j).
        d: Measured depth in meters.
        noise: 1-D noise model for the residual.

    Returns:
        A `gtsam.CustomFactor` on [pose_key, lm_key].
    """

    def error_func(this, values, H):
        pose_wTc = values.atPose3(pose_key)
        point_w = values.atPoint3(lm_key)

        H_pose = np.zeros((3, 6), dtype=np.float64, order="F")
        H_point = np.zeros((3, 3), dtype=np.float64, order="F")

        point_c = pose_wTc.transformTo(point_w, H_pose, H_point)
        z_pred = float(point_c[2])

        if H is not None:
            H[0] = H_pose[2:3, :]
            H[1] = H_point[2:3, :]

        return np.array([z_pred - d], dtype=np.float64)

    return gtsam.CustomFactor(noise, [pose_key, lm_key], error_func)


def make_bimodal_depth_factor(pose_key: int, lm_key: int, d: float, d_alt: float, noise) -> "gtsam.CustomFactor":
    """Max-mixture variant of the depth factor, for measurements near depth discontinuities.

    The residual is computed against whichever of the two depth hypotheses (`d`,
    `d_alt`) is closer to the predicted camera-frame Z (min-|residual| mode
    selection — the equal-weight, equal-sigma special case of max-mixtures). The
    Jacobian is mode-independent, since both residuals share d(z_pred)/dx and
    differ only by a constant.

    Args:
        pose_key: GTSAM key for the camera pose variable X(i) (world-to-camera as wTc).
        lm_key: GTSAM key for the landmark point variable P(j).
        d: Measured depth in meters (primary hypothesis, at the sampled pixel).
        d_alt: Second depth hypothesis in meters (the other side of the discontinuity).
        noise: 1-D noise model for the residual (shared by both modes).

    Returns:
        A `gtsam.CustomFactor` on [pose_key, lm_key].
    """

    def error_func(this, values, H):
        pose_wTc = values.atPose3(pose_key)
        point_w = values.atPoint3(lm_key)

        H_pose = np.zeros((3, 6), dtype=np.float64, order="F")
        H_point = np.zeros((3, 3), dtype=np.float64, order="F")

        point_c = pose_wTc.transformTo(point_w, H_pose, H_point)
        z_pred = float(point_c[2])
        r1, r2 = z_pred - d, z_pred - d_alt
        err = r1 if abs(r1) < abs(r2) else r2

        if H is not None:
            H[0] = H_pose[2:3, :]
            H[1] = H_point[2:3, :]

        return np.array([err], dtype=np.float64)

    return gtsam.CustomFactor(noise, [pose_key, lm_key], error_func)


def make_mixture_depth_factor(
    pose_key, lm_key, depths, sigmas, log_weights, noise, null_nsigma=None
) -> "gtsam.CustomFactor":
    """Weighted, per-mode-sigma max-mixture depth factor (N>=1 modes), with an optional null hypothesis.

    Picks the component minimizing ``0.5*(r/sigma)^2 + log(sigma) - log(weight)`` and returns the
    sigma-whitened residual under a *unit* ``noise`` model. One mode → a plain depth factor; two
    modes with equal sigma/weight → the old min-|r| bimodal factor. So treating a unimodal pixel as
    two near-identical modes is a no-op, and every measurement can use this.

    Null hypothesis (``null_nsigma``): when set, if even the *selected* mode is more than
    ``null_nsigma`` sigmas from the predicted depth, the factor opts out — it returns a zero residual
    with a zero Jacobian, so only the reprojection factor constrains that measurement. 
    """
    depths = np.asarray(depths, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)

    def error_func(this, values, H):
        pose_wTc = values.atPose3(pose_key)
        point_w = values.atPoint3(lm_key)

        H_pose = np.zeros((3, 6), dtype=np.float64, order="F")
        H_point = np.zeros((3, 3), dtype=np.float64, order="F")

        point_c = pose_wTc.transformTo(point_w, H_pose, H_point)
        z_pred = float(point_c[2])
        r = (z_pred - depths) / sigmas
        k = int(np.argmin(0.5 * r * r + np.log(sigmas))) # temporarily uniform prior
        if null_nsigma is not None and abs(r[k]) > null_nsigma:
            # Null hypothesis selected: no mode fits the geometry -> disable the depth term here.
            if H is not None:
                H[0] = np.zeros((1, 6), dtype=np.float64)
                H[1] = np.zeros((1, 3), dtype=np.float64)
            return np.array([0.0], dtype=np.float64)
        if H is not None:
            H[0] = H_pose[2:3, :] / sigmas[k]
            H[1] = H_point[2:3, :] / sigmas[k]
        return np.array([r[k]], dtype=np.float64)

    return gtsam.CustomFactor(noise, [pose_key, lm_key], error_func)


class DepthFactorRecord(NamedTuple):
    """One depth-factor measurement, kept so the per-image scale refit can rebuild factors.

    All quantities are post-``sf`` (recon units): ``depths`` are the mode means fed to the factor
    before any per-image scale ``a_i`` is applied. ``sigmas``/``log_weights`` are set for mixture
    records only (the unimodal/bimodal kinds take their noise model from the BA options).
    """

    i: int  # camera index (X key)
    j: int  # track index (P key)
    kind: str  # "unimodal" | "bimodal" | "mixture"
    depths: Tuple[float, ...]
    sigmas: Optional[Tuple[float, ...]] = None
    log_weights: Optional[Tuple[float, ...]] = None


def refit_depth_scales(
    records: List[DepthFactorRecord],
    values: Values,
    a_prev: Dict[int, float],
    *,
    min_factors: int,
    clamp: float,
    null_nsigma: Optional[float] = None,
    resid_scale: float = 1.0,
) -> Tuple[Dict[int, float], Dict[int, dict]]:
    """Per-image profiled depth scale: ``log a_i = median_k(log z_k - log m_k)`` (sigma_a -> inf).

    For each record, ``z_k`` is the camera-frame Z of the current point under the current pose (from
    ``values``); ``m_k`` is the record's measurement for the currently winning mode. Mixture records
    select the mode with the factor's own rule (``0.5 r^2 + log sigma`` on the a_prev-scaled modes;
    the common ``log a_prev`` term makes this equal to selecting against ``a_prev * mu``) and are
    excluded when routed to the null hypothesis; behind-camera factors (z <= 0) are excluded too.
    Images with fewer than ``min_factors`` usable factors fall back to a_i = 1.0 (global sf only).

    Deterministic: records are consumed in build order and images emitted in sorted order; no RNG.

    Args:
        records: Depth factor records in graph build order.
        values: Current estimate (poses X(i) and points P(j)).
        a_prev: Previous per-image scales (missing image -> 1.0).
        min_factors: Below this many usable factors, a_i = 1.0 and fallback is flagged.
        clamp: Hard bound on |log a_i|.
        null_nsigma: Mixture null-hypothesis band (None = off), matching the factor's.
        resid_scale: Multiplier mapping recon-unit residuals to meters (the ``sf`` of this stage),
            applied only to the logged median residuals.

    Returns:
        (a_new, info): new per-image scales, and per-image log entries
        {image_id, n_factors_used, log_a, a, fallback, clamped, median_abs_resid_pre/post}.
    """
    zm: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    image_ids = sorted({rec.i for rec in records})
    n_behind = 0
    n_null = 0
    for rec in records:
        pose_wTc = values.atPose3(X(rec.i))
        point_w = values.atPoint3(P(rec.j))
        z = float(pose_wTc.transformTo(point_w)[2])
        if z <= 0.0:
            n_behind += 1
            continue
        a = float(a_prev.get(rec.i, 1.0))
        if rec.kind == "mixture":
            mus = np.asarray(rec.depths, dtype=np.float64)
            sigs = np.asarray(rec.sigmas, dtype=np.float64)
            r = (z - a * mus) / (a * sigs)
            k = int(np.argmin(0.5 * r * r + np.log(a * sigs)))
            if null_nsigma is not None and abs(r[k]) > null_nsigma:
                n_null += 1
                continue
            m = float(mus[k])
        elif rec.kind == "bimodal":
            m = float(min(rec.depths, key=lambda d: abs(z - a * d)))
        else:
            m = float(rec.depths[0])
        if m <= 0.0:
            continue
        zm[rec.i].append((z, m))
    if n_behind or n_null:
        logger.info(
            "Depth scale refit: excluded %d behind-camera and %d null-routed factors.", n_behind, n_null
        )

    a_new: Dict[int, float] = {}
    info: Dict[int, dict] = {}
    for i in image_ids:
        pairs = zm.get(i, [])
        n = len(pairs)
        fallback = n < min_factors
        clamped = False
        if fallback:
            log_a = 0.0
        else:
            log_a = float(np.median([np.log(z) - np.log(m) for z, m in pairs]))
            if abs(log_a) > clamp:
                clamped = True
                log_a = float(np.clip(log_a, -clamp, clamp))
                logger.warning("Depth scale refit: image %d hit the |log a| <= %.3f clamp.", i, clamp)
        a_i = float(np.exp(log_a))
        a_pre = float(a_prev.get(i, 1.0))
        a_new[i] = a_i
        info[i] = {
            "image_id": i,
            "n_factors_used": n,
            "log_a": log_a,
            "a": a_i,
            "fallback": fallback,
            "clamped": clamped,
            "median_abs_resid_pre": float(np.median([abs(z - a_pre * m) for z, m in pairs]) * resid_scale)
            if pairs
            else None,
            "median_abs_resid_post": float(np.median([abs(z - a_i * m) for z, m in pairs]) * resid_scale)
            if pairs
            else None,
        }
    return a_new, info


def multi_view_retriangulate_from_2d_tracks(
    gtsfm_data: GtsfmData,
    tracks_2d: List["SfmTrack2d"],
    triangulation_options: Optional[TriangulationOptions] = None,
    min_track_length: int = 3,
) -> GtsfmData:
    """Re-triangulate the union-find 2D track set against post-BA cameras.

    Each input track is solved by `Point3dInitializer.triangulate`: RANSAC samples
    camera pairs, triangulates a candidate 3D point via 2-view DLT, scores by
    counting inliers (measurements within `reproj_error_threshold`) across the
    whole track, and picks the best sample. The final 3D point is then computed
    by a multi-view DLT over the inlier measurements.

    Recovers tracks that were dropped earlier in the pipeline when cameras
    weren't yet converged — those triangulations can now succeed against the
    refined cameras.

    2-view tracks are excluded by default (`min_track_length=3`). They are weak
    (no inlier consensus across views, single-pair baseline) and don't help BA.

    Args:
        gtsfm_data: Scene with optimized cameras (existing tracks ignored; cameras re-used).
        tracks_2d: Full 2D track set from `CppDsfTracksEstimator.run()`.
        triangulation_options: Per-track solve config. Defaults to RANSAC_SAMPLE_UNIFORM,
            reproj_error_threshold=10 px, max_num_hypotheses=100. For tracks with
            N measurements there are N*(N-1)/2 possible 2-view samples, so for
            typical short tracks the cap is rarely binding; set lower if RANSAC
            wall-time matters on long tracks.
        min_track_length: Drop tracks with fewer measurements than this.

    Returns:
        New GtsfmData with same cameras and an augmented track set.
    """
    if triangulation_options is None:
        triangulation_options = TriangulationOptions(
            reproj_error_threshold=10.0,
            mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
            max_num_hypotheses=100,
        )

    camera_dict = {i: cam for i, cam in gtsfm_data.cameras().items() if cam is not None}
    initializer = Point3dInitializer(camera_dict, triangulation_options)

    out = GtsfmData(gtsfm_data.number_images())
    for cam_idx, camera in camera_dict.items():
        out.add_camera(cam_idx, camera)
        info = gtsfm_data.get_image_info(cam_idx)
        out.set_image_info(cam_idx, name=info.name, shape=info.shape)

    n_in = len(tracks_2d)
    n_kept = 0
    n_short_input = 0
    n_no_cams = 0
    n_short_output = 0
    n_triangulation_failed = 0
    for track_2d in tracks_2d:
        if track_2d.number_measurements() < min_track_length:
            n_short_input += 1
            continue
        valid_measurements = [m for m in track_2d.measurements if m.i in camera_dict]
        if len(valid_measurements) < min_track_length:
            n_no_cams += 1
            continue
        track_2d_filtered = SfmTrack2d(measurements=valid_measurements)

        new_track, _, _ = initializer.triangulate(track_2d_filtered)
        if new_track is None:
            n_triangulation_failed += 1
            continue
        if new_track.numberMeasurements() < min_track_length:
            n_short_output += 1
            continue
        out.add_track(new_track)
        n_kept += 1

    logger.info(
        "Multi-view retriangulation: %d kept / %d input (min_len=%d): "
        "%d short_in, %d no_cams, %d tri_failed, %d short_out.",
        n_kept, n_in, min_track_length, n_short_input, n_no_cams,
        n_triangulation_failed, n_short_output,
    )
    return out


@dataclass
class BundleAdjustmentOptions:
    """Shared configuration for bundle adjustment across leaf BA, merging BA, etc.

    This dataclass captures the commonly-configured subset of BA parameters
    that vary between call sites (leaf vs. merging). Parameters with fixed
    defaults (e.g. ordering_type, allow_indeterminate_linear_system) are left
    to the ``BundleAdjustmentOptimizer`` constructor.
    """

    robust_ba_mode: Union[RobustBAMode, str] = RobustBAMode.GMC
    shared_calib: bool = False
    use_calibration_prior: bool = False
    use_pose_prior_all_cameras: bool = False
    use_pose_prior_first_camera: bool = False
    use_gnc: bool = False
    gnc_loss: Union[RobustBAMode, str] = RobustBAMode.GMC
    factor_weight_outlier_threshold: float = 0.0
    min_track_length: int = 2
    calibration_prior_focal_sigma: float = 20.0
    calibration_prior_dist_sigma: float | Sequence[float] = 0.1
    calibration_prior_pp_sigma: float = 1e-5
    robust_noise_basin: float = 1.345
    min_tracks_per_camera: int = 15
    compute_pose_covariances: bool = False
    optimizer_relative_cost_tol: float = 1e-5
    # Depth factors (opt-in; baseline BA is unchanged when depth_model is "none").
    # For on-disk depth (e.g. GT), set depth_map_dir; for transformer-predicted
    # depth (e.g. VGGT), depth arrays are injected at runtime instead.
    depth_model: Union[DepthFactorMode, str] = DepthFactorMode.NONE
    depth_factor_sigma: float = 0.1
    depth_factor_robust_loss: bool = False
    depth_map_dir: Optional[str] = None
    depth_min: float = 0.1
    depth_max: float = 20.0
    depth_scale: float = 1.0
    depth_filename_template: Optional[str] = "depth{:06d}.png"
    depth_ext: str = ".png"  # used when depth_filename_template is null: depth file = <image-stem><depth_ext>
    depth_patch_radius: int = 5
    depth_gap_thresh: float = 0.15
    depth_ambiguity_thresh: float = 0.20
    depth_min_valid: int = 10
    depth_mda_dir: Optional[str] = None  # precomputed MDA mixtures (dump_mda_mixture); overrides patch modes
    depth_mda_sigma_rel: float = 0.05    # MDA mixture sigma RELATIVE to depth (sigma = rel*depth); scale-invariant
    depth_mda_near_prior: float = 0.3    # slight log-weight penalty per depth rank (nearer mode preferred)
    depth_hypothesis_method: str = "gap"  # "gap" (largest-gap heuristic) | "gmm" (2-component GMM)
    depth_gmm_min_weight: float = 0.15   # GMM: min mass on the smaller mode to flag a sample ambiguous
    depth_gmm_sigma_floor: float = 0.05  # GMM: relative floor on per-mode sigma (frac of mode depth)
    depth_null_nsigma: Optional[float] = None  # mixture null hypothesis: opt out if best mode > N sigmas off (None=off)
    depth_auto_scale: bool = False  # divide metric depth by a robust global recon<-metric scale before BA (classical SfM)
    # GT null-hypothesis gate (oracle diagnostic): drop a measurement's depth factor unless some mode
    # backprojects within depth_gt_tau of the GT cloud. Isolates whether bad candidates (vs the BA
    # mechanism) are the bottleneck. Requires the GT cloud + the COLMAP GT-pose dir.
    depth_gt_gate: bool = False
    depth_gt_ply: Optional[str] = None
    depth_gt_align_ref: Optional[str] = None  # COLMAP GT dir (poses), as eval_geometry's --align_ref
    depth_gt_tau: float = 0.05
    depth_gt_oracle_select: bool = False  # also collapse to the GT-closest mode (mode-selection ceiling)
    depth_gt_scale: bool = False  # use the GT Sim(3) scale for sf even when gating is off (removes the auto_scale confound)
    # Relative (depth-proportional) sigma for unimodal/bimodal depth factors: sigma = max(rel * a_i * d, floor).
    # 0.0 keeps the legacy fixed depth_factor_sigma. Both sigmas are metric (meters), whitened by /sf as before.
    depth_relative_sigma: float = 0.0
    depth_sigma_floor: float = 0.02
    # Per-image profiled depth scale a_i (sigma_a -> inf limit of a scale-nuisance model), fit by
    # outer alternation (BA -> refit -> rebuild). sf handles the global gauge; a_i the per-image differential.
    depth_per_image_scale: bool = False
    depth_pis_min_factors: int = 20  # images with fewer usable factors keep a_i = 1.0 (global sf only)
    depth_pis_max_rounds: int = 4  # max (BA -> refit) alternation rounds per BA stage
    depth_pis_tol: float = 1e-3  # stop when max_i |delta log a_i| < tol
    depth_pis_clamp: float = 0.693  # hard clamp |log a_i| <= clamp (log 2)

    def to_optimizer(self, **overrides) -> "BundleAdjustmentOptimizer":
        """Construct a :class:`BundleAdjustmentOptimizer` from these options.

        Args:
            **overrides: Keyword arguments that override dataclass fields or add
                extra constructor params (e.g. ``min_track_length``,
                ``reproj_error_thresholds``).
        """
        kwargs = dict(
            robust_ba_mode=self.robust_ba_mode,
            shared_calib=self.shared_calib,
            use_calibration_prior=self.use_calibration_prior,
            use_pose_prior_all_cameras=self.use_pose_prior_all_cameras,
            use_pose_prior_first_camera=self.use_pose_prior_first_camera,
            use_gnc=self.use_gnc,
            gnc_loss=self.gnc_loss,
            factor_weight_outlier_threshold=self.factor_weight_outlier_threshold,
            min_track_length=self.min_track_length,
            calibration_prior_focal_sigma=self.calibration_prior_focal_sigma,
            calibration_prior_dist_sigma=self.calibration_prior_dist_sigma,
            calibration_prior_pp_sigma=self.calibration_prior_pp_sigma,
            robust_noise_basin=self.robust_noise_basin,
            min_tracks_per_camera=self.min_tracks_per_camera,
            compute_pose_covariances=self.compute_pose_covariances,
            optimizer_relative_cost_tol=self.optimizer_relative_cost_tol,
            depth_model=self.depth_model,
            depth_factor_sigma=self.depth_factor_sigma,
            depth_factor_robust_loss=self.depth_factor_robust_loss,
            depth_map_dir=self.depth_map_dir,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            depth_scale=self.depth_scale,
            depth_filename_template=self.depth_filename_template,
            depth_ext=self.depth_ext,
            depth_patch_radius=self.depth_patch_radius,
            depth_gap_thresh=self.depth_gap_thresh,
            depth_ambiguity_thresh=self.depth_ambiguity_thresh,
            depth_min_valid=self.depth_min_valid,
            depth_mda_dir=self.depth_mda_dir,
            depth_mda_sigma_rel=self.depth_mda_sigma_rel,
            depth_mda_near_prior=self.depth_mda_near_prior,
            depth_hypothesis_method=self.depth_hypothesis_method,
            depth_gmm_min_weight=self.depth_gmm_min_weight,
            depth_gmm_sigma_floor=self.depth_gmm_sigma_floor,
            depth_null_nsigma=self.depth_null_nsigma,
            depth_auto_scale=self.depth_auto_scale,
            depth_gt_gate=self.depth_gt_gate,
            depth_gt_ply=self.depth_gt_ply,
            depth_gt_align_ref=self.depth_gt_align_ref,
            depth_gt_tau=self.depth_gt_tau,
            depth_gt_oracle_select=self.depth_gt_oracle_select,
            depth_gt_scale=self.depth_gt_scale,
            depth_relative_sigma=self.depth_relative_sigma,
            depth_sigma_floor=self.depth_sigma_floor,
            depth_per_image_scale=self.depth_per_image_scale,
            depth_pis_min_factors=self.depth_pis_min_factors,
            depth_pis_max_rounds=self.depth_pis_max_rounds,
            depth_pis_tol=self.depth_pis_tol,
            depth_pis_clamp=self.depth_pis_clamp,
        )
        kwargs.update(overrides)
        return BundleAdjustmentOptimizer(**kwargs)


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation.

    Due to the process graph requiring separate classes for separate graph objects, this class is a superclass for
    TwoViewBundleAdjustment and GlobalBundleAdjustment (defined in gtsfm/bundle/).
    """

    def __init__(
        self,
        reproj_error_thresholds: Sequence[Optional[float]] = [None],
        robust_ba_mode: RobustBAMode = RobustBAMode.NONE,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
        cam_pose3_prior_noise_sigma: float = 0.1,
        calibration_prior_focal_sigma: float = 20.0,
        calibration_prior_dist_sigma: float | Sequence[float] = 0.1,
        calibration_prior_pp_sigma: float = 1e-5,
        measurement_noise_sigma: float = 2.0,
        allow_indeterminate_linear_system: bool = True,
        print_summary: bool = False,
        ordering_type: str = "METIS",
        save_iteration_visualization: bool = False,
        robust_noise_basin: float = 1.345,
        use_karcher_mean_factor: bool = True,
        use_pose_prior_all_cameras: bool = False,
        use_pose_prior_first_camera: bool = False,
        use_calibration_prior: bool = True,
        use_first_point_prior: bool = False,
        use_gnc: bool = False,
        gnc_loss: RobustBAMode | str = RobustBAMode.GMC,
        factor_weight_outlier_threshold: float = 0.0,
        min_track_length: int = 2,
        min_tracks_per_camera: int = 15,
        compute_pose_covariances: bool = False,
        optimizer_relative_cost_tol: float = 1e-5,
        # ── Optional depth factors (opt-in) ──
        # When `depth_model != "none"` and `depth_map_dir` is set, add a 1-D
        # camera-frame-Z depth factor per track measurement, alongside the
        # reprojection factor. Requires per-image filenames, supplied via
        # `image_fnames` to `create_computation_graph`. Baseline BA is unchanged
        # when `depth_model="none"` (default). The `drop_ambiguous` and `bimodal`
        # modes analyze a patch around each sample for fg/bg depth ambiguity
        # (see `DepthProvider`), respectively skipping ambiguous measurements or
        # giving them a two-hypothesis max-mixture factor.
        depth_model: DepthFactorMode | str = DepthFactorMode.NONE,
        depth_factor_sigma: float = 0.1,
        depth_factor_robust_loss: bool = False,
        depth_map_dir: Optional[str] = None,
        depth_min: float = 0.1,
        depth_max: float = 20.0,
        depth_scale: float = 1.0,
        depth_filename_template: Optional[str] = "depth{:06d}.png",
        depth_ext: str = ".png",
        depth_patch_radius: int = 5,
        depth_gap_thresh: float = 0.15,
        depth_ambiguity_thresh: float = 0.20,
        depth_min_valid: int = 10,
        depth_mda_dir: Optional[str] = None,
        depth_mda_sigma_rel: float = 0.05,
        depth_mda_near_prior: float = 0.3,
        depth_hypothesis_method: str = "gap",
        depth_gmm_min_weight: float = 0.15,
        depth_gmm_sigma_floor: float = 0.05,
        depth_null_nsigma: Optional[float] = None,
        depth_auto_scale: bool = False,
        depth_gt_gate: bool = False,
        depth_gt_ply: Optional[str] = None,
        depth_gt_align_ref: Optional[str] = None,
        depth_gt_tau: float = 0.05,
        depth_gt_oracle_select: bool = False,
        depth_gt_scale: bool = False,
        depth_relative_sigma: float = 0.0,
        depth_sigma_floor: float = 0.02,
        depth_per_image_scale: bool = False,
        depth_pis_min_factors: int = 20,
        depth_pis_max_rounds: int = 4,
        depth_pis_tol: float = 1e-3,
        depth_pis_clamp: float = 0.693,
        # ── Optional post-BA multi-view retriangulation (opt-in) ──
        # When `use_multi_view_retriangulation=True`: after the existing BA loop
        # converges, re-triangulate the union-find 2D tracks against the post-BA
        # cameras (recovers tracks dropped between union-find and BA's filter
        # passes) and run a final BA on the augmented set. Requires `tracks_2d` to
        # be passed to `create_computation_graph` / `_run_ba_and_evaluate`. The
        # final BA reuses the existing `reproj_error_thresholds[-1]` for filtering.
        use_multi_view_retriangulation: bool = False,
        mv_retri_min_track_length: int = 3,
        mv_retri_reproj_error_thresh: float = 10.0,
        mv_retri_max_num_hypotheses: int = 100,
    ) -> None:
        """Initializes the parameters for bundle adjustment module.

        Args:
            reproj_error_thresholds (optional): List of reprojection error thresholds used to perform filtering after
                each global bundle adjustment step. Implicitly defines the number of global BA steps, e.g., if
                len(reproj_error_thresholds) == 1, only one step will be performed. If the threshold is None, no
                filtering on output data is performed. Defaults to None.
            robust_ba_mode (optional): Robust BA mode to use, defaults to NONE.
            shared_calib (optional): Flag to enable shared calibration across all cameras. Defaults to False.
            max_iterations (optional): Max number of iterations when optimizing the factor graph. None means no cap.
                Defaults to None.
            cam_pose3_prior_noise_sigma (optional): Camera Pose3 prior noise sigma.
            calibration_prior_focal_sigma (optional): Sigma for the prior on the focal length, defaults to 20.0.
            calibration_prior_dist_sigma (optional): Sigma for the prior on the distortion parameters, defaults to 0.1.
            measurement_noise_sigma (optional): Measurement noise sigma in pixel units.
            allow_indeterminate_linear_system: Reject a two-view measurement if an indeterminate linear system is
                encountered during marginal covariance computation after bundle adjustment.
            ordering_type (optional): The ordering algorithm to use for variable elimination.
            save_iteration_visualization (optional): Save a Plotly animation showing optimization progress.
            robust_noise_basin (optional): Basin to use for the robust noise model.
            use_karcher_mean_factor (optional): Use Karcher mean factor to constrain the camera poses.
            use_pose_prior (optional): Use pose prior to constrain the camera poses. (only used if we use karcher mean)
            use_calibration_prior (optional): Use calibration prior to constrain the camera intrinsics.
            use_first_point_prior (optional): Use first point prior to constrain the scale of the reconstruction.
            use_gnc (optional): Use the GNC optimizer for bundle adjustment.
            gnc_loss (optional): GNC loss to use. Defaults to GMC.
            factor_weight_outlier_threshold (optional): Threshold weight for a reprojection factor to be kept.
            min_track_length: min number of measurements required to keep a track after weight filtering.
            compute_pose_covariances: If true, compute marginal covariance for all camera pose variables and return it.
        """
        self._reproj_error_thresholds = reproj_error_thresholds
        if isinstance(robust_ba_mode, str):
            self._robust_ba_mode = RobustBAMode[robust_ba_mode]
        else:
            self._robust_ba_mode = robust_ba_mode
        self._shared_calib = shared_calib
        self._max_iterations = max_iterations
        self._cam_pose3_prior_noise_sigma = cam_pose3_prior_noise_sigma
        self._use_calibration_prior = use_calibration_prior
        self._calibration_prior_focal_sigma = calibration_prior_focal_sigma
        self._calibration_prior_dist_sigma = calibration_prior_dist_sigma
        self._calibration_prior_pp_sigma = calibration_prior_pp_sigma
        self._measurement_noise_sigma = measurement_noise_sigma
        self._allow_indeterminate_linear_system = allow_indeterminate_linear_system
        self._ordering_type = ordering_type
        self._print_summary = print_summary
        self._save_iteration_visualization = save_iteration_visualization
        self._robust_noise_basin = robust_noise_basin
        self._use_karcher_mean_factor = use_karcher_mean_factor
        self._use_pose_prior_all_cameras = use_pose_prior_all_cameras
        self._use_pose_prior_first_camera = use_pose_prior_first_camera
        self._use_first_point_prior = use_first_point_prior
        self._use_gnc = use_gnc
        if isinstance(gnc_loss, str):
            self._gnc_loss = RobustBAMode[gnc_loss]
        else:
            self._gnc_loss = gnc_loss
        self._factor_weight_outlier_threshold = factor_weight_outlier_threshold
        self._min_track_length = min_track_length
        self._min_tracks_per_camera = min_tracks_per_camera
        self._compute_pose_covariances = compute_pose_covariances
        self._optimizer_relative_cost_tol = optimizer_relative_cost_tol

        # Depth factors (opt-in). See `__init__` docstring above.
        if isinstance(depth_model, str):
            self._depth_model = DepthFactorMode(depth_model)
        else:
            self._depth_model = depth_model
        self._depth_factor_sigma = depth_factor_sigma
        self._depth_factor_robust_loss = depth_factor_robust_loss
        self._depth_map_dir = depth_map_dir
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._depth_scale = depth_scale
        self._depth_filename_template = depth_filename_template
        self._depth_ext = depth_ext
        self._depth_patch_radius = depth_patch_radius
        self._depth_gap_thresh = depth_gap_thresh
        self._depth_ambiguity_thresh = depth_ambiguity_thresh
        self._depth_min_valid = depth_min_valid
        self._depth_mda_dir = depth_mda_dir
        self._depth_mda_sigma_rel = depth_mda_sigma_rel
        self._depth_mda_near_prior = depth_mda_near_prior
        self._depth_hypothesis_method = depth_hypothesis_method
        self._depth_gmm_min_weight = depth_gmm_min_weight
        self._depth_gmm_sigma_floor = depth_gmm_sigma_floor
        self._depth_null_nsigma = depth_null_nsigma
        self._depth_auto_scale = depth_auto_scale
        self._depth_gt_gate = depth_gt_gate
        self._depth_gt_ply = depth_gt_ply
        self._depth_gt_align_ref = depth_gt_align_ref
        self._depth_gt_tau = depth_gt_tau
        self._depth_gt_oracle_select = depth_gt_oracle_select
        self._depth_gt_scale = depth_gt_scale
        self._depth_relative_sigma = depth_relative_sigma
        self._depth_sigma_floor = depth_sigma_floor
        self._depth_per_image_scale = depth_per_image_scale
        self._depth_pis_min_factors = depth_pis_min_factors
        self._depth_pis_max_rounds = depth_pis_max_rounds
        self._depth_pis_tol = depth_pis_tol
        self._depth_pis_clamp = depth_pis_clamp
        if self._depth_per_image_scale and use_gnc:
            # GNC weight filtering renumbers tracks between rounds, invalidating the cached records.
            raise ValueError("depth_per_image_scale is not supported with use_gnc.")
        # Per-image profiled scale state (reset per BA invocation in _run_ba_and_evaluate).
        self._pis_a: Dict[int, float] = {}
        self._pis_records: Optional[List[DepthFactorRecord]] = None
        self._pis_sf: float = 1.0
        self._pis_round_active = False  # alternation rebuild: reuse cached records, don't resample
        self._pis_log: List[dict] = []
        self._pis_round_counter = 0
        self._pis_last_info: Dict[int, dict] = {}
        self._image_fnames: Optional[Dict[int, str]] = None
        self._depth_arrays: Optional[Dict[int, np.ndarray]] = None
        self._depth_factor_stats: Dict[str, int] = {"unimodal": 0, "bimodal": 0, "dropped_ambiguous": 0, "skipped": 0}
        self._depth_provider = None

        # Post-BA multi-view retriangulation (opt-in). See `__init__` docstring above.
        self._use_multi_view_retriangulation = use_multi_view_retriangulation
        self._mv_retri_min_track_length = mv_retri_min_track_length
        self._mv_retri_reproj_error_thresh = mv_retri_reproj_error_thresh
        self._mv_retri_max_num_hypotheses = mv_retri_max_num_hypotheses

    def __map_to_calibration_variable(self, camera_idx: int) -> int:
        return 0 if self._shared_calib else camera_idx

    def __get_cameras_with_insufficient_tracks(self, initial_data: GtsfmData) -> set[int]:
        """Get the cameras with insufficient fewer tracks that self._min_tracks_per_camera."""
        cameras_with_insufficient_tracks = set()
        for i in initial_data.get_valid_camera_indices():
            if len(initial_data.get_measurements_for_camera(i)) < self._min_tracks_per_camera:
                cameras_with_insufficient_tracks.add(i)
        return cameras_with_insufficient_tracks

    def __reprojection_factors(
        self, initial_data: GtsfmData, cameras_to_model: List[int], robust_noise_basin: float | None = None
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Generate reprojection factors using the tracks."""
        graph = NonlinearFactorGraph()

        # noise model for measurements -- one pixel in u and v
        measurement_noise = Isotropic.Sigma(IMG_MEASUREMENT_DIM, self._measurement_noise_sigma)
        noise_basin = robust_noise_basin if robust_noise_basin is not None else self._robust_noise_basin
        if self._robust_ba_mode == RobustBAMode.HUBER:
            measurement_noise = Robust(mEstimator.Huber(noise_basin), measurement_noise)
        elif self._robust_ba_mode == RobustBAMode.GMC:
            measurement_noise = Robust(mEstimator.GemanMcClure(noise_basin), measurement_noise)

        # Note: Assumes all calibration types are the same.
        first_camera = initial_data.get_camera(cameras_to_model[0])
        assert first_camera is not None, "First camera in initial data is None"
        sfm_factor_class = gtsfm_types.get_sfm_factor_for_calibration(first_camera.calibration())

        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            valid_measurements = [
                m_idx for m_idx in range(track.numberMeasurements()) if track.measurement(m_idx)[0] in cameras_to_model
            ]
            if len(valid_measurements) < self._min_track_length:
                continue
            # Retrieve the SfmMeasurement objects.
            for m_idx in valid_measurements:
                # `i` represents the camera index, and `uv` is the 2d measurement
                i, uv = track.measurement(m_idx)
                graph.push_back(
                    sfm_factor_class(
                        uv,
                        measurement_noise,
                        X(i),
                        P(j),
                        K(self.__map_to_calibration_variable(i)),
                    )  # type: ignore
                )

        return graph

    def __get_depth_provider(self) -> Optional[DepthProvider]:
        """Lazily build the depth provider, or None if depth factors are disabled.

        Requires `depth_model != NONE` plus a depth source: either in-memory
        `depth_arrays` (transformer-predicted depth, e.g. VGGT) or a `depth_map_dir`
        with per-image filenames (on-disk depth, e.g. GT). Both are set via
        `create_computation_graph`. Returns None (and logs once) if neither source
        is available, so depth factors are simply skipped.
        """
        if self._depth_model == DepthFactorMode.NONE:
            return None
        if self._depth_provider is not None:
            return self._depth_provider
        compute_hypotheses = self._depth_model in (DepthFactorMode.DROP_AMBIGUOUS, DepthFactorMode.BIMODAL)
        if self._depth_mda_dir is not None:
            # MDA mixture modes, aligned per-image to the in-memory VGGT depth (the affine fix).
            self._depth_provider = MdaDepthProvider(
                self._depth_mda_dir,
                self._depth_arrays,
                depth_min=self._depth_min,
                depth_max=self._depth_max,
                gap_thresh=self._depth_gap_thresh,
                near_prior=self._depth_mda_near_prior,
                sigma_rel=self._depth_mda_sigma_rel,
            )
        elif self._depth_arrays is not None:
            # In-memory depth keyed by image index; no filenames or scaling needed.
            self._depth_provider = DepthProvider(
                depth_arrays=self._depth_arrays,
                depth_min=self._depth_min,
                depth_max=self._depth_max,
                compute_hypotheses=compute_hypotheses,
                patch_radius=self._depth_patch_radius,
                gap_thresh=self._depth_gap_thresh,
                ambiguity_thresh=self._depth_ambiguity_thresh,
                min_valid=self._depth_min_valid,
                hypothesis_method=self._depth_hypothesis_method,
                gmm_min_weight=self._depth_gmm_min_weight,
                gmm_sigma_floor=self._depth_gmm_sigma_floor,
            )
        elif self._depth_map_dir is not None and self._image_fnames is not None:
            self._depth_provider = DepthProvider(
                depth_map_dir=self._depth_map_dir,
                image_fnames=self._image_fnames,
                depth_min=self._depth_min,
                depth_max=self._depth_max,
                depth_scale=self._depth_scale,
                depth_filename_template=self._depth_filename_template,
                depth_ext=self._depth_ext,
                compute_hypotheses=compute_hypotheses,
                patch_radius=self._depth_patch_radius,
                gap_thresh=self._depth_gap_thresh,
                ambiguity_thresh=self._depth_ambiguity_thresh,
                min_valid=self._depth_min_valid,
                hypothesis_method=self._depth_hypothesis_method,
                gmm_min_weight=self._depth_gmm_min_weight,
                gmm_sigma_floor=self._depth_gmm_sigma_floor,
            )
        else:
            logger.warning(
                "depth_model=%s but no depth source (depth_arrays or depth_map_dir+filenames); "
                "skipping depth factors.",
                self._depth_model.value,
            )
            return None
        return self._depth_provider

    def __estimate_recon_metric_scale(
        self, initial_data: GtsfmData, cameras_to_model: List[int], depth_provider: DepthProvider
    ) -> float:
        """Robust global scale ``s`` mapping metric depth into the gauge-arbitrary recon scale.

        Classical SfM fixes scale only up to one global similarity, so a single scalar
        ``s = exp(median(log d_metric - log z_pred_init))`` over sampled measurements aligns the
        (metric) depth maps to the reconstruction. Estimated in log space so far points don't
        dominate, and capped at a few thousand samples since one global DOF needs no more. Depths
        and per-mode sigmas are divided by ``s`` when building factors; the whitened
        ``null_nsigma`` is scale-invariant and unaffected.
        """
        max_samples = 5000
        log_ratios: List[float] = []
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)
            point_w = track.point3()
            for m_idx in range(track.numberMeasurements()):
                i, uv = track.measurement(m_idx)
                if i not in cameras_to_model:
                    continue
                cam = initial_data.get_camera(i)
                if cam is None:
                    continue
                z_init = float(cam.pose().transformTo(point_w)[2])
                if z_init <= 1e-6:
                    continue
                sample = depth_provider.get_depth(i, float(uv[0]), float(uv[1]))
                if sample is None or sample.depth <= 0.0:
                    continue
                log_ratios.append(float(np.log(sample.depth) - np.log(z_init)))
            if len(log_ratios) >= max_samples:
                break
        if not log_ratios:
            logger.warning("Depth auto-scale enabled but no valid samples; using s=1.0.")
            return 1.0
        s = float(np.exp(np.median(log_ratios)))
        logger.info("Depth auto-scale: recon<-metric s=%.4f from %d samples.", s, len(log_ratios))
        return s

    def __depth_scale(
        self, initial_data: GtsfmData, cameras_to_model: List[int], depth_provider: DepthProvider, gate
    ) -> float:
        """Scale ``sf`` mapping metric depth into the gauge-arbitrary recon frame (``d / sf``).

        Classical SfM fixes scale only up to one global similarity, so metric depth must be divided
        by ``sf`` before it can be compared to the recon-frame ``z_pred``. Preference order: the GT
        Sim(3) geometric scale (from the gate, or from ``depth_gt_scale`` when gating is off), then the
        point-ratio ``auto_scale``, then 1.0. The Sim(3) scale is robust; ``auto_scale`` is a median
        over ALL modes and collapses when most are garbage (delivery_area: 92% bad -> recon shrank to
        1/3). ``depth_gt_scale`` holds the scale fixed across conditions so a method isn't penalized by
        a collapsed ``auto_scale`` — it removes the scale confound in the sweep table (oracle: needs GT).
        """
        if gate is not None:
            sf = float(gate[2])
            logger.info("Depth factor scale: using GT-gate geometric scale sf=%.4f.", sf)
            return sf
        if self._depth_gt_scale:
            aln = self.__gt_sim3(initial_data, cameras_to_model)
            if aln is not None:
                sf = float(aln[0].scale())
                logger.info("Depth factor scale: using GT Sim(3) scale sf=%.4f (ungated).", sf)
                return sf
            logger.warning("depth_gt_scale set but GT Sim(3) unavailable; falling back to auto_scale/1.0.")
        if self._depth_auto_scale:
            return self.__estimate_recon_metric_scale(initial_data, cameras_to_model, depth_provider)
        return 1.0

    def __gt_sim3(self, initial_data: GtsfmData, cameras_to_model: List[int]):
        """Robust recon->GT Sim(3) ``(wSr, camera_rms_m)`` from GT poses, or None if unavailable.

        Reads GT poses straight from the COLMAP dir (``depth_gt_align_ref``) and aligns them to the
        current recon cameras (same Sim(3) as the pose-AUC / geometry eval), so it works even when the
        reconstruction used estimated extrinsics. Shared by the GT gate and the GT-scale path.
        """
        if self._depth_gt_align_ref is None or self._image_fnames is None:
            return None
        from gtsfm.evaluation.eval_geometry import _gt_poses_from_colmap, sim3_align

        # _gt_poses_from_colmap expects a list whose position == camera index (matches wTi_list);
        # convert the {idx: name} map, filling gaps with "" so missing slots are skipped, not crash.
        n = max(cameras_to_model) + 1
        wTi_list = [None] * n
        img_fnames_list = [""] * n
        for i in cameras_to_model:
            cam = initial_data.get_camera(i)
            if cam is not None:
                wTi_list[i] = cam.pose()
            if i in self._image_fnames:
                img_fnames_list[i] = self._image_fnames[i]
        try:
            return sim3_align(wTi_list, _gt_poses_from_colmap(self._depth_gt_align_ref, img_fnames_list))
        except (ValueError, KeyError) as exc:
            logger.warning("GT Sim(3): alignment failed (%s).", exc)
            return None

    def __build_gt_gate(self, initial_data: GtsfmData, cameras_to_model: List[int]):
        """Build the GT null-hypothesis gate, or None if disabled/unavailable.

        Aligns the current recon to the GT poses (same Sim(3) as the pose-AUC / geometry eval) and
        loads the GT surface mesh into a raycasting scene. Returns ``(dist_fn, to_world, scale)`` where
        ``to_world`` maps a recon point into the GT/world frame and ``scale`` is the recon->world factor, so a
        metric mode ``d`` placed via ``to_world(cam.backproject(uv, d/scale))`` lands at its true
        metric position. Reads GT poses straight from the COLMAP dir, so it works even when the
        reconstruction itself used estimated (non-GT) extrinsics.
        """
        if not self._depth_gt_gate:
            return None
        if self._depth_gt_ply is None or self._depth_gt_align_ref is None or self._image_fnames is None:
            logger.warning("depth_gt_gate set but missing ply / align_ref / image_fnames; skipping GT gate.")
            return None
        aln = self.__gt_sim3(initial_data, cameras_to_model)
        if aln is None:
            logger.warning("GT gate: alignment unavailable; skipping GT gate.")
            return None
        wSr, rms = aln
        # Distance query against the GT surface mesh (e.g. ETH3D occlusion/surface_mesh.ply): true
        # point-to-SURFACE distance via raycasting. The gate is only ever fed a mesh; if a non-mesh
        # ply slips through, skip the gate rather than silently fall back to (biased) nearest-point.
        import open3d as o3d

        o3d_mesh = o3d.io.read_triangle_mesh(self._depth_gt_ply)
        if len(o3d_mesh.triangles) == 0:
            logger.warning("GT gate: %s has no triangles (not a surface mesh); skipping GT gate.", self._depth_gt_ply)
            return None
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))

        def dist_fn(pts: np.ndarray) -> np.ndarray:
            return scene.compute_distance(o3d.core.Tensor(np.asarray(pts, dtype=np.float32))).numpy()

        logger.info(
            "GT gate: mesh surface, recon->world scale=%.4f, camera RMS=%.4f m, tau=%.3f m, oracle_select=%s",
            wSr.scale(), rms, self._depth_gt_tau, self._depth_gt_oracle_select,
        )
        return dist_fn, (lambda p: np.asarray(wSr.transformFrom(np.asarray(p, dtype=float)))), float(wSr.scale())

    def __gt_gate_modes(self, cam, uv, modes, gate) -> Tuple[bool, Optional[float]]:
        """Is any mode within tau of the GT geometry? Returns (keep, GT-closest mode)."""
        dist_fn, to_world, scale = gate
        valid = [float(d) for d in modes if d is not None and d > 0.0]
        if not valid:
            return False, None
        world_pts = np.array([to_world(cam.backproject(uv, d / scale)) for d in valid])
        dists = dist_fn(world_pts)
        k = int(np.argmin(dists))
        return float(dists[k]) < self._depth_gt_tau, valid[k]

    def __prepare_depth_records(
        self, initial_data: GtsfmData, cameras_to_model: List[int]
    ) -> Tuple[List[DepthFactorRecord], float]:
        """Sample depth, apply the GT gate, and compute ``sf`` — the once-per-stage measurement prep.

        Mirrors `__reprojection_factors`' track/measurement gating so the depth
        factors live on exactly the same observation edges. A measurement with no
        depth map, or an out-of-range/invalid depth, is skipped. Ambiguous
        measurements (near depth discontinuities) are handled per `depth_model`:
        dropped in DROP_AMBIGUOUS mode, given a max-mixture factor in BIMODAL
        mode, and treated as unimodal otherwise.

        Returns records (post-sf measurements, in graph build order) and ``sf``, so
        `__build_depth_graph` can rebuild the factors for any per-image scale
        assignment without resampling. The gate runs on the unscaled (post-sf)
        measurements, so a_i never affects gate decisions.
        """
        depth_provider = self.__get_depth_provider()
        if depth_provider is None:
            return [], 1.0

        # GT null-hypothesis gate (oracle diagnostic): drop factors whose modes all miss the GT surface.
        gate = self.__build_gt_gate(initial_data, cameras_to_model)

        # Scale mapping metric depth into the gauge-arbitrary recon frame (a separate concern from the gate).
        sf = self.__depth_scale(initial_data, cameras_to_model, depth_provider, gate)

        records: List[DepthFactorRecord] = []
        n_unimodal = 0
        n_bimodal = 0
        n_dropped_ambiguous = 0
        n_skipped = 0
        n_gt_gated = 0
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)
            valid_measurements = [
                m_idx for m_idx in range(track.numberMeasurements()) if track.measurement(m_idx)[0] in cameras_to_model
            ]
            if len(valid_measurements) < self._min_track_length:
                continue
            for m_idx in valid_measurements:
                i, uv = track.measurement(m_idx)
                sample = depth_provider.get_depth(i, float(uv[0]), float(uv[1]))
                if sample is None:
                    n_skipped += 1
                    continue
                if gate is not None:
                    modes = (
                        list(sample.depths) if sample.is_mixture
                        else ([sample.depth, sample.depth_alt] if sample.depth_alt is not None else [sample.depth])
                    )
                    keep, best_mode = self.__gt_gate_modes(initial_data.get_camera(i), uv, modes, gate)
                    if not keep:
                        n_gt_gated += 1
                        continue
                    if self._depth_gt_oracle_select and best_mode is not None:
                        # Mode-selection ceiling: collapse to the GT-closest hypothesis (unimodal).
                        records.append(DepthFactorRecord(i, j, "unimodal", (best_mode / sf,)))
                        n_unimodal += 1
                        continue
                if sample.is_mixture:
                    # Weighted max-mixture factor (no gating). The provider returns final per-mode
                    # sigmas in depth units (MDA bakes in its sigma_rel; GMM uses its fitted sigmas),
                    # so they are passed through directly here.
                    records.append(
                        DepthFactorRecord(
                            i, j, "mixture",
                            tuple(d / sf for d in sample.depths),
                            tuple(s_ / sf for s_ in sample.sigmas),
                            tuple(sample.log_weights),
                        )
                    )
                    n_bimodal += 1
                    continue
                if sample.ambiguous and self._depth_model == DepthFactorMode.DROP_AMBIGUOUS:
                    n_dropped_ambiguous += 1
                    continue
                if sample.ambiguous and self._depth_model == DepthFactorMode.BIMODAL:
                    assert sample.depth_alt is not None
                    records.append(DepthFactorRecord(i, j, "bimodal", (sample.depth / sf, sample.depth_alt / sf)))
                    n_bimodal += 1
                else:
                    records.append(DepthFactorRecord(i, j, "unimodal", (sample.depth / sf,)))
                    n_unimodal += 1

        logger.info(
            "Depth factors (%s): %d unimodal, %d bimodal, %d ambiguous dropped, %d skipped, %d GT-gated.",
            self._depth_model.value,
            n_unimodal,
            n_bimodal,
            n_dropped_ambiguous,
            n_skipped,
            n_gt_gated,
        )
        self._depth_factor_stats = {
            "unimodal": n_unimodal,
            "bimodal": n_bimodal,
            "dropped_ambiguous": n_dropped_ambiguous,
            "skipped": n_skipped,
            "gt_gated": n_gt_gated,
        }
        return records, sf

    def __build_depth_graph(
        self, records: List[DepthFactorRecord], sf: float, a: Optional[Dict[int, float]] = None
    ) -> NonlinearFactorGraph:
        """Depth factor graph from prepared records, optionally with per-image scales ``a``.

        a_i enters as a scaled measurement: unimodal/bimodal measurements become a_i * d, and every
        mixture mode mean AND its (mu-proportional) sigma scale by a_i — no new factor type. With
        ``a=None`` (or all-ones) and ``depth_relative_sigma=0`` this reproduces the legacy graph
        bit-identically (same floats, same push order, same shared noise objects).
        """
        graph = NonlinearFactorGraph()
        if not records:
            return graph

        # Noise models. depth_factor_sigma is a METRIC (meters) sigma; the residual is in recon units
        # (z_pred - d/sf), so whiten by sigma/sf — matching the mixture path, which divides its per-mode
        # sigmas by sf. unit_noise is scale-free: the mixture factor whitens by its own per-mode sigma.
        depth_noise = Isotropic.Sigma(1, self._depth_factor_sigma / sf)
        unit_noise = Isotropic.Sigma(1, 1.0)
        if self._depth_factor_robust_loss:
            depth_noise = Robust(mEstimator.Huber(self._robust_noise_basin), depth_noise)
            unit_noise = Robust(mEstimator.Huber(self._robust_noise_basin), unit_noise)

        def unimodal_noise(d_scaled: float):
            # Depth-proportional sigma on the (scaled) measured depth — fixed within a round. Both
            # rel*a*d and the floor are metric, whitened by /sf like depth_factor_sigma (d_scaled is
            # already post-sf, so rel * d_scaled == rel * d_metric / sf).
            if self._depth_relative_sigma <= 0.0:
                return depth_noise
            noise = Isotropic.Sigma(1, max(self._depth_relative_sigma * d_scaled, self._depth_sigma_floor / sf))
            if self._depth_factor_robust_loss:
                noise = Robust(mEstimator.Huber(self._robust_noise_basin), noise)
            return noise

        for rec in records:
            a_i = 1.0 if a is None else float(a.get(rec.i, 1.0))
            if rec.kind == "mixture":
                depths = list(rec.depths) if a_i == 1.0 else [a_i * d for d in rec.depths]
                sigmas = list(rec.sigmas) if a_i == 1.0 else [a_i * s_ for s_ in rec.sigmas]
                graph.push_back(
                    make_mixture_depth_factor(
                        X(rec.i), P(rec.j),
                        depths,
                        sigmas,
                        list(rec.log_weights),
                        unit_noise,
                        null_nsigma=self._depth_null_nsigma,
                    )
                )
            elif rec.kind == "bimodal":
                d, d_alt = a_i * rec.depths[0], a_i * rec.depths[1]
                graph.push_back(make_bimodal_depth_factor(X(rec.i), P(rec.j), d, d_alt, unimodal_noise(d)))
            else:
                d = a_i * rec.depths[0]
                graph.push_back(make_depth_factor(X(rec.i), P(rec.j), d, unimodal_noise(d)))
        return graph

    def __depth_factors(self, initial_data: GtsfmData, cameras_to_model: List[int]) -> NonlinearFactorGraph:
        """Generate camera-frame-Z depth factors (see `__prepare_depth_records` / `__build_depth_graph`).

        With `depth_per_image_scale` on, the first build of a stage also runs the round-0 scale refit
        against the initial structure (before any optimization); alternation rebuilds (flagged via
        `_pis_round_active`) reuse the stage's cached records so the gate/sf are computed once.
        """
        if self._pis_round_active and self._pis_records is not None:
            records, sf = self._pis_records, self._pis_sf
        else:
            records, sf = self.__prepare_depth_records(initial_data, cameras_to_model)
            self._pis_records, self._pis_sf = records, sf
            if self._depth_per_image_scale and records:
                a0, info = refit_depth_scales(
                    records,
                    initial_data.to_values(shared_calib=self._shared_calib),
                    self._pis_a,
                    min_factors=self._depth_pis_min_factors,
                    clamp=self._depth_pis_clamp,
                    null_nsigma=self._depth_null_nsigma,
                    resid_scale=sf,
                )
                self._pis_a = a0
                self.__log_pis_round(info)
        a = self._pis_a if (self._depth_per_image_scale and records) else None
        return self.__build_depth_graph(records, sf, a)

    def __log_pis_round(self, info: Dict[int, dict], cost_pre_refit: Optional[float] = None) -> None:
        """Append one refit round to the per-image scale log (every round is a deliverable)."""
        self._pis_last_info = info
        self._pis_log.append(
            {
                "round": self._pis_round_counter,
                "cost_pre_refit": cost_pre_refit,  # graph error of the solution this refit was computed from
                "images": [info[i] for i in sorted(info)],
            }
        )
        self._pis_round_counter += 1

    def __pis_alternation(
        self,
        initial_data: GtsfmData,
        optimized_data: GtsfmData,
        result_values: Values,
        final_error: float,
        gnc_valid_mask: List[bool],
        cameras_to_model: List[int],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        ordering_type: str,
    ) -> Tuple[GtsfmData, Values, float, List[bool]]:
        """Outer alternation for the per-image profiled depth scale (the sigma_a -> inf limit).

        Round 0 (the fit against the initial structure) ran at graph construction, so the caller's
        first optimize is round 1. Each round refits a from the current estimate, rebuilds the graph
        with the cached records (no resampling; gate/sf unchanged), and re-optimizes warm-started
        from the last solution — so the returned values are always consistent with the final a,
        including on the tol-exit (the spec's final consistency optimize is structural here).
        """
        if not (self._depth_per_image_scale and self._pis_records):
            return optimized_data, result_values, final_error, gnc_valid_mask

        delta = float("inf")
        converged = False
        for _ in range(self._depth_pis_max_rounds):
            a_prev = self._pis_a
            a_new, info = refit_depth_scales(
                self._pis_records,
                result_values,
                a_prev,
                min_factors=self._depth_pis_min_factors,
                clamp=self._depth_pis_clamp,
                null_nsigma=self._depth_null_nsigma,
                resid_scale=self._pis_sf,
            )
            delta = max((abs(np.log(a_new[i]) - np.log(a_prev.get(i, 1.0))) for i in a_new), default=0.0)
            self._pis_a = a_new
            self.__log_pis_round(info, cost_pre_refit=final_error)
            converged = delta < self._depth_pis_tol

            self._pis_round_active = True
            try:
                graph = self.__construct_factor_graph(
                    cameras_to_model, initial_data, absolute_pose_priors, relative_pose_priors
                )
            finally:
                self._pis_round_active = False
            optimized_data, result_values, final_error, gnc_valid_mask = self.__optimize_and_recover(
                optimized_data, graph, ordering_type
            )
            if converged:
                break
        if not converged:
            logger.warning(
                "Per-image depth scale: hit max_rounds=%d without convergence (max |dlog a| = %.4g).",
                self._depth_pis_max_rounds,
                delta,
            )
        return optimized_data, result_values, final_error, gnc_valid_mask

    def _between_factors(
        self, relative_pose_priors: Dict[Tuple[int, int], PosePrior], cameras_to_model: List[int]
    ) -> NonlinearFactorGraph:
        """Generate BetweenFactors on relative poses for pose variables."""
        graph = NonlinearFactorGraph()

        for (i1, i2), i2Ti1_prior in relative_pose_priors.items():
            if i1 not in cameras_to_model or i2 not in cameras_to_model:
                continue

            graph.push_back(
                BetweenFactorPose3(
                    X(i1),
                    X(i2),
                    i2Ti1_prior.value.inverse(),
                    Diagonal.Sigmas(i2Ti1_prior.covariance),
                )
            )

        return graph

    def __pose_priors(
        self,
        absolute_pose_priors: List[Optional[PosePrior]],
        initial_data: GtsfmData,
        cameras_to_model: List[int],
    ) -> NonlinearFactorGraph:
        """Generate prior factors (in the world frame) on pose variables."""
        graph = NonlinearFactorGraph()

        # TODO(Ayush): start using absolute prior factors.

        if self._use_karcher_mean_factor:
            camera_keys = [X(i) for i in cameras_to_model]
            graph.push_back(gtsam.KarcherMeanFactorPose3(camera_keys, 6, 1000))

        if self._use_pose_prior_all_cameras:
            for camera_idx in cameras_to_model:
                camera_i = initial_data.get_camera(camera_idx)
                assert camera_i is not None, f"Camera {camera_idx} in initial data is None"
                graph.push_back(
                    PriorFactorPose3(
                        X(camera_idx),
                        camera_i.pose(),
                        Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
                    )
                )
        elif self._use_pose_prior_first_camera:
            first_camera = initial_data.get_camera(cameras_to_model[0])
            assert first_camera is not None, "First camera in initial data is None"
            graph.push_back(
                PriorFactorPose3(
                    X(cameras_to_model[0]),
                    first_camera.pose(),
                    Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
                )
            )

        return graph

    def __calibration_priors(self, initial_data: GtsfmData, cameras_to_model: list[int]) -> NonlinearFactorGraph:
        """Generate prior factors on calibration parameters of the cameras."""
        graph = NonlinearFactorGraph()

        # Note: Assumes all calibration types are the same.
        first_valid_camera_idx = cameras_to_model[0]
        first_camera = initial_data.get_camera(first_valid_camera_idx)
        assert first_camera is not None, "First camera in initial data is None"
        calibration_prior_factor_class = gtsfm_types.get_prior_factor_for_calibration(first_camera.calibration())
        calibration_dim = first_camera.calibration().dim()
        noise_model = gtsfm_types.get_noise_model_for_calibration(
            first_camera.calibration(),
            focal_sigma=self._calibration_prior_focal_sigma,
            dist_sigma=self._calibration_prior_dist_sigma,
            pp_sigma=self._calibration_prior_pp_sigma,
            skew_sigma=1e-6,
        )
        if self._shared_calib:
            graph.push_back(
                calibration_prior_factor_class(
                    K(self.__map_to_calibration_variable(first_valid_camera_idx)),
                    gtsfm_data.get_average_calibration(initial_data, cameras_to_model),  # type: ignore
                    noise_model,
                )
            )
        else:
            for i in cameras_to_model:
                camera_i = initial_data.get_camera(i)
                assert camera_i is not None, f"Camera {i} in initial data is None"
                if camera_i.calibration().dim() != calibration_dim:
                    raise ValueError(
                        "BundleAdjustmentOptimizer: Assumption that all calibration types are the same is violated"
                    )
                graph.push_back(
                    calibration_prior_factor_class(
                        K(self.__map_to_calibration_variable(i)), camera_i.calibration(), noise_model  # type: ignore
                    )
                )

        return graph

    def __construct_simple_factor_graph(
        self, cameras_to_model: List[int], initial_data: GtsfmData, robust_noise_basin: float | None = None
    ) -> NonlinearFactorGraph:
        """Construct the factor graph with just reprojection factors and calibration priors."""

        graph = NonlinearFactorGraph()
        if not cameras_to_model:
            return graph

        reprojection_graph = self.__reprojection_factors(
            initial_data=initial_data,
            cameras_to_model=cameras_to_model,
            robust_noise_basin=robust_noise_basin,
        )
        graph.push_back(reprojection_graph)

        if self._depth_model != DepthFactorMode.NONE:
            graph.push_back(self.__depth_factors(initial_data=initial_data, cameras_to_model=cameras_to_model))

        graph.push_back(
            self.__pose_priors(absolute_pose_priors=[], initial_data=initial_data, cameras_to_model=cameras_to_model)
        )

        if self._use_first_point_prior and initial_data.number_tracks() > 0:
            graph.push_back(
                PriorFactorPoint3(P(0), initial_data.get_track(0).point3(), Isotropic.Sigma(POINT3_DOF, 0.1))
            )
        if self._use_calibration_prior:
            graph.push_back(self.__calibration_priors(initial_data, cameras_to_model))

        return graph

    def __construct_factor_graph(
        self,
        cameras_to_model: List[int],
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        robust_noise_basin: float | None = None,
    ) -> NonlinearFactorGraph:
        """Construct the factor graph with reprojection factors, BetweenFactors, and prior factors."""
        # Create a factor graph.
        graph = self.__construct_simple_factor_graph(cameras_to_model, initial_data, robust_noise_basin)

        # Add priors
        graph.push_back(
            self._between_factors(relative_pose_priors=relative_pose_priors, cameras_to_model=cameras_to_model)
        )

        return graph

    def __optimize_factor_graph(
        self, graph: NonlinearFactorGraph, initial_values: Values, ordering_type: str
    ) -> Tuple[Values, Optional[List[Values]], Optional[NDArray[np.float64]]]:
        """Optimize the factor graph, optionally capturing per-iteration values."""
        start_time = time.time()

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR" if not self._print_summary else "SUMMARY")
        params.setOrderingType(ordering_type)
        if self._max_iterations:
            params.setMaxIterations(self._max_iterations)

        if not self._use_gnc:
            if self._optimizer_relative_cost_tol is not None:
                params.setRelativeErrorTol(self._optimizer_relative_cost_tol)
            lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
        else:
            gnc_params = gtsam.GncLMParams(params)
            if self._gnc_loss == RobustBAMode.GMC:
                gnc_params.setLossType(gtsam.GncLossType.GM)
            elif self._gnc_loss == RobustBAMode.TLS:
                gnc_params.setLossType(gtsam.GncLossType.TLS)
            else:
                raise ValueError(f"Unsupported GNC loss type: {self._gnc_loss}.")
            gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.SUMMARY)
            gnc_params.setAllowNonNoiseModelFactors(True)
            if self._optimizer_relative_cost_tol is not None:
                gnc_params.setRelativeCostTol(self._optimizer_relative_cost_tol)
            lm = gtsam.GncLMOptimizer(graph, initial_values, gnc_params)

        # gnc does not support getAbsoluteErrorTol and getRelativeErrorTol params
        if not self._save_iteration_visualization or self._use_gnc:
            result_values = lm.optimize()
            values_trace = None
        else:
            values_trace = [initial_values]
            max_iters = self._max_iterations
            if max_iters is None:
                try:
                    max_iters = int(params.getMaxIterations())
                except Exception:
                    max_iters = 100

            abs_tol = float(params.getAbsoluteErrorTol())
            rel_tol = float(params.getRelativeErrorTol())
            prev_error = float(lm.error())

            for _ in range(max_iters):
                lm.iterate()
                values_trace.append(lm.values())
                curr_error = float(lm.error())
                error_delta = prev_error - curr_error
                if abs(error_delta) < abs_tol:
                    logger.info("🚀 Absolute error tolerance reached.")
                    break
                if prev_error > 0.0 and abs(error_delta) / prev_error < rel_tol:
                    logger.info("🚀 Relative error tolerance reached.")
                    break
                prev_error = curr_error
            result_values = lm.values()

        elapsed_time = time.time() - start_time
        logger.info(f"🚀 Factor graph optimization completed in {elapsed_time:.2f} seconds.")
        if self._use_gnc:
            weights = lm.getWeights()
            return result_values, values_trace, weights
        return result_values, values_trace, None

    def get_two_view_ba_pose_graph_keys(self, initial_data: GtsfmData):
        """Retrieves GTSAM keys for camera poses in a 2-view BA problem."""
        valid_camera_indices = initial_data.get_valid_camera_indices()
        return [X(valid_camera_indices[0]), X(valid_camera_indices[1])]

    def is_two_view_ba(self, initial_data: GtsfmData) -> bool:
        """Determines whether two-view bundle adjustment is being executed."""
        return len(initial_data.get_valid_camera_indices()) == 2

    def __compute_camera_pose_covariances(
        self, marginals: gtsam.Marginals, cameras_to_model: List[int]
    ) -> Dict[int, NDArray[np.float64]]:
        """Compute marginal covariance for all camera pose variables."""
        pose_covariances: Dict[int, NDArray[np.float64]] = {}
        for camera_idx in cameras_to_model:
            try:
                full_cov = np.asarray(marginals.marginalCovariance(X(camera_idx)))
                pose_covariances[camera_idx] = full_cov[:CAM_POSE3_DOF, :CAM_POSE3_DOF]
            except Exception:
                logger.error(
                    f"Error computing covariance for camera {camera_idx}, likely due to indeterminate linear system."
                )
                continue
        return pose_covariances

    def __optimize_and_recover(
        self, initial_data: GtsfmData, graph: NonlinearFactorGraph, ordering_type: str
    ) -> Tuple[GtsfmData, Values, float, List[bool]]:
        """Optimize the graph, report errors, and convert `Values` back to `GtsfmData`."""
        initial_values = initial_data.to_values(shared_calib=self._shared_calib)
        result_values, _, weights = self.__optimize_factor_graph(graph, initial_values, ordering_type)
        final_error = graph.error(result_values)
        optimized_data = GtsfmData.from_values(result_values, initial_data, self._shared_calib)
        gnc_valid_mask = [True] * initial_data.number_tracks()
        if self._use_gnc and weights is not None and self._factor_weight_outlier_threshold > 0:
            optimized_data, gnc_valid_mask = self.__filter_tracks_by_factor_weights(graph, optimized_data, weights)
        return optimized_data, result_values, final_error, gnc_valid_mask

    def __filter_tracks_by_factor_weights(
        self, graph: NonlinearFactorGraph, optimized_data: GtsfmData, weights: NDArray[np.float64]
    ) -> Tuple[GtsfmData, List[bool]]:
        """Filter tracks based on the weights of the reprojection factors, if GNC optimization is used."""
        gnc_valid_mask = [True] * optimized_data.number_tracks()
        if weights is None:
            logger.error("Weights array is None, cannot filter tracks by factor weights.")
            return optimized_data, gnc_valid_mask
        cameras_to_model = sorted(optimized_data.get_valid_camera_indices())
        first_camera = optimized_data.get_camera(cameras_to_model[0])
        assert first_camera is not None, "First camera in optimized factor graph is None"
        sfm_factor_class = gtsfm_types.get_sfm_factor_for_calibration(first_camera.calibration())
        cams_to_remove_per_track: dict[int, set[int]] = defaultdict(set)
        for i in range(graph.nrFactors()):
            if weights[i] >= self._factor_weight_outlier_threshold:
                continue
            factor = graph.at(i)
            if isinstance(factor, sfm_factor_class):
                camera_key, track_key, _ = factor.keys()
                camera_id = int(gtsam.symbolIndex(camera_key))
                track_id = int(gtsam.symbolIndex(track_key))
                cams_to_remove_per_track[track_id].add(camera_id)

        if not cams_to_remove_per_track:
            return optimized_data, gnc_valid_mask

        for track_id, camera_ids in cams_to_remove_per_track.items():
            track = optimized_data.get_track(track_id)
            new_measurements = []
            for m_idx in range(track.numberMeasurements()):
                cam_id, _ = track.measurement(m_idx)
                if cam_id not in camera_ids:
                    new_measurements.append(track.measurement(m_idx))
            track.measurements = new_measurements

        length_filtered_tracks = []
        for track_idx, track in enumerate(optimized_data.get_tracks()):
            if track.numberMeasurements() >= self._min_track_length:
                length_filtered_tracks.append(track)
            else:
                gnc_valid_mask[track_idx] = False

        if len(length_filtered_tracks) == optimized_data.number_tracks():
            optimized_data._camera_to_measurement_map = None
            return optimized_data, gnc_valid_mask

        image_info = {
            image_id: optimized_data.get_image_info(image_id) for image_id in optimized_data.get_all_image_ids()
        }

        filtered_data = GtsfmData.from_cameras_and_tracks(
            cameras=optimized_data.cameras(),
            tracks=length_filtered_tracks,
            number_images=optimized_data.number_images(),
            image_info=image_info,
            gaussian_splats=optimized_data.get_gaussian_splats(),
        )

        return filtered_data, gnc_valid_mask

    def run_simple_ba(
        self, initial_data: GtsfmData, robust_noise_basin: float | None = None
    ) -> Tuple[GtsfmData, float]:
        """Runs bundle adjustment and optionally filters the resulting tracks by reprojection error.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            robust_noise_basin: Robust noise basin to use for the BA, not used if self._robust_ba_mode is NONE.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Final error value of the optimization problem.
        """
        cameras_to_model = sorted(initial_data.get_valid_camera_indices())
        cameras_with_insufficient_tracks = self.__get_cameras_with_insufficient_tracks(initial_data)
        cameras_to_model = [i for i in cameras_to_model if i not in cameras_with_insufficient_tracks]
        logger.info(
            "Cameras with insufficient tracks (fewer than %d): %s, will be excluded from BA.",
            self._min_tracks_per_camera,
            cameras_with_insufficient_tracks,
        )
        graph = self.__construct_simple_factor_graph(cameras_to_model, initial_data, robust_noise_basin)
        if len(cameras_with_insufficient_tracks) == len(initial_data.cameras()):
            logger.warning("Skipping bundle adjustment because all cameras are without tracks.")
            return initial_data, 0.0
        optimized_data, result_values, final_error, _ = self.__optimize_and_recover(
            initial_data, graph, self._ordering_type if not cameras_with_insufficient_tracks else "COLAMD"
        )
        if self._compute_pose_covariances:
            try:
                marginals = gtsam.Marginals(graph, result_values)
                optimized_data.set_camera_pose_covariances(
                    self.__compute_camera_pose_covariances(marginals, cameras_to_model)
                )
            except Exception:
                logger.info("Error computing marginals, likely due to indeterminate linear system.")

        return optimized_data, final_error

    def run_iterative_robust_ba(
        self, initial_data: GtsfmData, robust_noise_basins: List[float]
    ) -> Tuple[GtsfmData, float]:
        """Runs iterative robust bundle adjustment, using different robust noise basins for each iteration."""
        optimized_data = initial_data
        final_error = float("nan")
        for robust_noise_basin in robust_noise_basins:
            optimized_data, final_error = self.run_simple_ba(optimized_data, robust_noise_basin)
        return optimized_data, final_error

    def run_ba_stage_with_filtering(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        reproj_error_thresh: Optional[float],
        verbose: bool = True,
    ) -> Tuple[Optional[GtsfmData], Optional[GtsfmData], Optional[List[bool]], Optional[float]]:
        """Runs bundle adjustment and optionally filters the resulting tracks by reprojection error.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: Priors to be used on cameras.
            relative_pose_priors: Priors on the pose between two cameras.
            reproj_error_thresh: Maximum 3D track reprojection error, for filtering tracks after BA.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Optimized camera poses after filtering landmarks (and cameras with no remaining landmarks).
            Valid mask as a list of booleans, indicating for each input track whether it was below the re-projection
                threshold.
            Final error value of the optimization problem.
        """
        logger.info(
            "Input: %d tracks on %d cameras", initial_data.number_tracks(), len(initial_data.get_valid_camera_indices())
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # No cameras or tracks to optimize, so bundle adjustment is not possible, return invalid result.
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return initial_data, initial_data, [False] * initial_data.number_tracks(), 0.0

        running_two_view_ba = self.is_two_view_ba(initial_data)

        cameras_to_model = sorted(initial_data.get_valid_camera_indices())
        cameras_with_insufficient_tracks = None
        if not running_two_view_ba:
            cameras_with_insufficient_tracks = self.__get_cameras_with_insufficient_tracks(initial_data)
            cameras_to_model = [i for i in cameras_to_model if i not in cameras_with_insufficient_tracks]
            logger.info(
                "Cameras with insufficient tracks (fewer than %d): %s, will be excluded from BA.",
                self._min_track_length,
                cameras_with_insufficient_tracks,
            )

        graph = self.__construct_factor_graph(
            cameras_to_model, initial_data, absolute_pose_priors, relative_pose_priors
        )
        ordering_type = self._ordering_type if not cameras_with_insufficient_tracks else "COLAMD"
        optimized_data, result_values, final_error, gnc_valid_mask = self.__optimize_and_recover(
            initial_data, graph, ordering_type
        )
        if self._depth_per_image_scale:
            optimized_data, result_values, final_error, gnc_valid_mask = self.__pis_alternation(
                initial_data, optimized_data, result_values, final_error, gnc_valid_mask,
                cameras_to_model, absolute_pose_priors, relative_pose_priors, ordering_type,
            )
        if not running_two_view_ba:
            # Add the non-BA cameras from initial_data back.
            for camera_idx in cameras_with_insufficient_tracks:
                optimized_data._cameras[camera_idx] = initial_data.get_camera(camera_idx)

        if running_two_view_ba or self._compute_pose_covariances:
            try:
                marginals = gtsam.Marginals(graph, result_values)
                if running_two_view_ba:
                    # Calculate marginal covariances for all two pose variables.
                    graph_keys = self.get_two_view_ba_pose_graph_keys(initial_data)
                    for key in graph_keys:
                        _ = marginals.marginalCovariance(key)
                if self._compute_pose_covariances:
                    optimized_data.set_camera_pose_covariances(
                        self.__compute_camera_pose_covariances(marginals, cameras_to_model)
                    )

            except RuntimeError:
                if running_two_view_ba and not self._allow_indeterminate_linear_system:
                    logger.error(
                        "BA result discarded due to Indeterminate Linear System (ILS) when computing marginals."
                    )
                    return None, None, None, None
                elif not running_two_view_ba:
                    logger.info("Error computing marginals, likely due to indeterminate linear system.")

        # Convert the `Values` results to a `GtsfmData` instance.
        # Filter landmarks by reprojection error.
        if reproj_error_thresh is not None:
            if verbose:
                logger.info("[Result] Number of tracks before filtering: %d", optimized_data.number_tracks())
            filtered_result, postfilter_valid_mask = optimized_data.filter_landmarks(reproj_error_thresh)
            if verbose:
                logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())

        else:
            postfilter_valid_mask = [True] * optimized_data.number_tracks()
            filtered_result = optimized_data

        valid_mask = [False] * initial_data.number_tracks()
        kept_track_indices = [track_idx for track_idx, is_valid in enumerate(gnc_valid_mask) if is_valid]
        for track_idx, is_valid in zip(kept_track_indices, postfilter_valid_mask):
            valid_mask[track_idx] = is_valid
        if self._compute_pose_covariances:
            filtered_pose_covariances = {
                i: cov
                for i, cov in optimized_data.get_camera_pose_covariances().items()
                if i in filtered_result.get_valid_camera_indices()
            }
            filtered_result.set_camera_pose_covariances(filtered_pose_covariances)
        return optimized_data, filtered_result, valid_mask, final_error

    def run_ba(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        verbose: bool = True,
    ) -> Tuple[Optional[GtsfmData], Optional[GtsfmData], Optional[List[bool]], List[float]]:
        """Runs bundle adjustment by forming a factor graph and optimizing it using Levenberg–Marquardt optimization.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: Priors to be used on cameras.
            relative_pose_priors: Priors on the pose between two cameras.
            verbose: Boolean flag to print out additional info for debugging.

        Returns:
            Optimized cameras and tracks with error metrics, aligned to GT if provided.
            Filtered result after removing high-reprojection-error landmarks and cameras with no remaining tracks.
            Boolean mask over input tracks: True if the track survived all BA filtering steps.
            Per-step wall-clock times for each BA iteration.
        """
        num_ba_steps = len(self._reproj_error_thresholds)
        assert num_ba_steps > 0, "No BA steps to perform"

        current_input = initial_data
        active_track_indices = list(range(initial_data.number_tracks()))
        cumulative_valid_mask = [True] * initial_data.number_tracks()
        step_times: List[float] = []

        for step, reproj_error_thresh in enumerate(self._reproj_error_thresholds):
            step_start_time = time.time()
            optimized_data, filtered_result, stage_valid_mask, final_error = self.run_ba_stage_with_filtering(
                current_input,
                absolute_pose_priors,
                relative_pose_priors,
                reproj_error_thresh,
                verbose,
            )
            step_times.append(time.time() - step_start_time)

            if optimized_data is None or filtered_result is None or stage_valid_mask is None:
                return optimized_data, filtered_result, stage_valid_mask, step_times  # type: ignore

            if len(stage_valid_mask) != len(active_track_indices):
                raise ValueError(
                    "Bundle adjustment stage returned a validity mask whose length does not match the input tracks."
                )

            next_active_track_indices = []
            for original_track_idx, is_valid in zip(active_track_indices, stage_valid_mask):
                cumulative_valid_mask[original_track_idx] = is_valid
                if is_valid:
                    next_active_track_indices.append(original_track_idx)
            active_track_indices = next_active_track_indices
            current_input = filtered_result

            if num_ba_steps > 1:
                logger.info(
                    "[BA Stage @ thresh=%.2f px %d/%d] Error: %.2f, Number of tracks: %d"
                    % (
                        reproj_error_thresh if reproj_error_thresh is not None else float("nan"),
                        step + 1,
                        num_ba_steps,
                        final_error,
                        filtered_result.number_tracks(),
                    )
                )

        return optimized_data, filtered_result, cumulative_valid_mask, step_times  # type: ignore

    def __stamp_image_fnames(self, data: Optional[GtsfmData]) -> None:
        """Write the real per-image filenames (from `self._image_fnames`) onto a BA result's
        image_info, so the COLMAP export uses actual names instead of placeholders. Only valid
        cameras in `data` are stamped; no-op if filenames were not provided."""
        if data is None or self._image_fnames is None:
            return
        for i in data.get_valid_camera_indices():
            if i in self._image_fnames:
                data.set_image_info(i, name=self._image_fnames[i])

    def _run_ba_and_evaluate(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
        verbose: bool = True,
        tracks_2d: Optional[List["SfmTrack2d"]] = None,
        image_fnames: Optional[Dict[int, str]] = None,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool], GtsfmMetricsGroup]:
        """Runs the equivalent of `run_ba()` and `evaluate()` in a single function, to enable time profiling.

        Args:
            tracks_2d: (optional) Union-find 2D track set from `CppDsfTracksEstimator.run()`.
                Required when `use_multi_view_retriangulation` is enabled — used by
                the post-BA retriangulation stage.
            image_fnames: (optional) Map from image index to filename. Required when
                `depth_model != "none"` — used to locate per-image depth maps.
        """
        self._image_fnames = image_fnames
        # Fresh per-image-scale state per BA invocation.
        self._pis_a = {}
        self._pis_records = None
        self._pis_log = []
        self._pis_round_counter = 0
        self._pis_last_info = {}
        logger.info(
            "Input: %d tracks on %d cameras", initial_data.number_tracks(), len(initial_data.get_valid_camera_indices())
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # No cameras or tracks to optimize, so bundle adjustment is not possible.
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return (
                initial_data,
                initial_data,
                [False] * initial_data.number_tracks(),
                GtsfmMetricsGroup(METRICS_GROUP, []),
            )
        start_time = time.time()
        optimized_data, filtered_result, valid_mask, step_times = self.run_ba(
            initial_data=initial_data,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
            verbose=verbose,
        )
        total_time = time.time() - start_time

        # ── Optional post-BA multi-view retriangulation stage ──
        # Re-triangulate union-find tracks against the post-BA cameras (recovers
        # tracks dropped between union-find and BA's filter passes), then run a
        # final BA on the augmented set. The final BA reuses the existing tightest
        # `reproj_error_thresholds[-1]` for inline filtering — same mechanism as
        # the upstream BA loop.
        if self._use_multi_view_retriangulation:
            if tracks_2d is None:
                logger.warning(
                    "use_multi_view_retriangulation is True but tracks_2d was not "
                    "passed to _run_ba_and_evaluate — skipping retri stage."
                )
            else:
                retri_start = time.time()
                retri_options = TriangulationOptions(
                    reproj_error_threshold=self._mv_retri_reproj_error_thresh,
                    mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
                    max_num_hypotheses=self._mv_retri_max_num_hypotheses,
                )
                logger.info(
                    "[Retri] Multi-view retriangulation on %d 2D tracks "
                    "(min_track_len=%d, reproj_error_thresh=%.1fpx)",
                    len(tracks_2d), self._mv_retri_min_track_length,
                    self._mv_retri_reproj_error_thresh,
                )
                retri_data = multi_view_retriangulate_from_2d_tracks(
                    gtsfm_data=filtered_result,
                    tracks_2d=tracks_2d,
                    triangulation_options=retri_options,
                    min_track_length=self._mv_retri_min_track_length,
                )
                if retri_data.number_tracks() > 0:
                    # Final BA on the retri'd track set. No inline filter — pose AUC
                    # is set by BA's converged cameras and is independent of any
                    # downstream track filtering. Callers can filter the returned
                    # GtsfmData themselves if they want.
                    (optimized_data, filtered_result, valid_mask, _) = self.run_ba_stage_with_filtering(
                        initial_data=retri_data,
                        absolute_pose_priors=absolute_pose_priors,
                        relative_pose_priors=relative_pose_priors,
                        reproj_error_thresh=None,
                        verbose=verbose,
                    )
                    logger.info(
                        "[Retri] Stage complete: %d tracks, %.1fs",
                        filtered_result.number_tracks(), time.time() - retri_start,
                    )
                step_times.append(time.time() - retri_start)

        total_time = time.time() - start_time

        # Stamp the real (loader-relative) image filenames onto the BA outputs. The GtsfmData flowing
        # through data association -> BA carries no names (they are threaded separately via
        # `image_fnames`, used only for depth). Downstream, ClusterMVO annotates ba_output via
        # clone_with_image_data, but that only fills the *basename* (Image.file_name), which drops any
        # image subdirectory -- so nested layouts (e.g. ETH3D's images/dslr_images_undistorted/*.JPG)
        # no longer resolve against --images_dir, while flat layouts (Replica results/frameNNN.jpg)
        # happened to work. Stamping the full loader-relative name here (before that annotation, which
        # is a no-op once name is set) fixes both. VGGT/deep front-ends set names via the tracker.
        self.__stamp_image_fnames(optimized_data)
        self.__stamp_image_fnames(filtered_result)

        # Per-image profiled scale log: every refit round for every image (the a-trajectory is a
        # deliverable — it gets overlaid against the offline bias_fit.csv per-image scales).
        if self._depth_per_image_scale and self._pis_log and save_dir is not None:
            fnames = self._image_fnames or {}
            for round_entry in self._pis_log:
                for img_entry in round_entry["images"]:
                    img_entry.setdefault("image_fname", fnames.get(img_entry["image_id"]))
            log_path = Path(save_dir) / "depth_per_image_scale_log.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps({"sf": self._pis_sf, "rounds": self._pis_log}, indent=1))
            logger.info("Wrote per-image depth scale log to %s.", log_path)

        metrics = self.evaluate(optimized_data, filtered_result, cameras_gt, save_dir)  # type: ignore
        for i, step_time in enumerate(step_times):
            metrics.add_metric(GtsfmMetric(f"step_{i}_run_duration_sec", step_time))
        metrics.add_metric(GtsfmMetric("total_run_duration_sec", total_time))

        return optimized_data, filtered_result, valid_mask, metrics  # type: ignore

    def evaluate(
        self,
        unfiltered_data: GtsfmData,
        filtered_data: GtsfmData,
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
    ) -> GtsfmMetricsGroup:
        """Computes metrics on the bundle adjustment result, and packages them in a GtsfmMetricsGroup object.

        Args:
            unfiltered_data: Optimized BA result, before filtering landmarks by reprojection error.
            filtered_data: Optimized BA result, after filtering landmarks and cameras.
            cameras_gt: Cameras with GT intrinsics and GT extrinsics.

        Returns:
            Metrics group containing metrics for both filtered and unfiltered BA results.
        """
        ba_metrics = GtsfmMetricsGroup(name=METRICS_GROUP, metrics=unfiltered_data.get_metrics(suffix="_unfiltered"))
        for stat_name, stat_value in self._depth_factor_stats.items():
            ba_metrics.add_metric(GtsfmMetric(name=f"num_depth_factors_{stat_name}", data=stat_value))
        if self._depth_per_image_scale and self._pis_last_info:
            # Scalar summaries of the final per-image scales (the aggregator carries these columns).
            entries = list(self._pis_last_info.values())
            a_vals = np.array([e["a"] for e in entries])
            for name, value in [
                ("pis_n_rounds", self._pis_round_counter),
                ("pis_a_mean", float(a_vals.mean())),
                ("pis_a_std", float(a_vals.std())),
                ("pis_a_min", float(a_vals.min())),
                ("pis_a_max", float(a_vals.max())),
                ("pis_n_fallback", sum(e["fallback"] for e in entries)),
                ("pis_n_clamped", sum(e["clamped"] for e in entries)),
            ]:
                ba_metrics.add_metric(GtsfmMetric(name=name, data=value))

        input_image_idxs = list(unfiltered_data._image_info.keys())
        poses_gt = {
            i: cameras_gt[i].pose() for i in input_image_idxs if i < len(cameras_gt) and cameras_gt[i] is not None
        }
        if not poses_gt:
            return ba_metrics

        # Align the sparse multi-view estimate after BA to the ground truth pose graph.
        aligned_filtered_data = filtered_data.align_via_sim3_and_transform(poses_gt)
        ba_pose_error_metrics = metrics_utils.compute_ba_pose_metrics(
            gt_wTi=poses_gt,
            computed_wTi=aligned_filtered_data.get_camera_poses(),
            save_dir=save_dir,
            metric_constructed_only=True,
        )
        ba_metrics.extend(metrics_group=ba_pose_error_metrics)

        output_tracks_exit_codes = track_utils.classify_tracks3d_with_gt_cameras(
            tracks=aligned_filtered_data.get_tracks(), cameras_gt=cameras_gt
        )
        output_tracks_exit_codes_distribution = Counter(output_tracks_exit_codes)

        for exit_code, count in output_tracks_exit_codes_distribution.items():
            metric_name = "Filtered tracks triangulated with GT cams: {}".format(exit_code.name)
            ba_metrics.add_metric(GtsfmMetric(name=metric_name, data=count))

        ba_metrics.add_metrics(aligned_filtered_data.get_metrics(suffix="_filtered"))

        logger.info("[Result] Mean track length %.3f", np.mean(aligned_filtered_data.get_track_lengths()))
        logger.info("[Result] Median track length %.3f", np.median(aligned_filtered_data.get_track_lengths()))
        aligned_filtered_data.log_scene_reprojection_error_stats()

        return ba_metrics

    def create_computation_graph(
        self,
        sfm_data_graph: Delayed,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
        tracks_2d: Optional[Delayed] = None,
        image_fnames: Optional[Dict[int, str]] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: An GtsfmData object wrapped up using dask.delayed.
            absolute_pose_priors: Priors on the poses of the cameras (not delayed).
            relative_pose_priors: Priors on poses between cameras (not delayed).
            cameras_gt: Ground truth camera calibration & pose for each image/camera.
            save_dir: Directory where artifacts and plots should be saved to disk.
            tracks_2d: (optional) Delayed list of 2D tracks from `CppDsfTracksEstimator`.
                Required when `use_multi_view_retriangulation` is enabled.
            image_fnames: (optional) Map from image index to filename. Required when
                `depth_model != "none"` — used to locate per-image depth maps.

        Returns:
            GtsfmData aligned to GT (if provided), wrapped up using dask.delayed
            Metrics group for BA results, wrapped up using dask.delayed
        """

        _, filtered_sfm_data, _, metrics_graph = dask.delayed(self._run_ba_and_evaluate, nout=4)(
            sfm_data_graph,
            absolute_pose_priors,
            relative_pose_priors,
            cameras_gt,
            save_dir=save_dir,
            tracks_2d=tracks_2d,
            image_fnames=image_fnames,
        )
        return filtered_sfm_data, metrics_graph
