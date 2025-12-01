from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class InitialPairResult:
    """Metadata describing the best two-view bootstrap pair."""

    i: int
    j: int
    rotation: np.ndarray
    translation: np.ndarray
    matches_used: int
    triangulated: int
    inlier_ratio: float


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _build_intrinsics(height: int, width: int) -> np.ndarray:
    """
    Approximate the camera intrinsics described in the CS436 brief:
    fx = fy = width, principal point centered in the image.
    """
    f = float(width)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _points_from_matches(
    keypoints_a: Sequence[cv2.KeyPoint],
    keypoints_b: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
    return pts_a, pts_b


def _triangulate(
    K: np.ndarray,
    pose_a: Tuple[np.ndarray, np.ndarray],
    pose_b: Tuple[np.ndarray, np.ndarray],
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    reproj_thresh: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate corresponding points and filter them with reprojection/cheirality checks."""
    R1, t1 = pose_a
    R2, t2 = pose_b
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pts4 = cv2.triangulatePoints(P1, P2, pts_a.T, pts_b.T)
        pts3 = (pts4[:3] / pts4[3]).T
        ones = np.ones((pts3.shape[0], 1))
        pts_h = np.hstack((pts3, ones))

        proj1 = (P1 @ pts_h.T).T
        proj2 = (P2 @ pts_h.T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        proj2 = proj2[:, :2] / proj2[:, 2:3]
    err1 = np.linalg.norm(proj1 - pts_a, axis=1)
    err2 = np.linalg.norm(proj2 - pts_b, axis=1)

    z1 = (R1 @ pts3.T + t1).T[:, 2]
    z2 = (R2 @ pts3.T + t2).T[:, 2]
    finite = np.isfinite(pts3).all(axis=1)
    mask = (z1 > 0.0) & (z2 > 0.0) & (err1 < reproj_thresh) & (err2 < reproj_thresh) & finite
    return pts3[mask], mask


class IncrementalSfM:
    """Minimal incremental SfM pipeline covering Phases 1-2 of the CS436 project."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        target_long_edge: int = 1400,
        sift_nfeatures: int = 6000,
        match_ratio: float = 0.75,
        min_init_matches: int = 80,
        min_triangulated_init: int = 80,
        min_pnp_correspondences: int = 8,
        min_triangulation_matches: int = 12,
        reproj_error: float = 2.5,
    ) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        if len(self.image_paths) < 2:
            raise ValueError("Need at least two images to run SfM.")
        self.target_long_edge = target_long_edge
        self.sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.match_ratio = match_ratio
        self.min_init_matches = min_init_matches
        self.min_triangulated_init = min_triangulated_init
        self.min_pnp_correspondences = min_pnp_correspondences
        self.min_triangulation_matches = min_triangulation_matches
        self.reproj_error = reproj_error

        self._rgb_images: List[np.ndarray] = []
        self._gray_images: List[np.ndarray] = []
        self._keypoints: List[List[cv2.KeyPoint]] = []
        self._descriptors: List[Optional[np.ndarray]] = []
        self._intrinsic: Optional[np.ndarray] = None
        self._target_hw: Optional[Tuple[int, int]] = None

        self.initial_pair: Optional[InitialPairResult] = None
        self.camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.registered_order: List[int] = []
        self.kp_to_point: Dict[Tuple[int, int], int] = {}
        self.points3d: List[np.ndarray] = []
        self.point_colors: List[np.ndarray] = []
        self.stats: Dict[str, List[Tuple[int, int, int]]] = {
            "pair_scores": [],
            "registered": [],
        }

    def prepare_inputs(self) -> None:
        """Load the dataset, resize consistently, and compute SIFT features."""
        for idx, path in enumerate(self.image_paths):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Failed to read {path}")
            bgr = self._resize_consistently(bgr)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb = _ensure_rgb(bgr)

            kp, des = self.sift.detectAndCompute(gray, None)
            self._rgb_images.append(rgb)
            self._gray_images.append(gray)
            self._keypoints.append(list(kp) if kp is not None else [])
            self._descriptors.append(des)

        h, w = self._gray_images[0].shape
        self._intrinsic = _build_intrinsics(h, w)

    def _resize_consistently(self, image: np.ndarray) -> np.ndarray:
        """Force all frames to share the same target_long_edge dimension."""
        h, w = image.shape[:2]
        if self._target_hw is None:
            if self.target_long_edge is None:
                self._target_hw = (h, w)
            else:
                if h >= w:
                    scale = self.target_long_edge / h
                    target_h = self.target_long_edge
                    target_w = int(round(w * scale))
                else:
                    scale = self.target_long_edge / w
                    target_w = self.target_long_edge
                    target_h = int(round(h * scale))
                self._target_hw = (target_h, target_w)
        target_h, target_w = self._target_hw
        if (h, w) == (target_h, target_w):
            return image
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    @property
    def intrinsic(self) -> np.ndarray:
        if self._intrinsic is None:
            raise RuntimeError("Call prepare_inputs() first.")
        return self._intrinsic

    def _match(self, i: int, j: int) -> List[cv2.DMatch]:
        desc_i = self._descriptors[i]
        desc_j = self._descriptors[j]
        if desc_i is None or desc_j is None:
            return []
        raw = self.matcher.knnMatch(desc_i, desc_j, k=2)
        good: List[cv2.DMatch] = []
        for m, n in raw:
            if m.distance < self.match_ratio * n.distance:
                good.append(m)
        return good

    def find_best_initial_pair(self, neighbor_gap: int = 4) -> InitialPairResult:
        """Score close-by image pairs and keep the one that triangulates the most points."""
        if not self._keypoints:
            raise RuntimeError("Please call prepare_inputs() first.")
        best: Optional[InitialPairResult] = None
        num_imgs = len(self.image_paths)
        for i in range(num_imgs - 1):
            for j in range(i + 1, min(num_imgs, i + 1 + neighbor_gap)):
                candidate = self._score_pair(i, j)
                if candidate is None:
                    continue
                self.stats["pair_scores"].append((i, j, candidate.triangulated))
                if best is None or candidate.triangulated > best.triangulated:
                    best = candidate
        if best is None:
            raise RuntimeError("Could not find a valid bootstrap pair.")
        self.initial_pair = best
        return best

    def _score_pair(self, i: int, j: int) -> Optional[InitialPairResult]:
        matches = self._match(i, j)
        if len(matches) < self.min_init_matches:
            return None
        pts_i, pts_j = _points_from_matches(self._keypoints[i], self._keypoints[j], matches)
        K = self.intrinsic
        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            return None
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < self.min_init_matches:
            return None
        pts_i = pts_i[inliers]
        pts_j = pts_j[inliers]
        match_inliers = [matches[idx] for idx in np.where(inliers)[0]]

        retval, R, t, pose_mask = cv2.recoverPose(E, pts_i, pts_j, K)
        if retval < 10 or pose_mask is None:
            return None
        pose_inliers = pose_mask.ravel().astype(bool)
        pts_i = pts_i[pose_inliers]
        pts_j = pts_j[pose_inliers]
        match_pose = [match_inliers[idx] for idx in np.where(pose_inliers)[0]]

        pts3d, mask3d = _triangulate(
            K,
            (np.eye(3), np.zeros((3, 1))),
            (R, t),
            pts_i,
            pts_j,
            reproj_thresh=self.reproj_error,
        )
        if pts3d.shape[0] < self.min_triangulated_init:
            return None
        ratio = float(pts3d.shape[0]) / float(len(match_pose))
        return InitialPairResult(
            i=i,
            j=j,
            rotation=R,
            translation=t,
            matches_used=len(match_pose),
            triangulated=int(pts3d.shape[0]),
            inlier_ratio=ratio,
        )

    def bootstrap_from_pair(
        self, seed_pair: Optional[Tuple[int, int]] = None, neighbor_gap: int = 4
    ) -> InitialPairResult:
        """Run the two-view bootstrap, optionally forcing a specific image pair."""
        if seed_pair is not None:
            i, j = seed_pair
        else:
            if self.initial_pair is None:
                self.find_best_initial_pair(neighbor_gap=neighbor_gap)
            assert self.initial_pair is not None
            i, j = self.initial_pair.i, self.initial_pair.j

        return self._build_initial_map(i, j)

    def _build_initial_map(self, i: int, j: int) -> InitialPairResult:
        self.camera_poses.clear()
        self.registered_order = []
        self.kp_to_point.clear()
        self.points3d.clear()
        self.point_colors.clear()
        matches = self._match(i, j)
        if len(matches) < self.min_init_matches:
            raise RuntimeError(f"Pair ({i}, {j}) does not have enough matches to bootstrap.")
        pts_i, pts_j = _points_from_matches(self._keypoints[i], self._keypoints[j], matches)
        K = self.intrinsic
        E, mask = cv2.findEssentialMat(
            pts_i, pts_j, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask is None:
            raise RuntimeError("Essential matrix estimation failed for the bootstrap pair.")
        inliers = mask.ravel().astype(bool)
        pts_i = pts_i[inliers]
        pts_j = pts_j[inliers]
        match_inliers = [matches[idx] for idx in np.where(inliers)[0]]

        _, R, t, pose_mask = cv2.recoverPose(E, pts_i, pts_j, K)
        pose_inliers = pose_mask.ravel().astype(bool)
        pts_i = pts_i[pose_inliers]
        pts_j = pts_j[pose_inliers]
        match_pose = [match_inliers[idx] for idx in np.where(pose_inliers)[0]]

        pts3d, mask3d = _triangulate(
            K,
            (np.eye(3), np.zeros((3, 1))),
            (R, t),
            pts_i,
            pts_j,
            reproj_thresh=self.reproj_error,
        )
        if pts3d.shape[0] == 0:
            raise RuntimeError("No valid triangulations for the bootstrap pair.")

        self.camera_poses[i] = (np.eye(3), np.zeros((3, 1)))
        self.camera_poses[j] = (R, t)
        self.registered_order = [i, j]

        rgb_i = self._rgb_images[i]
        rgb_j = self._rgb_images[j]
        mask_idx = np.where(mask3d)[0]
        for out_idx, match_idx in enumerate(mask_idx):
            m = match_pose[match_idx]
            pi = pts_i[match_idx]
            pj = pts_j[match_idx]
            color = self._sample_color(rgb_i, pi, rgb_j, pj)
            pid = len(self.points3d)
            self.points3d.append(pts3d[out_idx])
            self.point_colors.append(color)
            self.kp_to_point[(i, m.queryIdx)] = pid
            self.kp_to_point[(j, m.trainIdx)] = pid

        pair_info = InitialPairResult(
            i=i,
            j=j,
            rotation=R,
            translation=t,
            matches_used=len(match_pose),
            triangulated=len(self.points3d),
            inlier_ratio=(len(self.points3d) / max(1, len(match_pose))),
        )
        self.initial_pair = pair_info
        return pair_info

    def _sample_color(
        self,
        rgb_a: np.ndarray,
        pt_a: np.ndarray,
        rgb_b: Optional[np.ndarray] = None,
        pt_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        xy_a = np.clip(np.round(pt_a).astype(int), [0, 0], np.array(rgb_a.shape[1::-1]) - 1)
        color = rgb_a[xy_a[1], xy_a[0]].astype(np.float32)
        if rgb_b is not None and pt_b is not None:
            xy_b = np.clip(np.round(pt_b).astype(int), [0, 0], np.array(rgb_b.shape[1::-1]) - 1)
            color = 0.5 * (color + rgb_b[xy_b[1], xy_b[0]].astype(np.float32))
        return color

    def run_incremental(self, max_passes: int = 4) -> None:
        """Register additional cameras via PnP and triangulate new structure."""
        if not self.registered_order:
            self.bootstrap_from_pair()
        remaining = set(range(len(self.image_paths))) - set(self.registered_order)
        passes = 0
        while remaining and passes < max_passes:
            added = []
            for idx in list(remaining):
                pts3d, pts2d, point_ids, kp_indices = self._collect_2d3d(idx)
                if len(pts3d) < self.min_pnp_correspondences:
                    continue
                pose_res = self._solve_pnp(pts3d, pts2d)
                if pose_res is None:
                    continue
                pose, inlier_idx = pose_res
                self.camera_poses[idx] = pose
                self.registered_order.append(idx)
                remaining.remove(idx)
                added.append(idx)
                self._register_observations(idx, point_ids, kp_indices, inlier_idx)
                new_pts = self._triangulate_with_registered(idx)
                self.stats["registered"].append((idx, len(pts3d), new_pts))
            if not added:
                break
            passes += 1

    def _collect_2d3d(
        self, idx: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
        pts3d: List[np.ndarray] = []
        pts2d: List[np.ndarray] = []
        point_ids: List[int] = []
        kp_indices: List[int] = []
        used: Dict[int, Tuple[float, int]] = {}
        for ref in self.registered_order:
            matches = self._match(ref, idx)
            for m in matches:
                pid = self.kp_to_point.get((ref, m.queryIdx))
                if pid is None:
                    continue
                prev = used.get(m.trainIdx, (np.inf, -1))
                if m.distance >= prev[0]:
                    continue
                used[m.trainIdx] = (m.distance, pid)
        for kp_idx, (_, pid) in used.items():
            pts3d.append(self.points3d[pid])
            pts2d.append(self._keypoints[idx][kp_idx].pt)
            point_ids.append(pid)
            kp_indices.append(kp_idx)
        return pts3d, pts2d, point_ids, kp_indices

    def _solve_pnp(
        self, points3d: List[np.ndarray], points2d: List[np.ndarray]
    ) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
        if len(points3d) < self.min_pnp_correspondences:
            return None
        pts3d = np.asarray(points3d, dtype=np.float32)
        pts2d = np.asarray(points2d, dtype=np.float32)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            self.intrinsic,
            None,
            iterationsCount=500,
            reprojectionError=self.reproj_error,
            confidence=0.999,
        )
        if not success:
            if len(points3d) >= 4:
                success, rvec, tvec = cv2.solvePnP(
                    pts3d,
                    pts2d,
                    self.intrinsic,
                    None,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if not success:
                    return None
                inliers = None
            else:
                return None
        if inliers is not None and inliers.shape[0] >= 6:
            rvec, tvec = cv2.solvePnPRefineLM(
                pts3d[inliers[:, 0]],
                pts2d[inliers[:, 0]],
                self.intrinsic,
                None,
                rvec,
                tvec,
            )
            inlier_idx = inliers[:, 0]
        else:
            inlier_idx = np.arange(len(points3d))
        R, _ = cv2.Rodrigues(rvec)
        return (R, tvec), inlier_idx

    def _register_observations(
        self, image_idx: int, point_ids: List[int], kp_indices: List[int], inlier_idx: np.ndarray
    ) -> None:
        if not point_ids:
            return
        for obs_idx in np.asarray(inlier_idx).ravel():
            if obs_idx >= len(point_ids):
                continue
            kp = kp_indices[obs_idx]
            pid = point_ids[obs_idx]
            if (image_idx, kp) in self.kp_to_point:
                continue
            self.kp_to_point[(image_idx, kp)] = pid

    def _triangulate_with_registered(self, idx: int) -> int:
        new_points = 0
        rgb_new = self._rgb_images[idx]
        kp_new = self._keypoints[idx]
        pose_new = self.camera_poses[idx]
        for ref in self.registered_order:
            if ref == idx:
                continue
            matches = self._match(ref, idx)
            fresh: List[cv2.DMatch] = []
            for m in matches:
                if (ref, m.queryIdx) in self.kp_to_point:
                    continue
                if (idx, m.trainIdx) in self.kp_to_point:
                    continue
                fresh.append(m)
            if len(fresh) < self.min_triangulation_matches:
                continue
            pts_ref, pts_new = _points_from_matches(self._keypoints[ref], kp_new, fresh)
            pts3d, mask = _triangulate(
                self.intrinsic,
                self.camera_poses[ref],
                pose_new,
                pts_ref,
                pts_new,
                reproj_thresh=self.reproj_error,
            )
            rgb_ref = self._rgb_images[ref]
            mask_idx = np.where(mask)[0]
            for out_idx, match_idx in enumerate(mask_idx):
                m = fresh[match_idx]
                pid = len(self.points3d)
                color = self._sample_color(rgb_ref, pts_ref[match_idx], rgb_new, pts_new[match_idx])
                self.points3d.append(pts3d[out_idx])
                self.point_colors.append(color)
                self.kp_to_point[(ref, m.queryIdx)] = pid
                self.kp_to_point[(idx, m.trainIdx)] = pid
                new_points += 1
        return new_points

    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.points3d:
            return np.empty((0, 3)), np.empty((0, 3))
        pts = np.asarray(self.points3d, dtype=np.float32)
        colors = np.asarray(self.point_colors, dtype=np.float32) / 255.0
        return pts, colors

    def save_point_cloud(self, path: Path) -> None:
        pts, colors = self.get_point_cloud()
        if pts.size == 0:
            raise RuntimeError("No 3D points available to export.")
        colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        header = "ply\nformat ascii 1.0\nelement vertex {}\n".format(len(pts))
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        lines = [
            "{:.6f} {:.6f} {:.6f} {} {} {}".format(x, y, z, r, g, b)
            for (x, y, z), (r, g, b) in zip(pts, colors_u8)
        ]
        Path(path).write_text(header + "\n".join(lines))
