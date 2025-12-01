# CV_PROJECT

Incremental Structure-from-Motion Pipeline (Phases 1–3)
This project implements the foundational stages of a classical Structure-from-Motion (SfM) system:
feature extraction,
two-view reconstruction, and
incremental multi-view mapping.

# Phase 1 — Preprocessing & Feature Extraction
Overview
Convert images to grayscale
Apply Gaussian smoothing
Extract SIFT keypoints + 128-dim descriptors
Store feature sets for all images


# Phase 2 — Two-View Geometry & Initial Reconstruction
Overview
Match descriptors between the first two images using BFMatcher + Lowe ratio test
Estimate Essential Matrix with RANSAC
Recover relative camera pose (R, t)
Build projection matrices:
P₁ = K [I | 0]
P₂ = K [R | t]
Triangulate inlier correspondences
Filter points using depth + reprojection error


# Phase 3 — Incremental Multi-View Pose Estimation & Mapping
Overview
For each new image:
Match features to the previous frame using SIFT + Lowe ratio
Associate 2D features with existing 3D points via reprojection
Solve PnP (RANSAC) to estimate the new camera pose
Triangulate new points using the new pose and a previous view
(Optional) Run a local bundle adjustment on the latest pose + points

