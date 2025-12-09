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

Phase 4 — Data Post-Processing & Coordinate Alignment
Overview
Parse camera extrinsics (R,t) from XML outputs (Agisoft/SfM)
Compute world-space camera centers using 
Align point cloud orientation to match standard graphics axes (Y-up vs Z-up)
Apply Statistical Outlier Removal (SOR) using Open3D to reduce floating noise
Generate a sequential view graph to define valid navigation paths
Export optimized assets (JSON poses, binary PLY cloud) for the web engine

Phase 5 — Interactive Web Visualization (Three.js)
Overview
Initialize WebGL rendering context, scene, and camera frustum
Async load the sparse point cloud and camera marker geometry
Visualize camera trajectory as a sequence of interactable nodes
Implement dual navigation logic: Free Orbit vs. Horizon-Stabilized view
Apply SLERP (Spherical Linear Interpolation) for smooth rotational transitions
Handle keyboard events for UI toggling, re-centering, and state switching

## Notebook 19.ipynb
The `19.ipynb` notebook walks through the entire SfM pipeline on the provided `converted_jpg` image set and produces the intermediate artifacts consumed by the viewer:

- **Image loading & preprocessing** – images are natural-sorted, converted to grayscale, sharpened, Laplacian-enhanced, and written to `converted_images_preprocessed/` for downstream steps while also being cached in-memory for visualization.
- **Feature extraction & pair scoring** – SIFT keypoints/descriptors are computed for every preprocessed image, then brute-force matches with a Lowe ratio test build `pair_scores`. A disparity heuristic picks the best stereo pair for two-view initialization.
- **Two-view reconstruction** – the selected pair (indices 22 & 23 by default) is matched, the essential matrix is estimated, `recoverPose` yields `R` and `t`, and triangulation followed by normalization creates `sfm_best_pair_pointcloud.ply` along with colorized scatter plots.
- **Metadata utilities** – helper routines extract focal lengths from EXIF (`get_focal_length_px_jpg`) and convert Metashape camera exports (`parse_metashape_xml`) into `cameras.json` for the viewer, plus HEIC➜JPG conversion via `sips`.
- **Point cloud helpers** – functions load OBJ/PLY assets with Open3D, prune them, and export lightweight variants used in the web viewer.
- **View-graph builder** – `build_viewgraph("cameras.json")` creates a simple sequential adjacency list consumed by the front-end for camera sequencing.

