# ViewFusion3D v0.3 — Creator Hybrid

Goal: output a real creator-style asset, not a point cloud.

## Modes

- Eco: Stable Fast 3D (SF3D), ~6 GB VRAM for one image, UV + texture/material output.
- Ultra: TRELLIS.2 on a larger GPU for higher-end PBR/topology.

## Hybrid pipeline

1. Input image / background removal
2. Creator backbone (SF3D or TRELLIS.2) -> base mesh
3. Render confidence views from the base mesh
4. ViewFusion gap/confidence detector selects only weak angles
5. Optional image-to-novel-view generation, one view at a time
6. Depth/normal consistency refinement
7. UV texture rebake
8. Final GLB validation + cleanup

The CPU part runs in GitHub Actions. The creator backbone is GPU work.

## Kaggle

Use `kaggle_sf3d.ipynb` on a T4/P100-class GPU. It installs the official
Stable Fast 3D repo, generates a GLB, then runs our CPU postprocess stage.
