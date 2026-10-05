# ViewFusion3D v0.2

CPU-first adaptive multi-view reconstruction demo.

- Starts with 0/90/180/270 degree RGB-D views.
- Detects angular gaps.
- Requests only missing 45-degree midpoint views.
- Back-projects pixels to 3D surfels.
- Fuses them with voxel averaging.
- Exports PLY point/surfel data and GLB examples.
- GitHub Actions runs entirely on an Ubuntu CPU runner.

The procedural sci-fi helmet renderer is only a deterministic test backend. A future image-to-multiview + depth model can replace it without changing the CPU fusion loop.
