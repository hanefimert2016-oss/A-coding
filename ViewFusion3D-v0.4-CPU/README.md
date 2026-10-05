# ViewFusion3D v0.4 CPU Creator

CPU-first single-image 3D reconstruction target:
- 0 MB VRAM required for geometry
- roughly 200-300 MB process RAM in the demo
- closed watertight mesh
- UV/PBR GLB
- one-image input
- hidden geometry completed with symmetry/thickness priors

Run:
```bash
pip install -r requirements.txt
python -m viewfusion3d.demo_v04 --out v04_out
```

For real files use RGB + relative depth NPY + object mask PNG with cpu_from_files.py.
