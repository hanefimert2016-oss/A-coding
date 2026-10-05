from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import trimesh

def clean_creator_asset(src: str, dst: str):
    scene = trimesh.load(src, force='scene')
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError('No mesh geometry found in input asset')
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0].copy()
    else:
        mesh = scene.copy()

    before = {'vertices': int(len(mesh.vertices)), 'faces': int(len(mesh.faces))}
    mesh.remove_unreferenced_vertices()
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        pass
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()
    try:
        mesh.fix_normals()
    except Exception:
        pass

    # Keep meaningful components; tiny floating fragments are usually reconstruction noise.
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        max_area = max(float(p.area) for p in parts)
        parts = [p for p in parts if float(p.area) >= max_area * 0.002]
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)

    report = {
        'input': src, 'output': dst,
        'before': before,
        'after': {'vertices': int(len(mesh.vertices)), 'faces': int(len(mesh.faces))},
        'watertight': bool(mesh.is_watertight),
        'bounds': np.asarray(mesh.bounds).round(6).tolist(),
    }
    Path(str(dst) + '.json').write_text(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output')
    a = ap.parse_args()
    print(json.dumps(clean_creator_asset(a.input, a.output), indent=2))
