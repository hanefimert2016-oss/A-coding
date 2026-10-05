from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, math
import numpy as np
from PIL import Image

@dataclass
class Camera:
    width:int; height:int; fx:float; fy:float; cx:float; cy:float
    world_from_camera:np.ndarray

@dataclass
class Config:
    seed_angles:tuple=(0.0,90.0,180.0,270.0)
    target_max_gap_deg:float=45.0
    max_rounds:int=3
    max_new_views_per_round:int=8
    size:int=192
    fov_deg:float=50.0
    radius:float=3.2
    stride:int=2
    voxel_size:float=0.025

def _norm(v):
    return v/max(np.linalg.norm(v),1e-8)

def camera(angle,size=192,radius=3.2,fov=50):
    a=math.radians(angle)
    pos=np.array([radius*math.sin(a),0.0,radius*math.cos(a)],float)
    fwd=_norm(-pos); up=np.array([0.,1.,0.])
    right=_norm(np.cross(up,fwd)); tup=_norm(np.cross(fwd,right))
    R=np.stack([right,-tup,fwd],axis=1)
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=pos
    fl=0.5*size/math.tan(math.radians(fov)/2)
    return Camera(size,size,fl,fl,(size-1)/2,(size-1)/2,T)

def _ray_ellipsoid(cam,center,radii):
    ys,xs=np.indices((cam.height,cam.width),dtype=float)
    dcam=np.stack([(xs-cam.cx)/cam.fx,(ys-cam.cy)/cam.fy,np.ones_like(xs)],axis=-1)
    dcam/=np.linalg.norm(dcam,axis=-1,keepdims=True)
    R=cam.world_from_camera[:3,:3]; o=cam.world_from_camera[:3,3]
    d=dcam@R.T; c=np.asarray(center,float); r=np.asarray(radii,float)
    oo=(o-c)/r; dd=d/r
    A=np.sum(dd*dd,axis=-1); B=2*np.sum(dd*oo[None,None,:],axis=-1); C=float(oo@oo-1)
    disc=B*B-4*A*C; ok=disc>=0; sq=np.zeros_like(disc); sq[ok]=np.sqrt(disc[ok])
    t0=(-B-sq)/(2*A); t1=(-B+sq)/(2*A)
    t=np.where(t0>1e-6,t0,t1); ok&=t>1e-6
    return np.where(ok,t,np.inf),d,o

def render_helmet(cam):
    shapes=[
      ((0,.08,0),(.88,1.08,.78),(124,20,25)),
      ((0,-.62,.10),(.72,.48,.64),(105,15,20)),
      ((0,.08,.67),(.68,.78,.18),(202,147,54)),
      ((-.89,.03,0),(.15,.25,.18),(182,45,40)),
      ((.89,.03,0),(.15,.25,.18),(182,45,40)),
      ((-.28,.18,.83),(.18,.055,.055),(185,240,255)),
      ((.28,.18,.83),(.18,.055,.055),(185,240,255)),
      ((0,-.22,.84),(.16,.13,.05),(70,120,150)),
    ]
    h,w=cam.height,cam.width
    best=np.full((h,w),np.inf); ids=np.full((h,w),-1,np.int16); d=o=None
    for i,(c,r,col) in enumerate(shapes):
        t,dd,oo=_ray_ellipsoid(cam,c,r)
        if d is None: d,o=dd,oo
        m=t<best; best[m]=t[m]; ids[m]=i
    hit=np.isfinite(best); world=np.zeros((h,w,3),float)
    world[hit]=o[None,:]+d[hit]*best[hit,None]
    R=cam.world_from_camera[:3,:3]
    depth=np.zeros((h,w),np.float32)
    depth[hit]=((world[hit]-o[None,:])@R)[:,2]
    rgb=np.zeros((h,w,3),np.uint8); rgb[:]=[8,10,14]
    ld=np.array([.25,.45,1.]); ld/=np.linalg.norm(ld)
    for i,(c,r,col) in enumerate(shapes):
        m=hit&(ids==i)
        if not np.any(m): continue
        rr=np.asarray(r,float); n=(world[m]-np.asarray(c,float))/(rr*rr)
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        shade=.52+.48*np.clip(n@ld,0,1)
        rgb[m]=np.clip(np.asarray(col)[None,:]*shade[:,None],0,255).astype(np.uint8)
    return rgb,depth

def unproject(rgb,depth,cam,stride=2):
    ys=np.arange(0,cam.height,stride); xs=np.arange(0,cam.width,stride)
    u,v=np.meshgrid(xs,ys); z=depth[v,u].astype(float); ok=np.isfinite(z)&(z>1e-6)&(z<20)
    uu=u[ok].astype(float); vv=v[ok].astype(float); z=z[ok]
    xyz=np.stack([(uu-cam.cx)/cam.fx*z,(vv-cam.cy)/cam.fy*z,z],axis=1)
    R=cam.world_from_camera[:3,:3]; t=cam.world_from_camera[:3,3]
    return (xyz@R.T+t).astype(np.float32),rgb[v[ok],u[ok]].astype(np.uint8)

def voxel_fuse(points,colors,size=.025):
    vox=np.floor(points/size).astype(np.int64)
    _,inv=np.unique(vox,axis=0,return_inverse=True); n=int(inv.max())+1
    cnt=np.bincount(inv,minlength=n).astype(float)
    p=np.empty((n,3)); c=np.empty((n,3))
    for k in range(3):
        p[:,k]=np.bincount(inv,weights=points[:,k],minlength=n)/cnt
        c[:,k]=np.bincount(inv,weights=colors[:,k],minlength=n)/cnt
    return p.astype(np.float32),np.clip(np.rint(c),0,255).astype(np.uint8)

def suggest(angles,max_gap=45,max_new=8):
    a=sorted({float(x)%360 for x in angles})
    gaps=[]
    for i,cur in enumerate(a):
        nxt=a[(i+1)%len(a)]; gap=(nxt-cur)%360
        if gap>max_gap: gaps.append((gap,(cur+gap/2)%360))
    gaps.sort(reverse=True)
    return [round(x[1],3) for x in gaps[:max_new]]

def coverage(angles):
    a=sorted({float(x)%360 for x in angles})
    if len(a)<2:return 0.0
    gaps=[(a[(i+1)%len(a)]-x)%360 for i,x in enumerate(a)]
    return float(np.clip(1-max(0,max(gaps)-45)/315,0,1))

def save_ply(path,p,c):
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    a=np.empty(len(p),dtype=dtype)
    a['x'],a['y'],a['z']=p[:,0],p[:,1],p[:,2]; a['r'],a['g'],a['b']=c[:,0],c[:,1],c[:,2]
    hdr=f"ply\nformat binary_little_endian 1.0\nelement vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(path,'wb') as f:f.write(hdr.encode());a.tofile(f)

def export_glb(out,p,c):
    import trimesh
    pc=trimesh.points.PointCloud(p,colors=c); pc.export(out/'helmet_points.glb')
    mesh=pc.convex_hull
    vc=np.zeros((len(mesh.vertices),4),np.uint8)
    for i,v in enumerate(mesh.vertices):
        j=int(np.argmin(np.sum((p-v[None,:])**2,axis=1))); vc[i,:3]=c[j]; vc[i,3]=255
    mesh.visual.vertex_colors=vc; mesh.export(out/'helmet_mesh.glb')

def run(outdir,cfg):
    out=Path(outdir); vd=out/'views'; vd.mkdir(parents=True,exist_ok=True)
    cache={}; hist=[]
    def ensure(a):
        a=float(a)%360
        if a in cache:return
        cam=camera(a,cfg.size,cfg.radius,cfg.fov_deg); rgb,depth=render_helmet(cam); cache[a]=(rgb,depth,cam)
        Image.fromarray(rgb).save(vd/f"view_{int(round(a))%360:03d}.png")
        np.save(vd/f"view_{int(round(a))%360:03d}_depth.npy",depth)
    for a in cfg.seed_angles:ensure(a)
    for r in range(cfg.max_rounds+1):
        miss=suggest(cache.keys(),cfg.target_max_gap_deg,cfg.max_new_views_per_round)
        hist.append({'round':r,'angles':sorted(cache),'coverage_score':coverage(cache),'requested_angles':miss})
        if not miss or r>=cfg.max_rounds:break
        for a in miss:ensure(a)
    pts=[]; cols=[]
    for a in sorted(cache):
        p,c=unproject(*cache[a][:2],cache[a][2],cfg.stride); pts.append(p); cols.append(c)
    rawp=np.concatenate(pts); rawc=np.concatenate(cols); p,c=voxel_fuse(rawp,rawc,cfg.voxel_size)
    save_ply(out/'helmet_surfel.ply',p,c); export_glb(out,p,c)
    rep={'version':'0.2.0','cpu_only_demo':True,'config':asdict(cfg),'history':hist,'final_angles':sorted(cache),'final_coverage_score':coverage(cache),'raw_points':len(rawp),'fused_points':len(p),'outputs':{'surfel_ply':'helmet_surfel.ply','point_glb':'helmet_points.glb','solid_preview_glb':'helmet_mesh.glb'}}
    (out/'report.json').write_text(json.dumps(rep,indent=2)); return rep
