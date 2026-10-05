import json, math, resource
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
import trimesh

def make_single_view(n=256):
    y,x=np.indices((n,n),dtype=np.float32)
    X=(x-(n-1)/2)/(n*.34); Y=(y-(n-1)/2)/(n*.42)
    body=(X*X+Y*Y)<1
    eye1=((X+.32)/.22)**2+((Y+.12)/.07)**2<1
    eye2=((X-.32)/.22)**2+((Y+.12)/.07)**2<1
    mask=body
    relief=np.sqrt(np.clip(1-X*X-Y*Y,0,1))
    depth=np.zeros((n,n),np.float32)
    depth[mask]=1-relief[mask]
    rgb=np.zeros((n,n,3),np.uint8); rgb[:]=[8,10,14]
    rgb[mask]=[135,24,28]
    gold=mask&(Y<.25)&(np.abs(X)<.72)
    rgb[gold]=[200,142,45]
    rgb[eye1|eye2]=[190,240,255]
    return rgb,depth,(mask.astype(np.uint8)*255)

def build(rgb,depth,mask,out,grid=176):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    H=W=grid; D=144
    m=ndimage.zoom((mask>127).astype(np.float32),(H/mask.shape[0],W/mask.shape[1]),order=0)>0.5
    d=ndimage.zoom(depth.astype(np.float32),(H/depth.shape[0],W/depth.shape[1]),order=1)
    vals=d[m]; lo,hi=np.percentile(vals,[2,98]); d=np.clip((d-lo)/max(hi-lo,1e-6),0,1)
    dist=ndimage.distance_transform_edt(m).astype(np.float32); dist/=max(float(dist.max()),1e-6)
    roundness=np.power(dist,.75)
    zf=.22+(0.5-d)*.72+.48*roundness
    zb=-.18-.92*.52*roundness
    edge=np.clip(dist*5,0,1); zf=edge*zf+(1-edge)*(.08+.44*roundness)
    za=np.linspace(-1.15,1.15,D,dtype=np.float32)
    Z=za[None,None,:]
    vol=(m[:,:,None]&(Z>=zb[:,:,None])&(Z<=zf[:,:,None])).astype(np.float32)
    vol=ndimage.gaussian_filter(vol,sigma=1.0)
    v,f,_,_=measure.marching_cubes(vol,.5,spacing=(2/(H-1),2/(W-1),2.3/(D-1)))
    v[:,0]-=1; v[:,1]-=1; v[:,2]-=1.15; v=v[:,[1,0,2]]; v[:,1]*=-1
    mesh=trimesh.Trimesh(v,f,process=True)
    try: trimesh.smoothing.filter_taubin(mesh,lamb=.42,nu=.5,iterations=10)
    except Exception: pass
    mesh.fix_normals()
    vv=np.asarray(mesh.vertices); xmin,ymin=vv[:,:2].min(0); xmax,ymax=vv[:,:2].max(0)
    uv=np.stack([(vv[:,0]-xmin)/max(xmax-xmin,1e-8),1-(vv[:,1]-ymin)/max(ymax-ymin,1e-8)],1)
    back=vv[:,2]<0; uva=uv.copy(); uva[:,0]*=.5; uva[back,0]=.5+(1-uv[back,0])*.5
    im=Image.fromarray(rgb).resize((384,768),Image.Resampling.LANCZOS)
    atlas=Image.new('RGBA',(768,768),(128,128,128,255)); atlas.paste(im.convert('RGBA'),(0,0)); atlas.paste(im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).convert('RGBA'),(384,0))
    mat=trimesh.visual.material.PBRMaterial(name='CPUCreator_PBR',baseColorTexture=atlas,metallicFactor=.18,roughnessFactor=.48)
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uva,image=atlas,material=mat)
    mesh.export(out/'cpu_creator.glb'); atlas.save(out/'basecolor_atlas.png')
    rep={'version':'0.4.0-cpu','gpu_required':False,'vram_required_mb':0,'peak_process_rss_mb':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'watertight':bool(mesh.is_watertight)}
    (out/'report.json').write_text(json.dumps(rep,indent=2)); return rep

if __name__=='__main__':
    rgb,depth,mask=make_single_view()
    out=Path('v04_out'); out.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(out/'single_input.png'); Image.fromarray(mask).save(out/'single_mask.png'); np.save(out/'single_depth.npy',depth)
    print(json.dumps(build(rgb,depth,mask,out),indent=2))
