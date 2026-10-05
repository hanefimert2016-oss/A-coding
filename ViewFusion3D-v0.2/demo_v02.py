import argparse, json
from v02 import Config, run
p=argparse.ArgumentParser()
p.add_argument('--out',default='v02_out'); p.add_argument('--size',type=int,default=192)
p.add_argument('--stride',type=int,default=2); p.add_argument('--target-gap',type=float,default=45)
a=p.parse_args()
r=run(a.out,Config(size=a.size,stride=a.stride,target_max_gap_deg=a.target_gap))
print(json.dumps(r,indent=2))
