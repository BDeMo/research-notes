import numpy as np, glob, os
def sp(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<6: return np.nan
    x,y=x[m],y[m];rx=x.argsort().argsort().astype(float);ry=y.argsort().argsort().astype(float)
    rx-=rx.mean();ry-=ry.mean();d=np.linalg.norm(rx)*np.linalg.norm(ry)
    return (rx@ry)/d if d>0 else np.nan
def pcorr(x,y,z):  # partial corr x,y | z
    rxy,rxz,ryz=sp(x,y),sp(x,z),sp(y,z)
    den=np.sqrt(max(1-rxz**2,1e-9)*max(1-ryz**2,1e-9));return (rxy-rxz*ryz)/den if den>0 else np.nan
def wl(f2d,g2d,ctrl=None):  # within-layer (partial if ctrl)
    v=[]
    for l in range(f2d.shape[0]):
        if ctrl is None: r=sp(f2d[l],g2d[l])
        else: r=pcorr(f2d[l],g2d[l],ctrl[l])
        if np.isfinite(r): v.append(r)
    return float(np.mean(v)) if v else np.nan
order={"qwen3-0.6b":0,"qwen3-8b":1,"qwen3-14b":2,"glm4-9b":3}
files=sorted(glob.glob('/tmp/mnpz/mechanism_*.npz'),key=lambda f:order.get(os.path.basename(f)[10:-4],9))
print("Carrier disentangle: within-layer rho(grad, attn_distance), then partialling out each carrier")
print("%-12s %8s %12s %12s %12s" % ("model","raw","|out_norm","|attn_entropy","|sink"))
for f in files:
    z=np.load(f); g=z["grad_mag"]; ad=z["attn_distance"]
    raw=wl(ad,g); po=wl(ad,g,z["out_norm"]); pe=wl(ad,g,z["attn_entropy"]); ps=wl(ad,g,z["sink"])
    print("%-12s %+8.2f %+12.2f %+12.2f %+12.2f" % (os.path.basename(f)[10:-4],raw,po,pe,ps))
