import os, json, numpy as np
A = os.environ.get("JANUS_OUT"); T = f"{A}/_grid_tables"
FR = json.load(open(f"{T}/summary.json"))["frontier"]
def dump(npz, label, topn=60):
    z = np.load(npz, allow_pickle=True); names=list(z["names"]); C=z["corr"]; n=len(names)
    rows=[]
    for i in range(n):
        for j in range(i+1,n):
            if np.isfinite(C[i,j]): rows.append((names[i],names[j],round(float(C[i,j]),3)))
    rows.sort(key=lambda r:-abs(r[2]))
    print(f"\n##### {label} — {len(rows)} pairs, top {topn} #####")
    for a,b,v in rows[:topn]:
        print(f"{v:+.3f}  {a}[{FR.get(a,'?')}] ~ {b}[{FR.get(b,'?')}]")
    return rows
rh=dump(f"{T}/corr_head.npz","PER-HEAD")
rl=dump(f"{T}/corr_layer.npz","PER-LAYER")
json.dump({"head":rh,"layer":rl,"frontier":FR}, open(f"{T}/full_ranking.json","w"), indent=1)
print(f"\nwrote {T}/full_ranking.json")
