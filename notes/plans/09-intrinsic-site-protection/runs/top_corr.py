import os, json, numpy as np
A = os.environ.get("JANUS_OUT", "/home/devuser/janus/artifacts")
T = f"{A}/_grid_tables"
FR = json.load(open(f"{T}/summary.json"))["frontier"]
TAUT = {tuple(sorted(t)) for t in [
    ("eff_rank","repr_entropy"),("w_eff_rank","w_stable_rank"),("grad_mag","fisher"),
    ("ll_kl_to_final","ll_top1_depth"),("tuned_lens_kl","tuned_lens_depth"),
    ("ll_kl_to_final","tuned_lens_kl"),("ll_top1_depth","tuned_lens_depth"),
    ("massive_count","massive_max"),("ll_kl_to_final","tuned_lens_depth"),
    ("ll_top1_depth","tuned_lens_kl"),("hd_grad_mag","hd_fisher"),
    ("w_eff_rank","w_ht_alpha"),("eff_rank","intrinsic_dim")]}
def top(npz, label, k=18):
    z = np.load(npz, allow_pickle=True); names=list(z["names"]); C=z["corr"]; n=len(names)
    rows=[]
    for i in range(n):
        for j in range(i+1,n):
            if not np.isfinite(C[i,j]): continue
            if tuple(sorted([names[i],names[j]])) in TAUT: continue
            rows.append((names[i],names[j],float(C[i,j])))
    rows.sort(key=lambda r:-abs(r[2]))
    print(f"\n===== TOP |rho| {label} =====")
    for a,b,v in rows[:k]:
        print(f"  {v:+.3f}  {a:16s}[{FR.get(a,'?')}] ~ {b:16s}[{FR.get(b,'?')}]")
top(f"{T}/corr_head.npz","PER-HEAD")
top(f"{T}/corr_layer.npz","PER-LAYER")
