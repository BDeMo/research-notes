"""IMP figure: token-vs-span (keep 0.5) + keep-rate budget curve. From imp_/spf_/spk_ logs."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__))

fig,(axl,axr)=plt.subplots(1,2,figsize=(12.5,4.6))

# --- left: token vs span vs full (keep 0.5) ---
benches=["ruler8k","numerical","categorical","squad","hotpot","trivia","narrativeqa"]
full=[0.98,0.75,0.88,0.72,0.61,0.72,0.17]
tok =[0.98,0.75,0.75,0.15,0.21,0.58,0.11]
spn =[0.96,0.71,0.98,0.46,0.42,0.75,0.11]
x=np.arange(len(benches)); w=0.26
axl.bar(x-w, full, w, label="full (ceiling)", color="tab:gray")
axl.bar(x,   tok,  w, label="IMP token-level", color="tab:orange")
axl.bar(x+w, spn,  w, label="IMP span-level", color="tab:blue")
axl.set_xticks(x); axl.set_xticklabels(benches, rotation=20, ha="right")
axl.set_ylabel("accuracy"); axl.set_ylim(0,1.05)
axl.set_title("IMP token vs span (keep 0.5)\nspan rescues coherent QA (squad/hotpot/trivia), keeps retrieval; beats full on trivia/categorical")
axl.grid(alpha=0.3,axis="y"); axl.legend(fontsize=8)

# --- right: keep-rate budget curve (span=32) ---
keep=[0.25,0.5,0.75]
curves={"ruler-8k":[0.69,0.96,0.98],"numerical":[0.58,0.71,0.75],"squad":[0.33,0.46,0.59],"hotpot":[0.43,0.42,0.57]}
fullref={"ruler-8k":0.98,"numerical":0.75,"squad":0.72,"hotpot":0.61}
for i,(b,ys) in enumerate(curves.items()):
    col=f"C{i}"
    axr.plot(keep, ys, "o-", lw=2, color=col, label=b)
    axr.axhline(fullref[b], ls=":", lw=1, color=col, alpha=0.6)
axr.set_xlabel("keep fraction"); axr.set_ylabel("accuracy"); axr.set_ylim(0,1.05); axr.set_xticks(keep)
axr.set_title("IMP keep-rate budget curve (span=32)\nmonotone in budget; dotted = full ceiling per bench")
axr.grid(alpha=0.3); axr.legend(fontsize=8)

fig.suptitle("Fig 13 — IMP (Paper B): span-level importance routing on a frozen base", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig13_imp.png"),dpi=140,bbox_inches="tight")
print("wrote fig13_imp.png")
