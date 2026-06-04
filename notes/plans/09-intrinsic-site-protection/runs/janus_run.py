#!/usr/bin/env python3
"""Janus (Plan 09) overnight exploration runner.

Single-file CLI with subcommands. Direction: long-context <-> forgetting
coupling at intrinsic transformer sites.

Subcommands:
  detect   : site detectors (sink / retrieval / induction heads, massive-act channels)
  niah     : NIAH accuracy over depth x length grid (optionally ablate retrieval heads)
  sft      : fine-tune (full-FT or LoRA), optionally PROTECT a head set via grad masking;
             snapshots base proj weights and saves per-head drift after training
  capeval  : capability retention (MMLU, TriviaQA) + new-domain (MATH) accuracy

All artifacts -> $JANUS_OUT (default /home/devuser/janus/artifacts) as JSON/NPZ.
Everything wrapped so a single failure never kills the overnight batch.
"""
import argparse, json, os, sys, time, math, random, gc
import numpy as np
import torch
import torch.nn.functional as F

ART = os.environ.get("JANUS_OUT", "/home/devuser/janus/artifacts")
os.makedirs(ART, exist_ok=True)
HFHOME = os.environ.get("HF_HOME", "/home/devuser/.cache/huggingface")
os.environ.setdefault("HF_HOME", HFHOME)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    # cross-family (different vendor, standard softmax-attn arch Glm4ForCausalLM)
    "glm4-9b": "THUDM/GLM-4-9B-0414",
    "glm4-32b": "zai-org/GLM-4-32B-0414",
    # Qwen3.5 (multimodal, hybrid linear attention — needs newer transformers;
    # attention-head metrics only valid on its full-attention layers)
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}
SIZE_B = {"qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7, "qwen3-4b": 4.0, "qwen3-8b": 8.0,
          "qwen3-14b": 14.0, "qwen2.5-1.5b": 1.5, "qwen2.5-1.5b-instruct": 1.5,
          "qwen2.5-7b-instruct": 7.0, "glm4-9b": 9.0, "glm4-32b": 32.0,
          "qwen3.5-4b": 4.0, "qwen3.5-9b": 9.0}

def log(*a):
    print(f"[janus {time.strftime('%H:%M:%S')}]", *a, flush=True)

def tag_path(tag, name):
    d = os.path.join(ART, tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

# ---------------------------------------------------------------- model utils
def native_dtype(model_id):
    # honor the checkpoint's native precision: fp32 if released as fp32, else bf16 floor.
    # NEVER quantize and never go below bf16 fidelity.
    from transformers import AutoConfig
    try:
        c = AutoConfig.from_pretrained(model_id)
        c = getattr(c, "text_config", c)
        nd = getattr(c, "torch_dtype", None) or getattr(c, "dtype", None)
        nd = str(nd)
        if "float32" in nd: return torch.float32
    except Exception:
        pass
    return torch.bfloat16  # bf16 floor (fp16 checkpoints also load at bf16, never int8/int4)

def load_model(model_id, eager=False, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if dtype is None:
        dtype = native_dtype(model_id)
    kw = dict(dtype=dtype)
    if eager:
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).cuda().eval()
    log(f"loaded {model_id} dtype={dtype}")
    return model, tok

def head_dim_of(cfg):
    return getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

def layers_of(model):
    for path in [lambda m: m.model.layers,
                 lambda m: m.model.language_model.layers,
                 lambda m: m.language_model.model.layers,
                 lambda m: m.model.model.layers]:
        try:
            ls = path(model)
            if ls is not None and len(ls) > 0:
                return ls
        except Exception:
            continue
    return model.model.layers

# ---------------------------------------------------------------- generic text
GENERIC = [
    "The history of cartography is closely tied to the development of mathematics and astronomy. Early maps were often based on travelers' accounts and were highly inaccurate by modern standards, yet they shaped how empires imagined their reach.",
    "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. It occurs in the chloroplasts and releases oxygen as a byproduct, sustaining nearly all aerobic life on the planet.",
    "Economic inflation refers to a sustained increase in the general price level of goods and services. Central banks attempt to manage inflation through monetary policy, including adjusting short-term interest rates and open-market operations.",
    "The Roman aqueducts were a remarkable feat of ancient engineering, carrying water across vast distances using a slight downhill gradient and a system of bridges, tunnels, and siphons that supplied public baths and fountains.",
    "In organic chemistry, a functional group is a specific substituent within molecules responsible for the characteristic chemical reactions of those molecules. The same functional group behaves similarly across different compounds.",
    "Plate tectonics is the scientific theory describing the large-scale motion of the lithosphere. The model builds on the concept of continental drift and explains earthquakes, volcanism, and mountain building at plate boundaries.",
]
FILLER = ("The garden was quiet in the early morning, and a light breeze moved through the leaves. "
          "People walking by paused to look at the flowers before continuing on their way. ")

def filler_to_len(tok, n_tokens):
    s = ""
    while len(tok(s, add_special_tokens=False).input_ids) < n_tokens:
        s += FILLER
    ids = tok(s, add_special_tokens=False).input_ids[:n_tokens]
    return tok.decode(ids)

# ================================================================ DETECT
@torch.no_grad()
def cmd_detect(args):
    model_id = MODELS.get(args.model, args.model)
    log(f"DETECT {args.model} ({model_id})")
    model, tok = load_model(model_id, eager=True)
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    sink = np.zeros((L, H)); n = 0
    massive = np.zeros((L, cfg.hidden_size))
    for text in GENERIC:
        ids = tok(FILLER + text, return_tensors="pt", truncation=True, max_length=480).input_ids.cuda()
        out = model(ids, output_attentions=True, output_hidden_states=True)
        for l in range(L):
            if out.attentions is None or l >= len(out.attentions) or out.attentions[l] is None:
                continue
            att = out.attentions[l][0]
            sink[l] += att[:, 4:, 0].mean(dim=1).float().cpu().numpy()
        for l in range(L):
            hs = out.hidden_states[l + 1][0].abs().float().cpu().numpy()
            massive[l] = np.maximum(massive[l], hs.max(axis=0))
        n += 1
    sink /= n

    # retrieval heads: attention from the final query to needle value tokens,
    # averaged over several needle variants (per [ret-dyn]: heads are dynamic)
    retr = np.zeros((L, H)); rc = 0
    needles = [
        ("the secret access code for project Helios is 47192", " 47192"),
        ("the magic number assigned to Dr. Lin is 80356", " 80356"),
        ("the vault password for the north office is 13947", " 13947"),
    ]
    for fact, val in needles:
        pre = "Read the passage and answer the question.\n" + filler_to_len(tok, 120)
        mid = f"\nNote: {fact}.\n" + filler_to_len(tok, 120)
        post = "\nQuestion: recall the number mentioned above. Answer:"
        ids = tok(pre + mid + post, return_tensors="pt", truncation=True, max_length=512).input_ids
        seq = ids[0].tolist()
        vt = tok(val, add_special_tokens=False).input_ids
        pos = []
        for i in range(len(seq) - len(vt) + 1):
            if seq[i:i + len(vt)] == vt:
                pos = list(range(i, i + len(vt))); break
        if not pos:
            continue
        ids = ids.cuda()
        out = model(ids, output_attentions=True)
        last = ids.shape[1] - 1
        for l in range(L):
            retr[l] += out.attentions[l][0][:, last, pos].sum(dim=1).float().cpu().numpy()
        rc += 1
    if rc:
        retr /= rc

    # induction heads: repeated random token sequence, prefix-matching score
    induction = np.zeros((L, H))
    vocab = tok.vocab_size
    rng = random.Random(0)
    seq_a = [rng.randint(1000, min(vocab, 30000) - 1) for _ in range(60)]
    rep = seq_a + seq_a
    ids = torch.tensor([rep]).cuda()
    out = model(ids, output_attentions=True)
    T = len(rep); half = len(seq_a)
    for l in range(L):
        att = out.attentions[l][0]  # [H,T,T]
        # for query position t in second copy, attend to t-half+1 (the induction offset)
        sc = np.zeros(H)
        for t in range(half + 1, T):
            tgt = t - half
            sc += att[:, t, tgt].float().cpu().numpy()
        induction[l] = sc / (T - half - 1)

    chan_max = massive.max(axis=0)
    med = float(np.median(chan_max))
    outliers = sorted([(int(c), float(chan_max[c])) for c in np.where(chan_max >= 6 * med)[0]],
                      key=lambda x: -x[1])

    def topk(mat, k=20):
        idx = np.dstack(np.unravel_index(np.argsort(mat.ravel())[::-1][:k], mat.shape))[0]
        return [[int(l), int(h), round(float(mat[l, h]), 4)] for l, h in idx]

    np.savez(tag_path(args.model, "detect.npz"), sink=sink, retr=retr, induction=induction,
             chan_max=chan_max)
    res = {
        "model": args.model, "model_id": model_id, "layers": L, "heads": H,
        "hidden": cfg.hidden_size,
        "gate": {
            "bos_sink_present": bool((sink > 0.3).any()),
            "n_sink_heads_gt0.5": int((sink > 0.5).sum()),
            "n_massive_channels": len(outliers),
            "max_sink": round(float(sink.max()), 4),
            "max_retr": round(float(retr.max()), 4),
            "max_induction": round(float(induction.max()), 4),
        },
        "top_sink": topk(sink), "top_retr": topk(retr), "top_induction": topk(induction),
        "massive_channels": outliers[:30], "median_channel": round(med, 3),
        "sink_vs_retr_spearman": round(spearman(sink.ravel(), retr.ravel()), 4),
        "sink_vs_retr_jaccard_top15": jaccard_topk(sink, retr, 15),
    }
    json.dump(res, open(tag_path(args.model, "detect.json"), "w"), indent=2)
    log("detect gate:", res["gate"])
    free(model)

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra @ rb) / d) if d > 0 else 0.0

def jaccard_topk(a, b, k):
    fa = set(map(tuple, np.dstack(np.unravel_index(np.argsort(a.ravel())[::-1][:k], a.shape))[0].tolist()))
    fb = set(map(tuple, np.dstack(np.unravel_index(np.argsort(b.ravel())[::-1][:k], b.shape))[0].tolist()))
    u = fa | fb
    return round(len(fa & fb) / len(u), 4) if u else 0.0

# ================================================================ NIAH
@torch.no_grad()
def cmd_niah(args):
    model_id = MODELS.get(args.model, args.model)
    log(f"NIAH {args.model} ablate={args.ablate} k={args.ablate_k} ckpt={args.ckpt}")
    if args.ckpt and os.path.isdir(args.ckpt) and os.path.exists(os.path.join(args.ckpt, "config.json")):
        model, tok = load_model(args.ckpt, eager=False)
    else:
        model, tok = load_model(model_id, eager=False)
        if args.ckpt and os.path.isdir(args.ckpt):
            model = load_into(model, tok, args.ckpt)
    handles = []
    if args.ablate:
        det = os.path.join(ART, args.model, "detect.npz")
        if os.path.exists(det):
            mat = np.load(det)[args.ablate_site]
            heads = [tuple(x) for x in np.dstack(np.unravel_index(
                np.argsort(mat.ravel())[::-1][:args.ablate_k], mat.shape))[0].tolist()]
            handles = install_head_ablation(model, heads)
            log(f"ablating {len(heads)} {args.ablate_site} heads")
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = {}
    for Ln in lengths:
        for d in depths:
            ok = 0; tot = 0
            for s in range(args.samples):
                code = f"{random.randint(10000,99999)}"
                key = f"K{s}"
                needle = f" The special {key} value is {code}. "
                body = filler_to_len(tok, Ln)
                ins = int(len(body) * d)
                ctx = body[:ins] + needle + body[ins:]
                prompt = ("Read the text and answer.\n" + ctx +
                          f"\nQuestion: What is the special {key} value? Answer with the number only:")
                ids = tok(prompt, return_tensors="pt", truncation=True, max_length=Ln + 200).input_ids.cuda()
                g = model.generate(ids, max_new_tokens=10, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
                ans = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
                tot += 1; ok += int(code in ans)
            grid[f"{Ln}|{d}"] = round(ok / max(tot, 1), 3)
    for h in handles:
        h.remove()
    suffix = f"_ablate_{args.ablate_site}" if args.ablate else ("_" + os.path.basename(args.ckpt.rstrip("/")) if args.ckpt else "_base")
    out = {"model": args.model, "lengths": lengths, "depths": depths, "grid": grid,
           "ablate": bool(args.ablate), "ablate_k": args.ablate_k}
    json.dump(out, open(tag_path(args.model, f"niah{suffix}.json"), "w"), indent=2)
    log("niah mean:", round(float(np.mean(list(grid.values()))), 3))
    free(model)

def install_head_ablation(model, heads):
    """Zero the o_proj contribution of given (layer,head) at runtime."""
    cfg = model.config; hd = head_dim_of(cfg)
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    handles = []
    for l, hs in by_layer.items():
        attn = layers_of(model)[l].self_attn
        def mk(hs):
            def hook(mod, inp):
                x = inp[0].clone()
                for h in hs:
                    x[..., h * hd:(h + 1) * hd] = 0
                return (x,) + inp[1:]
            return hook
        handles.append(attn.o_proj.register_forward_pre_hook(mk(hs)))
    return handles

# ================================================================ SFT
def build_sft_data(tok, domain, n, max_len):
    from datasets import load_dataset
    rows = []
    try:
        if domain == "math":
            for cfg_name in ["algebra", "prealgebra", "number_theory"]:
                try:
                    ds = load_dataset("EleutherAI/hendrycks_math", cfg_name, split="train")
                    for r in ds:
                        rows.append((f"Problem: {r['problem']}\nSolution:", " " + r["solution"]))
                    if len(rows) >= n: break
                except Exception as e:
                    log("math cfg fail", cfg_name, e)
        elif domain == "trivia":
            ds = load_trivia("train")
            for r in ds:
                a = r["answer"]["value"]
                rows.append((f"Question: {r['question']}\nAnswer:", " " + a))
                if len(rows) >= n: break
    except Exception as e:
        log("dataset load failed", domain, e)
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    log(f"sft data {domain}: {len(rows)} rows")
    feats = []
    for p, c in rows:
        pi = tok(p, add_special_tokens=False).input_ids
        ci = tok(c, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (pi + ci)[:max_len]
        labels = ([-100] * len(pi) + ci)[:max_len]
        feats.append((ids, labels))
    return feats

def protected_head_indices(model_name, kind, k):
    det = os.path.join(ART, model_name, "detect.npz")
    if kind == "random" or not os.path.exists(det):
        cfg = None
    if kind == "none":
        return []
    if kind in ("retrieval", "sink", "deltaw"):
        if not os.path.exists(det):
            return []
        z = np.load(det)
        if kind == "deltaw":
            # defined later from a prior vanilla run's drift; fall back to retrieval
            dpath = os.path.join(ART, model_name, "drift_math_full_none.npz")
            mat = np.load(dpath)["head_drift"] if os.path.exists(dpath) else z["retr"]
        else:
            mat = z["retr"] if kind == "retrieval" else z["sink"]
        idx = np.dstack(np.unravel_index(np.argsort(mat.ravel())[::-1][:k], mat.shape))[0]
        return [tuple(map(int, x)) for x in idx]
    return []

def cmd_sft(args):
    model_id = MODELS.get(args.model, args.model)
    tag = args.run_tag or f"{args.domain}_{args.mode}_{args.protect}"
    log(f"SFT {args.model} {tag} steps={args.steps} lr={args.lr}")
    model, tok = load_model(model_id, eager=False)
    # snapshot base proj weights for drift
    base_snap = snapshot_proj(model)
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()
        if args.model in ("qwen3-8b", "qwen3-14b"):
            model.gradient_checkpointing_enable()
    model.train()
    feats = build_sft_data(tok, args.domain, args.steps * args.bs + 64, args.maxlen)
    if not feats:
        log("NO DATA, abort sft"); free(model); return
    prot = protected_head_indices(args.model, args.protect, args.protect_k)
    log(f"protect={args.protect} -> {len(prot)} heads")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    hd = head_dim_of(model.config)
    losses = []
    step = 0
    while step < args.steps:
        batch = feats[(step * args.bs) % (len(feats) - args.bs): (step * args.bs) % (len(feats) - args.bs) + args.bs]
        ids, labels = collate(batch, tok.pad_token_id)
        ids = ids.cuda(); labels = labels.cuda()
        out = model(ids, labels=labels)
        loss = out.loss
        opt.zero_grad(); loss.backward()
        if prot and args.mode == "full":
            mask_head_grads(model, prot, hd)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(loss))
        if step % 25 == 0:
            log(f"  step {step}/{args.steps} loss {loss:.4f}")
        step += 1
    model.eval()
    # save tuned ckpt
    ckpt = tag_path(args.model, f"ckpt_{tag}")
    if args.mode == "lora":
        model.save_pretrained(ckpt)
    else:
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    # compute + save drift (full-FT: weight diff; lora: merged diff)
    try:
        compute_drift(model, base_snap, args.model, tag, args.mode)
    except Exception as e:
        log("drift failed", e)
    json.dump({"tag": tag, "losses": losses, "final_loss": losses[-1],
               "protect": args.protect, "n_protected": len(prot)},
              open(tag_path(args.model, f"sftlog_{tag}.json"), "w"), indent=2)
    log(f"SFT done {tag}, ckpt={ckpt}")
    free(model)

def snapshot_proj(model):
    snap = {}
    for i, lyr in enumerate(layers_of(model)):
        a = getattr(lyr, "self_attn", None)
        if a is None or not hasattr(a, "q_proj"): continue
        for nm in ["q_proj", "o_proj"]:
            snap[(i, nm)] = getattr(a, nm).weight.detach().float().cpu().clone()
    return snap

def mask_head_grads(model, prot, hd):
    by = {}
    for l, h in prot:
        by.setdefault(l, []).append(h)
    for l, hs in by.items():
        a = layers_of(model)[l].self_attn
        if a.q_proj.weight.grad is not None:
            for h in hs:
                a.q_proj.weight.grad[h * hd:(h + 1) * hd, :] = 0
        if a.o_proj.weight.grad is not None:
            for h in hs:
                a.o_proj.weight.grad[:, h * hd:(h + 1) * hd] = 0

@torch.no_grad()
def compute_drift(model, base_snap, model_name, tag, mode):
    m = model.merge_and_unload() if mode == "lora" else model
    cfg = m.config; hd = head_dim_of(cfg); H = cfg.num_attention_heads
    L = cfg.num_hidden_layers
    head_drift = np.zeros((L, H))
    for i, lyr in enumerate(layers_of(m)):
        a = lyr.self_attn
        for nm, axis in [("q_proj", 0), ("o_proj", 1)]:
            w = getattr(a, nm).weight.detach().float().cpu()
            dw = (w - base_snap[(i, nm)]).numpy()
            if nm == "q_proj":
                dw = dw.reshape(H, hd, -1)
                head_drift[i] += np.linalg.norm(dw.reshape(H, -1), axis=1)
            else:
                dw = dw.reshape(-1, H, hd)
                head_drift[i] += np.linalg.norm(dw.transpose(1, 0, 2).reshape(H, -1), axis=1)
    np.savez(os.path.join(ART, model_name, f"drift_{tag}.npz"), head_drift=head_drift)
    det = os.path.join(ART, model_name, "detect.npz")
    if os.path.exists(det):
        retr = np.load(det)["retr"]; sink = np.load(det)["sink"]
        rho_r = spearman(retr.ravel(), head_drift.ravel())
        rho_s = spearman(sink.ravel(), head_drift.ravel())
        json.dump({"tag": tag, "spearman_retr_vs_drift": round(rho_r, 4),
                   "spearman_sink_vs_drift": round(rho_s, 4)},
                  open(os.path.join(ART, model_name, f"coupling_{tag}.json"), "w"), indent=2)
        log(f"COUPLING {tag}: rho(retr,drift)={rho_r:.3f}  rho(sink,drift)={rho_s:.3f}")

# ================================================================ CAPEVAL
def load_into(model, tok, ckpt):
    if os.path.exists(os.path.join(ckpt, "adapter_config.json")):
        from peft import PeftModel
        m = PeftModel.from_pretrained(model, ckpt).merge_and_unload()
        return m
    return model  # full ckpts are loaded directly via from_pretrained elsewhere

@torch.no_grad()
def cmd_capeval(args):
    model_id = MODELS.get(args.model, args.model)
    if args.ckpt and os.path.isdir(args.ckpt) and os.path.exists(os.path.join(args.ckpt, "config.json")):
        model, tok = load_model(args.ckpt, eager=False)
        which = os.path.basename(args.ckpt.rstrip("/"))
    else:
        model, tok = load_model(model_id, eager=False)
        if args.ckpt:
            model = load_into(model, tok, args.ckpt)
        which = "base" if not args.ckpt else os.path.basename(args.ckpt.rstrip("/"))
    log(f"CAPEVAL {args.model} ckpt={which}")
    res = {"model": args.model, "ckpt": which}
    try: res["mmlu"] = eval_mmlu(model, tok, args.n)
    except Exception as e: log("mmlu fail", e); res["mmlu"] = None
    try: res["trivia"] = eval_trivia(model, tok, args.n)
    except Exception as e: log("trivia fail", e); res["trivia"] = None
    try: res["math"] = eval_math(model, tok, min(max(args.n // 3, 40), 60))
    except Exception as e: log("math fail", e); res["math"] = None
    json.dump(res, open(tag_path(args.model, f"cap_{which}.json"), "w"), indent=2)
    log("capeval:", res)
    free(model)

@torch.no_grad()
def loglik(model, tok, prompt, cont):
    pi = tok(prompt, add_special_tokens=False).input_ids
    ci = tok(cont, add_special_tokens=False).input_ids
    ids = torch.tensor([pi + ci]).cuda()
    out = model(ids)
    logp = F.log_softmax(out.logits[0].float(), dim=-1)
    s = 0.0
    for j, t in enumerate(ci):
        s += float(logp[len(pi) + j - 1, t])
    return s / max(len(ci), 1)

def eval_mmlu(model, tok, n):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds))); random.Random(1).shuffle(idx); idx = idx[:n]
    ok = 0
    for i in idx:
        r = ds[i]
        q = r["question"]; ch = r["choices"]; ans = r["answer"]
        scores = []
        for c in ch:
            scores.append(loglik(model, tok, f"Question: {q}\nAnswer:", " " + c))
        ok += int(int(np.argmax(scores)) == ans)
    return round(ok / len(idx), 4)

def load_trivia(split):
    from datasets import load_dataset
    last = None
    for cfg in ["rc", "rc.nocontext", "unfiltered.nocontext", "unfiltered"]:
        try:
            return load_dataset("mandarjoshi/trivia_qa", cfg, split=split)
        except Exception as e:
            last = e
    raise last

def eval_trivia(model, tok, n):
    ds = load_trivia("validation")
    idx = list(range(len(ds))); random.Random(2).shuffle(idx); idx = idx[:n]
    ok = 0
    for i in idx:
        r = ds[i]
        ids = tok(f"Question: {r['question']}\nAnswer:", return_tensors="pt").input_ids.cuda()
        g = model.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        ans = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().lower()
        aliases = [a.lower() for a in r["answer"]["aliases"]] + [r["answer"]["value"].lower()]
        ok += int(any(a in ans for a in aliases if a))
    return round(ok / len(idx), 4)

def eval_math(model, tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    except Exception:
        return None
    idx = list(range(len(ds))); random.Random(3).shuffle(idx); idx = idx[:n]
    ok = 0
    for i in idx:
        r = ds[i]
        gold = extract_boxed(r["solution"])
        ids = tok(f"Problem: {r['problem']}\nSolution:", return_tensors="pt",
                  truncation=True, max_length=512).input_ids.cuda()
        g = model.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        out = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(out)
        ok += int(gold is not None and pred is not None and gold == pred)
    return round(ok / len(idx), 4)

def extract_boxed(s):
    k = s.rfind("\\boxed{")
    if k < 0: return None
    i = k + 7; depth = 1; out = ""
    while i < len(s) and depth > 0:
        if s[i] == "{": depth += 1
        elif s[i] == "}": depth -= 1
        if depth > 0: out += s[i]
        i += 1
    return out.strip()

# ================================================================ INTRINSIC
# Rich per-(layer,head) intrinsic-metric collector + metric x metric correlation.
# Metrics (all reduced to per-head vectors of length L*H):
#   static (forward-only, generic+needle text):
#     retrieval, sink, induction, attn_entropy, attn_distance, kv_norm, v_norm, out_norm
#   a-priori forgetting-vulnerability (backward on SFT-domain at BASE, no step):
#     grad_mag (consistent-direction), fisher (empirical diag)
#   outcome (optional short SFT):
#     dW_drift, act_drift
INTRINSIC_METRICS = ["retrieval", "sink", "induction", "attn_entropy", "attn_distance",
                     "kv_norm", "v_norm", "out_norm", "head_alpha", "grad_mag", "fisher",
                     "grad_noise", "dW_drift", "act_drift"]

def _attn_modules(model):
    # returns per-layer attention module or None (None = non-softmax/linear-attn layer)
    out = []
    for layer in layers_of(model):
        a = getattr(layer, "self_attn", None)
        if a is None or not hasattr(a, "q_proj"):
            a = None
        out.append(a)
    return out

def _attn_layer_indices(model):
    return [i for i, a in enumerate(_attn_modules(model)) if a is not None]

def _iter_attn_maps(out_attentions, attn_idx, L):
    # yield (real_layer_index, att[0]) handling hybrid models where
    # len(out_attentions) == #full-attn layers < L (e.g. Qwen3.5 3:1 hybrid)
    if out_attentions is None:
        return
    n = len(out_attentions)
    if n == L:
        for l in range(L):
            if out_attentions[l] is not None:
                yield l, out_attentions[l][0]
    else:
        # map k-th produced attention to its real layer index
        for k in range(n):
            if out_attentions[k] is None:
                continue
            real = attn_idx[k] if k < len(attn_idx) else k
            yield real, out_attentions[k][0]

def cmd_intrinsic(args):
    model_id = MODELS.get(args.model, args.model)
    log(f"INTRINSIC {args.model} domain={args.domain} sft_steps={args.sft_steps}")
    import torch
    model, tok = load_model(model_id, eager=True)
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    hd = head_dim_of(cfg)
    n_kv = getattr(cfg, "num_key_value_heads", H)
    grp = max(H // n_kv, 1)
    M = {m: np.zeros((L, H)) for m in INTRINSIC_METRICS}

    # ---- static forward metrics on generic text ----
    attn_idx = _attn_layer_indices(model)
    kv_acc = np.zeros((L, n_kv)); v_acc = np.zeros((L, n_kv)); out_acc = np.zeros((L, H)); ns = 0
    hooks, store = _install_norm_hooks(model)
    with torch.no_grad():
        for text in GENERIC:
            for d in store.values(): d.clear()
            ids = tok(FILLER + text, return_tensors="pt", truncation=True, max_length=480).input_ids.cuda()
            out = model(ids, output_attentions=True)
            for l, att in _iter_attn_maps(out.attentions, attn_idx, L):
                att = att.float()                       # [H,T,T]
                M["sink"][l] += att[:, 4:, 0].mean(dim=1).cpu().numpy()
                p = att.clamp_min(1e-9)
                ent = -(p * p.log()).sum(-1).mean(dim=1)  # [H]
                M["attn_entropy"][l] += ent.cpu().numpy()
                T = att.shape[-1]
                pos = torch.arange(T, device=att.device).float()
                dist = (att * (pos.view(1, 1, T) - pos.view(1, T, 1)).abs()).sum(-1).mean(dim=1)
                M["attn_distance"][l] += dist.cpu().numpy()
            for l in range(L):
                if l in store.get("k", {}):
                    kv_acc[l] += store["k"][l]; v_acc[l] += store["v"][l]; out_acc[l] += store["out"][l]
            ns += 1
    for h in hooks: h.remove()
    for m in ["sink", "attn_entropy", "attn_distance"]:
        M[m] /= ns
    kv_acc /= ns; v_acc /= ns; out_acc /= ns
    for l in range(L):
        for h in range(H):
            M["kv_norm"][l, h] = kv_acc[l, h // grp]
            M["v_norm"][l, h] = v_acc[l, h // grp]
    M["out_norm"] = out_acc

    # ---- retrieval + induction (reuse detect logic) ----
    M["retrieval"] = _retrieval_scores(model, tok, L, H, attn_idx)
    M["induction"] = _induction_scores(model, tok, L, H, attn_idx)

    # ---- per-head spectral alpha (HT-SR on per-head q/o weight slices) ----
    try:
        M["head_alpha"] = _head_alpha(model, L, H, hd)
    except Exception as e:
        log("head_alpha failed", e)

    big = SIZE_B.get(args.model, 0) >= 20  # grad-checkpoint + short seq for big models
    ml = 256 if big else 512
    # ---- a-priori grad/Fisher/grad-noise on SFT-domain at BASE (no optimizer step) ----
    try:
        gm, fi, gns = _grad_fisher(model, tok, args.domain, L, H, hd, n_kv, grp,
                                   n_batches=args.fisher_batches, use_ckpt=big, maxlen=ml)
        M["grad_mag"], M["fisher"], M["grad_noise"] = gm, fi, gns
    except Exception as e:
        log("grad/fisher failed", e)

    # ---- optional outcome: short SFT -> dW drift + activation drift ----
    if args.sft_steps > 0:
        try:
            base_snap = snapshot_proj(model)
            base_out = _probe_head_acts(model, tok)
            _quick_sft(model, tok, args.domain, args.sft_steps, args.lr, use_ckpt=big, maxlen=ml)
            model.eval()
            dW = _head_drift_from_snap(model, base_snap, L, H, hd)
            M["dW_drift"] = dW
            tuned_out = _probe_head_acts(model, tok)
            M["act_drift"] = _act_drift(base_out, tuned_out)
        except Exception as e:
            log("drift stage failed", e)

    # ---- correlation matrix across metrics (Spearman, over all heads) ----
    present = [m for m in INTRINSIC_METRICS if np.any(M[m])]
    corr = np.full((len(present), len(present)), np.nan)
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            corr[i, j] = spearman(M[a].ravel(), M[b].ravel())
    np.savez(tag_path(args.model, "intrinsic.npz"),
             **{m: M[m] for m in INTRINSIC_METRICS}, present=np.array(present), corr=corr)
    out = {"model": args.model, "layers": L, "heads": H, "metrics_present": present,
           "corr_matrix": {present[i]: {present[j]: round(float(corr[i, j]), 4)
                                        for j in range(len(present))} for i in range(len(present))}}
    # headline couplings vs the two outcome metrics
    for outcome in ["dW_drift", "act_drift", "fisher", "grad_mag"]:
        if outcome in present:
            out[f"vs_{outcome}"] = {a: round(spearman(M[a].ravel(), M[outcome].ravel()), 4)
                                    for a in present if a != outcome}
    json.dump(out, open(tag_path(args.model, "intrinsic.json"), "w"), indent=2)
    log(f"INTRINSIC done; metrics={present}")
    free(model)

def _install_norm_hooks(model):
    store = {"k": {}, "v": {}, "out": {}}
    hooks = []
    cfg = model.config; hd = head_dim_of(cfg)
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    H = cfg.num_attention_heads
    for l, attn in enumerate(_attn_modules(model)):
        if attn is None:
            continue
        def mk_kv(l):
            def hook(mod, inp, outp):
                o = outp.detach()[0].float()       # [T, n_kv*hd]
                o = o.view(o.shape[0], -1, hd).norm(dim=-1).mean(dim=0)  # [n_kv]
                key = "k" if mod is _attn_modules(model)[l].k_proj else "v"
                store[key][l] = o.cpu().numpy()
            return hook
        hooks.append(attn.k_proj.register_forward_hook(mk_kv(l)))
        hooks.append(attn.v_proj.register_forward_hook(mk_kv(l)))
        def mk_out(l):
            def hook(mod, inp):
                x = inp[0].detach()[0].float()     # [T, H*hd]
                store["out"][l] = x.view(x.shape[0], H, hd).norm(dim=-1).mean(dim=0).cpu().numpy()
            return hook
        hooks.append(attn.o_proj.register_forward_pre_hook(mk_out(l)))
    return hooks, store

@torch.no_grad()
def _retrieval_scores(model, tok, L, H, attn_idx=None):
    if attn_idx is None: attn_idx = list(range(L))
    retr = np.zeros((L, H)); rc = 0
    needles = [("the secret access code for project Helios is 47192", " 47192"),
               ("the magic number assigned to Dr. Lin is 80356", " 80356"),
               ("the vault password for the north office is 13947", " 13947")]
    for fact, val in needles:
        pre = "Read the passage and answer the question.\n" + filler_to_len(tok, 120)
        mid = f"\nNote: {fact}.\n" + filler_to_len(tok, 120)
        post = "\nQuestion: recall the number mentioned above. Answer:"
        ids = tok(pre + mid + post, return_tensors="pt", truncation=True, max_length=512).input_ids
        seq = ids[0].tolist(); vt = tok(val, add_special_tokens=False).input_ids; pos = []
        for i in range(len(seq) - len(vt) + 1):
            if seq[i:i+len(vt)] == vt: pos = list(range(i, i+len(vt))); break
        if not pos: continue
        ids = ids.cuda(); out = model(ids, output_attentions=True); last = ids.shape[1]-1
        for l, att in _iter_attn_maps(out.attentions, attn_idx, L):
            retr[l] += att[:, last, pos].sum(dim=1).float().cpu().numpy()
        rc += 1
    return retr / max(rc, 1)

@torch.no_grad()
def _induction_scores(model, tok, L, H, attn_idx=None):
    if attn_idx is None: attn_idx = list(range(L))
    induction = np.zeros((L, H)); rng = random.Random(0)
    seq_a = [rng.randint(1000, min(tok.vocab_size, 30000)-1) for _ in range(60)]
    rep = seq_a + seq_a; ids = torch.tensor([rep]).cuda()
    out = model(ids, output_attentions=True); T = len(rep); half = len(seq_a)
    for l, att in _iter_attn_maps(out.attentions, attn_idx, L):
        sc = np.zeros(H)
        for t in range(half+1, T): sc += att[:, t, t-half].float().cpu().numpy()
        induction[l] = sc / (T-half-1)
    return induction

def _grad_fisher(model, tok, domain, L, H, hd, n_kv, grp, n_batches=8, use_ckpt=False, maxlen=512):
    for p in model.parameters(): p.requires_grad_(False)
    attn = _attn_modules(model)
    for a in attn:
        if a is None: continue
        for nm in ["q_proj", "o_proj"]:
            getattr(a, nm).weight.requires_grad_(True)
    if use_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    feats = build_sft_data(tok, domain, n_batches * 2 + 8, maxlen)
    if not feats: raise RuntimeError("no data for grad/fisher")
    # per-batch reduction to per-head scalars (no full-weight accumulators -> memory-safe at 32B+)
    grad_mag = np.zeros((L, H)); fisher = np.zeros((L, H)); nb = 0
    per_batch = []  # [nb, L, H] per-batch per-head grad norm for grad-noise scale
    model.train()
    for bi in range(min(n_batches, len(feats))):
        ids, labels = collate([feats[bi]], tok.pad_token_id)
        out = model(ids.cuda(), labels=labels.cuda())
        model.zero_grad(set_to_none=True)
        out.loss.backward()
        pb = np.zeros((L, H))
        for l, a in enumerate(attn):
            if a is None: continue
            for nm in ["q_proj", "o_proj"]:
                g = getattr(a, nm).weight.grad
                if g is None: continue
                g = g.detach().float()
                if nm == "q_proj":
                    gh = g.view(H, hd, -1)
                else:
                    gh = g.view(-1, H, hd).permute(1, 0, 2)
                gh = gh.reshape(H, -1)
                grad_mag[l] += gh.abs().mean(dim=1).cpu().numpy()   # L1 (direction-insensitive scale)
                fisher[l] += (gh * gh).sum(dim=1).cpu().numpy()      # empirical Fisher diag
                pb[l] += gh.norm(dim=1).cpu().numpy()
        per_batch.append(pb); nb += 1
    model.zero_grad(set_to_none=True); model.eval()
    if use_ckpt: model.gradient_checkpointing_disable()
    grad_mag /= max(nb, 1); fisher /= max(nb, 1)
    pb = np.stack(per_batch, 0) if per_batch else np.zeros((1, L, H))  # [nb,L,H]
    grad_noise = pb.var(axis=0) / (pb.mean(axis=0) ** 2 + 1e-9)        # coeff of variation^2 per head
    for p in model.parameters(): p.requires_grad_(True)
    return grad_mag, fisher, grad_noise

@torch.no_grad()
def _head_alpha(model, L, H, hd):
    # HT-SR power-law exponent per head, from the per-head q_proj+o_proj weight slice
    out = np.zeros((L, H))
    for l, a in enumerate(_attn_modules(model)):
        if a is None: continue
        q = getattr(a, "q_proj", None); o = getattr(a, "o_proj", None)
        if q is None or o is None: continue
        qw = q.weight.detach().float().view(H, hd, -1)              # [H, hd, in]
        ow = o.weight.detach().float().t().contiguous().view(H, hd, -1)  # [H, hd, in]
        for h in range(H):
            w = torch.cat([qw[h], ow[h]], dim=1)                    # [hd, 2*in]
            try:
                s = torch.linalg.svdvals(w)
                out[l, h] = _ht_alpha(s)
            except Exception:
                pass
    return out

@torch.no_grad()
def _probe_head_acts(model, tok):
    hooks, store = _install_norm_hooks(model)
    for d in store.values(): d.clear()
    ids = tok(FILLER + GENERIC[0] + GENERIC[3], return_tensors="pt",
              truncation=True, max_length=400).input_ids.cuda()
    model(ids)
    out = dict(store["out"])
    for h in hooks: h.remove()
    return out

def _act_drift(base_out, tuned_out):
    L = max(base_out) + 1 if base_out else 0
    H = len(next(iter(base_out.values()))) if base_out else 0
    drift = np.zeros((L, H))
    for l in base_out:
        if l in tuned_out:
            b = base_out[l]; t = tuned_out[l]
            drift[l] = np.abs(t - b) / (np.abs(b) + 1e-6)
    return drift

def _quick_sft(model, tok, domain, steps, lr, use_ckpt=False, maxlen=512):
    # train only attention q/o proj — consistent with the head-drift/Fisher param set
    # and memory-safe for big models (no full-param Adam states).
    for p in model.parameters(): p.requires_grad_(False)
    train_params = []
    # big models: train only q/o proj + SGD (no Adam states) to fit memory
    proj_set = ["q_proj", "o_proj"] if use_ckpt else ["q_proj", "k_proj", "v_proj", "o_proj"]
    for a in _attn_modules(model):
        if a is None: continue
        for nm in proj_set:
            w = getattr(a, nm).weight; w.requires_grad_(True); train_params.append(w)
    if use_ckpt:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    feats = build_sft_data(tok, domain, steps * 2 + 16, maxlen)
    opt = torch.optim.SGD(train_params, lr=lr*10) if use_ckpt else torch.optim.AdamW(train_params, lr=lr)
    model.train()
    for s in range(steps):
        b = feats[s % (len(feats)-1):s % (len(feats)-1)+1]
        ids, labels = collate(b, tok.pad_token_id)
        out = model(ids.cuda(), labels=labels.cuda())
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
        if s % 50 == 0: log(f"  isft {s}/{steps} loss {float(out.loss):.4f}")
    if use_ckpt: model.gradient_checkpointing_disable()

@torch.no_grad()
def _head_drift_from_snap(model, snap, L, H, hd):
    drift = np.zeros((L, H))
    for i, lyr in enumerate(layers_of(model)):
        a = getattr(lyr, "self_attn", None)
        if a is None or not hasattr(a, "q_proj") or (i, "q_proj") not in snap: continue
        for nm in ["q_proj", "o_proj"]:
            w = getattr(a, nm).weight.detach().float().cpu()
            dw = (w - snap[(i, nm)]).numpy()
            if nm == "q_proj":
                drift[i] += np.linalg.norm(dw.reshape(H, -1), axis=1)
            else:
                drift[i] += np.linalg.norm(dw.reshape(-1, H, hd).transpose(1, 0, 2).reshape(H, -1), axis=1)
    return drift

# ================================================================ FACTS
# Broad per-LAYER fact-finding across angles:
#  representation geometry · information theory (logit lens) · parameter spectra
#  (+ cross-link per-head coupling metrics from intrinsic.npz, mean-pooled to layers)
# Goal: a per-layer metric table + metric×metric correlation matrix -> find facts.
FACTS_METRICS = [
    # representation geometry (from hidden states)
    "resid_norm", "anisotropy", "eff_rank", "repr_entropy", "act_kurtosis",
    "act_sparsity", "update_norm", "curvature", "intrinsic_dim", "cka_adjacent",
    # information theory (logit lens through final norm + lm_head)
    "ll_entropy", "ll_kl_to_final", "ll_top1_depth", "tuned_lens_kl", "tuned_lens_depth",
    # parameter spectra (per layer; aggregated over the 7 proj matrices unless noted)
    "w_stable_rank", "w_eff_rank", "w_spectral_norm", "w_ht_alpha",
    "down_stable_rank", "down_ht_alpha", "mlp_gain",
    # massive activations
    "massive_count", "massive_max",
]

def _twonn_id(H):
    # TwoNN intrinsic-dimension estimator (Facco et al. 2017) on token reps H [T,d].
    T = H.shape[0]
    if T < 10: return 0.0
    d = torch.cdist(H, H)
    d.fill_diagonal_(float("inf"))
    r1 = d.min(dim=1).values
    d2 = d.clone()
    d2[torch.arange(T), d.argmin(dim=1)] = float("inf")
    r2 = d2.min(dim=1).values
    mu = (r2 / (r1 + 1e-9)).clamp_min(1.0 + 1e-6)
    mu = mu[torch.isfinite(mu)]
    val = float(mu.log().sum())
    return (mu.numel() / val) if val > 0 else 0.0

def _linear_cka(X, Y):
    # linear CKA between two [T,d] activation matrices
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.t() @ Y).norm() ** 2
    xx = (X.t() @ X).norm(); yy = (Y.t() @ Y).norm()
    return float(xy / (xx * yy + 1e-9))

def _eff_rank_from_svals(s):
    s = s[s > 0]
    if s.numel() == 0: return 0.0
    p = (s / s.sum())
    ent = -(p * (p + 1e-12).log()).sum()
    return float(ent.exp())

def _stable_rank_from_svals(s):
    s2 = s * s
    return float(s2.sum() / (s2.max() + 1e-12))

def _ht_alpha(s):
    # Hill estimator of the power-law tail exponent of the eigenvalue (s^2) spectrum.
    lam = (s * s).sort(descending=True).values
    k = max(int(0.5 * lam.numel()), 10)
    lam = lam[:k]
    lam_min = lam[-1]
    n = lam.numel() - 1
    if n <= 0 or lam_min <= 0: return 0.0
    hill = (lam[:-1] / lam_min).log().mean()
    return float(1.0 + 1.0 / (hill + 1e-9))

@torch.no_grad()
def cmd_facts(args):
    model_id = MODELS.get(args.model, args.model)
    log(f"FACTS {args.model}")
    model, tok = load_model(model_id, eager=False)
    cfg = model.config
    L = cfg.num_hidden_layers
    F = {m: np.zeros(L) for m in FACTS_METRICS}

    # ---- representation + info-theory from hidden states ----
    texts = GENERIC + [FILLER + g for g in GENERIC[:3]]
    nrm = getattr(model.model, "norm", None)
    lm_head = model.lm_head if hasattr(model, "lm_head") else None
    accum = {m: np.zeros(L) for m in
             ["resid_norm", "anisotropy", "eff_rank", "repr_entropy", "act_kurtosis",
              "act_sparsity", "update_norm", "curvature", "intrinsic_dim", "cka_adjacent",
              "ll_entropy", "ll_kl_to_final", "ll_top1_depth"]}
    ns = 0
    ridge_Hl = {l: [] for l in range(L)}; ridge_Hf = []  # for tuned-lens
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids.cuda()
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple L+1 of [1,T,d]
        final_logits = out.logits[0].float()
        final_p = F_softmax_safe(final_logits)
        hfinal = hs[-1][0].float()
        ridge_Hf.append(hfinal.cpu())
        for l in range(L):
            h = hs[l + 1][0].float()                  # [T, d]
            T = h.shape[0]
            accum["resid_norm"][l] += float(h.norm(dim=-1).mean())
            hc = h - h.mean(0, keepdim=True)
            hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6)
            G = hn @ hn.t()
            off = (G.sum() - G.diag().sum()) / (T * (T - 1) + 1e-9)
            accum["anisotropy"][l] += float(off)
            try:
                s = torch.linalg.svdvals(hc)
                accum["eff_rank"][l] += _eff_rank_from_svals(s)
                p = s / (s.sum() + 1e-9)
                accum["repr_entropy"][l] += float(-(p * (p + 1e-12).log()).sum())
            except Exception:
                pass
            try: accum["intrinsic_dim"][l] += _twonn_id(h)
            except Exception: pass
            if l < L - 1:
                try: accum["cka_adjacent"][l] += _linear_cka(h, hs[l + 2][0].float())
                except Exception: pass
            x = h.flatten()
            mu = x.mean(); sd = x.std() + 1e-6
            accum["act_kurtosis"][l] += float((((x - mu) / sd) ** 4).mean())
            accum["act_sparsity"][l] += float((x.abs() < 0.1 * sd).float().mean())
            accum["update_norm"][l] += float((hs[l + 1][0].float() - hs[l][0].float()).norm(dim=-1).mean())
            if 0 < l < L - 1:
                curv = (hs[l + 2][0].float() - 2 * hs[l + 1][0].float() + hs[l][0].float()).norm(dim=-1).mean()
                accum["curvature"][l] += float(curv)
            ridge_Hl[l].append(h.cpu())
            if nrm is not None and lm_head is not None:
                try:
                    ll = lm_head(nrm(hs[l + 1][0].float().to(hs[l + 1].dtype))).float()
                    pl = F_softmax_safe(ll)
                    accum["ll_entropy"][l] += float((-(pl * (pl + 1e-12).log()).sum(-1)).mean())
                    kl = (pl * ((pl + 1e-12).log() - (final_p + 1e-12).log())).sum(-1)
                    accum["ll_kl_to_final"][l] += float(kl.mean())
                    accum["ll_top1_depth"][l] += float((pl.argmax(-1) == final_p.argmax(-1)).float().mean())
                except Exception:
                    pass
        ns += 1
    for m in accum:
        F[m] = accum[m] / max(ns, 1)

    # ---- tuned-lens-lite: ridge map h_l -> h_final, then lm_head; KL & depth vs final ----
    if nrm is not None and lm_head is not None:
        try:
            Hf = torch.cat(ridge_Hf, 0).cuda()                     # [N, d]
            pf = F_softmax_safe(lm_head(nrm(Hf.to(hs[-1].dtype))).float())
            d = Hf.shape[1]; I = torch.eye(d, device=Hf.device)
            for l in range(L):
                Hl = torch.cat(ridge_Hl[l], 0).cuda()              # [N, d]
                A = torch.linalg.solve(Hl.t() @ Hl + 1e-2 * I, Hl.t() @ Hf)  # [d,d]
                pred = Hl @ A
                pl = F_softmax_safe(lm_head(nrm(pred.to(hs[-1].dtype))).float())
                kl = (pl * ((pl + 1e-12).log() - (pf + 1e-12).log())).sum(-1)
                F["tuned_lens_kl"][l] = float(kl.mean())
                F["tuned_lens_depth"][l] = float((pl.argmax(-1) == pf.argmax(-1)).float().mean())
                del Hl, A, pred, pl
        except Exception as e:
            log("tuned-lens failed", e)

    # ---- massive activations per layer ----
    for text in GENERIC[:3]:
        ids = tok(FILLER + text, return_tensors="pt", truncation=True, max_length=256).input_ids.cuda()
        out = model(ids, output_hidden_states=True)
        for l in range(L):
            cm = out.hidden_states[l + 1][0].abs().float().max(dim=0).values
            med = cm.median() + 1e-6
            F["massive_count"][l] = max(F["massive_count"][l], float((cm >= 6 * med).sum()))
            F["massive_max"][l] = max(F["massive_max"][l], float((cm / med).max()))

    # ---- parameter spectra per layer (robust to hybrid/MoE layers) ----
    lyrs = layers_of(model)
    for l in range(L):
        lyr = lyrs[l]
        a = getattr(lyr, "self_attn", None) or getattr(lyr, "linear_attn", None) or getattr(lyr, "attn", None)
        mlp = getattr(lyr, "mlp", None) or getattr(lyr, "feed_forward", None)
        srs, ers, sns, alphas = [], [], [], []
        cand = []
        if a is not None:
            cand += [(a, "q_proj"), (a, "k_proj"), (a, "v_proj"), (a, "o_proj")]
        if mlp is not None:
            cand += [(mlp, "gate_proj"), (mlp, "up_proj"), (mlp, "down_proj")]
        for mod, nm in cand:
            w = getattr(mod, nm, None)
            if w is None or not hasattr(w, "weight"): continue
            try:
                s = torch.linalg.svdvals(w.weight.detach().float())
                srs.append(_stable_rank_from_svals(s)); ers.append(_eff_rank_from_svals(s))
                sns.append(float(s.max())); alphas.append(_ht_alpha(s))
                if nm == "down_proj":
                    F["down_stable_rank"][l] = _stable_rank_from_svals(s); F["down_ht_alpha"][l] = _ht_alpha(s)
            except Exception:
                pass
        if srs:
            F["w_stable_rank"][l] = float(np.mean(srs)); F["w_eff_rank"][l] = float(np.mean(ers))
            F["w_spectral_norm"][l] = float(np.mean(sns)); F["w_ht_alpha"][l] = float(np.mean(alphas))
        # mlp gain: ||up||/||down|| norm proxy
        up = getattr(mlp, "up_proj", None) if mlp is not None else None
        dn = getattr(mlp, "down_proj", None) if mlp is not None else None
        if up is not None and dn is not None and hasattr(up, "weight") and hasattr(dn, "weight"):
            F["mlp_gain"][l] = float(up.weight.detach().float().norm() / (dn.weight.detach().float().norm() + 1e-6))

    # ---- cross-link per-head coupling metrics (mean-pool to layers) ----
    intr = os.path.join(ART, args.model, "intrinsic.npz")
    extra = {}
    if os.path.exists(intr):
        z = np.load(intr, allow_pickle=True)
        for k in ["retrieval", "sink", "fisher", "dW_drift", "grad_mag", "attn_entropy"]:
            if k in z.files:
                arr = z[k]
                if arr.ndim == 2 and arr.shape[0] == L:
                    extra["hd_" + k] = arr.mean(axis=1)

    # ---- correlation matrix across all per-layer metrics ----
    allm = {**{m: F[m] for m in FACTS_METRICS if np.any(F[m])}, **extra}
    names = list(allm.keys())
    corr = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            corr[i, j] = spearman(allm[a], allm[b])
    np.savez(tag_path(args.model, "facts.npz"),
             **{("F_" + k): v for k, v in allm.items()}, names=np.array(names), corr=corr)
    # strongest off-diagonal correlations = the "facts"
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not np.isnan(corr[i, j]):
                pairs.append((names[i], names[j], round(float(corr[i, j]), 3)))
    pairs.sort(key=lambda x: -abs(x[2]))
    out = {"model": args.model, "layers": L, "metrics": names,
           "top_correlations": pairs[:30],
           "corr": {names[i]: {names[j]: round(float(corr[i, j]), 3)
                               for j in range(len(names)) if not np.isnan(corr[i, j])}
                    for i in range(len(names))}}
    json.dump(out, open(tag_path(args.model, "facts.json"), "w"), indent=2)
    log(f"FACTS done; {len(names)} metrics; top fact: {pairs[0] if pairs else None}")
    free(model)

def F_softmax_safe(logits):
    return torch.softmax(logits, dim=-1)

# ================================================================ GRID DATASETS
# Unified loader: returns (texts_for_passA, sft_pairs_for_passCD) for each benchmark.
# texts = raw strings for forward metrics (truncated to ctxlen by caller).
# sft_pairs = (prompt, completion) for Fisher/drift SFT.
# is_long flags benchmarks whose inputs exercise the long-context (inference) frontier.
GRID_DATASETS = {
    # name: (long-context?, domain-tag)
    "wikitext":   (False, "generic"),
    "mmlu":       (False, "knowledge"),
    "math":       (False, "math"),
    "gsm8k":      (False, "math"),
    "triviaqa":   (False, "knowledge"),
    "bbh":        (False, "reasoning"),
    "squad":      (True,  "qa"),
    "hotpotqa":   (True,  "multihop"),
    "quality":    (True,  "longdoc"),
    "narrativeqa":(True,  "longdoc"),
    "musr":       (True,  "reasoning"),
    "msmarco":    (True,  "retrieval"),
}

def _ld(*a, **k):
    from datasets import load_dataset
    return load_dataset(*a, **k)

def load_grid_dataset(tok, name, n=12):
    texts, pairs = [], []
    try:
        if name == "wikitext":
            ds = None
            for c in ["wikitext-103-v1", "wikitext-103-raw-v1", "wikitext-2-raw-v1"]:
                try: ds = _ld("Salesforce/wikitext", c, split="test"); break
                except Exception: pass
            buf = [r["text"] for r in ds if len(r["text"]) > 200][:n * 3]
            texts = buf[:n]; pairs = [(t[:400], " " + t[400:800]) for t in buf[:n] if len(t) > 500]
        elif name == "mmlu":
            ds = _ld("cais/mmlu", "all", split="test")
            idx = list(range(len(ds))); random.Random(1).shuffle(idx)
            for i in idx[:n * 2]:
                r = ds[i]; ch = "\n".join(f"{c}. {o}" for c, o in zip("ABCD", r["choices"]))
                p = f"Question: {r['question']}\n{ch}\nAnswer:"
                texts.append(p + f" {'ABCD'[r['answer']]}"); pairs.append((p, f" {'ABCD'[r['answer']]}"))
        elif name == "math":
            for cfg in ["algebra", "prealgebra", "number_theory"]:
                try:
                    ds = _ld("EleutherAI/hendrycks_math", cfg, split="train")
                    for r in ds:
                        texts.append(f"Problem: {r['problem']}\nSolution: {r['solution'][:600]}")
                        pairs.append((f"Problem: {r['problem']}\nSolution:", " " + r["solution"]))
                        if len(pairs) >= n * 2: break
                except Exception: pass
                if len(pairs) >= n * 2: break
        elif name == "gsm8k":
            ds = _ld("openai/gsm8k", "main", split="train")
            for r in list(ds)[:n * 2]:
                p = f"Question: {r['question']}\nAnswer:"
                texts.append(p + " " + r["answer"]); pairs.append((p, " " + r["answer"]))
        elif name == "triviaqa":
            ds = load_trivia("train")
            for r in list(ds)[:n * 2]:
                a = r["answer"]["value"]; p = f"Question: {r['question']}\nAnswer:"
                texts.append(p + " " + a); pairs.append((p, " " + a))
        elif name == "bbh":
            got = False
            for repo, cfg in [("lukaemon/bbh", "date_understanding"), ("SaylorTwift/bbh", "date_understanding"),
                              ("maveriq/bigbenchhard", "date_understanding")]:
                try:
                    ds = _ld(repo, cfg, split="test")
                    for r in list(ds)[:n * 2]:
                        q = r.get("input") or r.get("question"); a = str(r.get("target") or r.get("answer"))
                        p = f"{q}\nAnswer:"; texts.append(p + " " + a); pairs.append((p, " " + a))
                    got = True; break
                except Exception: pass
        elif name == "squad":
            ds = _ld("rajpurkar/squad_v2", split="validation")
            for r in list(ds)[:n * 2]:
                if not r["answers"]["text"]: continue
                a = r["answers"]["text"][0]
                p = f"Context: {r['context']}\nQuestion: {r['question']}\nAnswer:"
                texts.append(p + " " + a); pairs.append((p, " " + a))
        elif name == "hotpotqa":
            ds = _ld("hotpotqa/hotpot_qa", "distractor", split="validation")
            for r in list(ds)[:n * 2]:
                ctx = " ".join(" ".join(s) for s in r["context"]["sentences"])
                p = f"Context: {ctx[:3000]}\nQuestion: {r['question']}\nAnswer:"
                texts.append(p + " " + r["answer"]); pairs.append((p[:2000], " " + r["answer"]))
        elif name == "quality":
            ds = _ld("emozilla/quality", split="validation")
            for r in list(ds)[:n * 2]:
                art = r.get("article") or ""; q = r.get("question") or ""
                p = f"{art[:4000]}\nQuestion: {q}\nAnswer:"
                texts.append(p); pairs.append((p[:2000], " " + str(r.get("options", [""])[0])[:80]))
        elif name == "narrativeqa":
            ds = _ld("deepmind/narrativeqa", split="validation")
            for r in list(ds)[:n * 2]:
                doc = r["document"]["summary"]["text"]; q = r["question"]["text"]; a = r["answers"][0]["text"]
                p = f"{doc[:4000]}\nQuestion: {q}\nAnswer:"
                texts.append(p + " " + a); pairs.append((p[:2000], " " + a))
        elif name == "musr":
            ds = _ld("TAUR-Lab/MuSR", split="murder_mysteries")
            for r in list(ds)[:n * 2]:
                p = f"{r['narrative'][:4000]}\nQuestion: {r['question']}\nAnswer:"
                texts.append(p); pairs.append((p[:2000], " " + str(r.get('answer_choice', ''))[:60]))
        elif name == "msmarco":
            ds = _ld("microsoft/ms_marco", "v2.1", split="validation")
            for r in list(ds)[:n * 2]:
                psg = " ".join(r["passages"]["passage_text"][:5]); q = r["query"]
                p = f"Passages: {psg[:3500]}\nQuery: {q}\nAnswer:"
                texts.append(p); pairs.append((p[:2000], " " + (r["answers"][0] if r["answers"] else "")[:80]))
    except Exception as e:
        log(f"load_grid_dataset {name} failed", e)
    return texts[:n], [p for p in pairs if p[1].strip()][:max(n * 2, 32)]

# ================================================================ GRID (unified)
# Frontier tags: LC = long-context/inference, CF = catastrophic-forgetting/training,
# ST = structural substrate (covariate). Used to organize the correlation tables.
FRONTIER = {
    # per-head
    "sink": "LC", "retrieval": "LC", "induction": "LC", "attn_entropy": "LC",
    "attn_distance": "LC", "prev_token": "LC", "receptive_field": "LC",
    "kv_norm": "LC", "v_norm": "LC", "out_norm": "ST",
    "head_alpha": "CF", "head_wnorm": "ST", "ov_norm": "ST",
    "fisher": "CF", "grad_mag": "CF", "grad_noise": "CF", "dW_drift": "CF", "act_drift": "CF",
    # per-layer
    "resid_norm": "ST", "anisotropy": "ST", "eff_rank": "ST", "intrinsic_dim": "ST",
    "spectral_decay": "ST", "update_norm": "ST", "curvature": "ST", "cka_adjacent": "ST",
    "token_mixing": "LC", "ll_kl_to_final": "ST", "ll_top1_depth": "ST", "ll_entropy": "ST",
    "tuned_lens_kl": "ST", "tuned_lens_depth": "ST", "lens_gap": "ST",
    "act_kurtosis": "ST", "act_sparsity": "ST", "gini": "ST", "dead_frac": "ST",
    "massive_max": "ST", "massive_count": "ST",
    "w_stable_rank": "ST", "w_eff_rank": "ST", "w_spectral_norm": "ST", "w_ht_alpha": "CF",
    "condition_num": "ST", "weight_entropy": "ST", "down_stable_rank": "ST", "mlp_gain": "ST",
}

def _gini(x):
    x = np.sort(np.abs(x.ravel())); n = x.size
    if n == 0 or x.sum() == 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

@torch.no_grad()
def _grid_spectra(model, L, H, hd):
    # Pass B: parameter spectra (weights only, dataset-independent). Cached per model.
    cache = os.path.join(ART, "_spectra_cache.npz")  # keyed by model dir below
    pl = {k: np.zeros(L) for k in ["w_stable_rank", "w_eff_rank", "w_spectral_norm",
          "w_ht_alpha", "condition_num", "weight_entropy", "down_stable_rank", "mlp_gain"]}
    ph = {k: np.zeros((L, H)) for k in ["head_alpha", "head_wnorm", "ov_norm"]}
    lyrs = layers_of(model)
    for l in range(L):
        lyr = lyrs[l]
        a = getattr(lyr, "self_attn", None) or getattr(lyr, "linear_attn", None)
        mlp = getattr(lyr, "mlp", None) or getattr(lyr, "feed_forward", None)
        srs, ers, sns, alphas, conds, ents = [], [], [], [], [], []
        cand = []
        if a is not None: cand += [(a, n) for n in ["q_proj", "k_proj", "v_proj", "o_proj"]]
        if mlp is not None: cand += [(mlp, n) for n in ["gate_proj", "up_proj", "down_proj"]]
        for mod, nm in cand:
            w = getattr(mod, nm, None)
            if w is None or not hasattr(w, "weight"): continue
            try:
                s = torch.linalg.svdvals(w.weight.detach().float())
                srs.append(_stable_rank_from_svals(s)); ers.append(_eff_rank_from_svals(s))
                sns.append(float(s.max())); alphas.append(_ht_alpha(s))
                conds.append(float(s.max() / (s[s > 0].min() + 1e-9)))
                p = s / (s.sum() + 1e-9); ents.append(float(-(p * (p + 1e-12).log()).sum()))
                if nm == "down_proj": pl["down_stable_rank"][l] = _stable_rank_from_svals(s)
            except Exception: pass
        if srs:
            pl["w_stable_rank"][l] = np.mean(srs); pl["w_eff_rank"][l] = np.mean(ers)
            pl["w_spectral_norm"][l] = np.mean(sns); pl["w_ht_alpha"][l] = np.mean(alphas)
            pl["condition_num"][l] = np.mean(conds); pl["weight_entropy"][l] = np.mean(ents)
        if mlp is not None:
            up = getattr(mlp, "up_proj", None); dn = getattr(mlp, "down_proj", None)
            if up is not None and dn is not None and hasattr(up, "weight"):
                pl["mlp_gain"][l] = float(up.weight.detach().float().norm() / (dn.weight.detach().float().norm() + 1e-6))
        # per-head param geometry
        if a is not None and hasattr(a, "q_proj") and hasattr(a, "o_proj"):
            qw = a.q_proj.weight.detach().float().view(H, hd, -1)
            ow = a.o_proj.weight.detach().float().t().contiguous().view(H, hd, -1)
            vproj = getattr(a, "v_proj", None)
            vw = vproj.weight.detach().float() if vproj is not None else None
            n_kv = vw.shape[0] // hd if vw is not None else H
            grp = max(H // n_kv, 1)
            for h in range(H):
                w = torch.cat([qw[h], ow[h]], dim=1)
                try: ph["head_alpha"][l, h] = _ht_alpha(torch.linalg.svdvals(w))
                except Exception: pass
                ph["head_wnorm"][l, h] = float(qw[h].norm() + ow[h].norm())
                if vw is not None:
                    vh = vw[(h // grp) * hd:(h // grp) * hd + hd]    # [hd, in]
                    ph["ov_norm"][l, h] = float((ow[h] @ vh).norm()) if ow[h].shape[1] == vh.shape[0] else float(ow[h].norm() * vh.norm())
    return pl, ph

def cmd_grid(args):
    model_id = MODELS.get(args.model, args.model)
    ds = args.dataset
    log(f"GRID {args.model} x {ds} ctxlen={args.ctxlen}")
    model, tok = load_model(model_id, eager=True)
    cfg = model.config
    L, H, hd = cfg.num_hidden_layers, cfg.num_attention_heads, head_dim_of(cfg)
    n_kv = getattr(cfg, "num_key_value_heads", H); grp = max(H // n_kv, 1)
    attn_idx = _attn_layer_indices(model)
    PH = {}; PL = {}; SC = {}

    # ---- Pass B: parameter spectra (cache per model; concurrency-safe) ----
    spec_f = os.path.join(ART, args.model, "spectra.npz")
    loaded = False
    if os.path.exists(spec_f):
        try:
            z = np.load(spec_f)
            for k in z.files:
                if k.startswith("pl_"): PL[k[3:]] = z[k]
                elif k.startswith("ph_"): PH[k[3:]] = z[k]
            loaded = True
        except Exception:
            loaded = False
    if not loaded:
        pl_spec, ph_spec = _grid_spectra(model, L, H, hd)
        PL.update(pl_spec); PH.update(ph_spec)
        try:
            tmp = spec_f + f".{os.getpid()}.tmp.npz"   # end in .npz so savez keeps the name
            np.savez(tmp, **{"pl_" + k: v for k, v in pl_spec.items()},
                     **{"ph_" + k: v for k, v in ph_spec.items()})
            os.replace(tmp, spec_f)  # atomic
        except Exception as e:
            log("spectra cache write skipped", e)

    # ---- dataset text + sft pairs ----
    texts, pairs = load_grid_dataset(tok, ds, n=args.n_texts)
    if not texts:
        log(f"no texts for {ds}; abort"); free(model); return

    # ---- Pass A: ONE forward/text (hidden + attn + labels) -> angles 1-8 + surprisal ----
    headm = {k: np.zeros((L, H)) for k in ["sink", "attn_entropy", "attn_distance",
             "prev_token", "receptive_field", "kv_norm", "v_norm", "out_norm"]}
    laym = {k: np.zeros(L) for k in ["resid_norm", "anisotropy", "eff_rank", "intrinsic_dim",
            "spectral_decay", "update_norm", "curvature", "cka_adjacent", "token_mixing",
            "ll_kl_to_final", "ll_top1_depth", "ll_entropy", "act_kurtosis", "act_sparsity",
            "gini", "dead_frac", "massive_max", "massive_count"]}
    nrm = getattr(model.model, "norm", None) or getattr(getattr(model, "model", model), "norm", None)
    lm_head = getattr(model, "lm_head", None)
    hooks, store = _install_norm_hooks(model)
    ns = 0; nll_sum = 0.0; nll_tok = 0
    ridge_Hl = {l: [] for l in range(L)}; ridge_Hf = []
    with torch.no_grad():
        for text in texts:
            for d in store.values(): d.clear()
            ids = tok(text, return_tensors="pt", truncation=True, max_length=args.ctxlen).input_ids.cuda()
            if ids.shape[1] < 8: continue
            out = model(ids, output_attentions=True, output_hidden_states=True)
            # surprisal (NLL) on this text
            lg = out.logits[0].float()
            nll = F.cross_entropy(lg[:-1], ids[0, 1:], reduction="sum")
            nll_sum += float(nll); nll_tok += ids.shape[1] - 1
            T = ids.shape[1]
            for l, att in _iter_attn_maps(out.attentions, attn_idx, L):
                att = att.float()
                headm["sink"][l] += att[:, 4:, 0].mean(dim=1).cpu().numpy()
                p = att.clamp_min(1e-9); ent = -(p * p.log()).sum(-1)
                headm["attn_entropy"][l] += ent.mean(dim=1).cpu().numpy()
                headm["receptive_field"][l] += ent.mean(dim=1).exp().cpu().numpy()
                pos = torch.arange(T, device=att.device).float()
                dist = (att * (pos.view(1, 1, T) - pos.view(1, T, 1)).abs()).sum(-1).mean(dim=1)
                headm["attn_distance"][l] += dist.cpu().numpy()
                pt = att.diagonal(offset=-1, dim1=1, dim2=2).mean(dim=1)
                headm["prev_token"][l] += pt.cpu().numpy()
                # token mixing = off-diagonal mass (1 - self+sink); per layer avg over heads
                offdiag = 1.0 - att.diagonal(dim1=1, dim2=2).mean(dim=1) - att[:, :, 0].mean(dim=1)
                laym["token_mixing"][l] += float(offdiag.mean())
            for l in range(L):
                if l in store.get("k", {}):
                    headm["kv_norm"][l] += np.repeat(store["k"][l], grp)[:H]
                    headm["v_norm"][l] += np.repeat(store["v"][l], grp)[:H]
                    headm["out_norm"][l] += store["out"][l]
            hs = out.hidden_states; hf = hs[-1][0].float(); ridge_Hf.append(hf.cpu())
            final_p = torch.softmax(lg, -1)
            for l in range(L):
                h = hs[l + 1][0].float()
                laym["resid_norm"][l] += float(h.norm(dim=-1).mean())
                hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6); Gm = hn @ hn.t()
                laym["anisotropy"][l] += float((Gm.sum() - Gm.diag().sum()) / (T * (T - 1) + 1e-9))
                try:
                    s = torch.linalg.svdvals(h - h.mean(0, keepdim=True))
                    laym["eff_rank"][l] += _eff_rank_from_svals(s)
                    ls = s[s > 0].log(); xi = torch.arange(ls.numel(), device=ls.device).float()
                    laym["spectral_decay"][l] += float(-((xi - xi.mean()) * (ls - ls.mean())).sum() / ((xi - xi.mean()) ** 2).sum().clamp_min(1e-9))
                except Exception: pass
                try: laym["intrinsic_dim"][l] += _twonn_id(h)
                except Exception: pass
                if l < L - 1:
                    try: laym["cka_adjacent"][l] += _linear_cka(h, hs[l + 2][0].float())
                    except Exception: pass
                laym["update_norm"][l] += float((h - hs[l][0].float()).norm(dim=-1).mean())
                if 0 < l < L - 1:
                    laym["curvature"][l] += float((hs[l + 2][0].float() - 2 * h + hs[l][0].float()).norm(dim=-1).mean())
                x = h.flatten(); mu = x.mean(); sd = x.std() + 1e-6
                laym["act_kurtosis"][l] += float((((x - mu) / sd) ** 4).mean())
                laym["act_sparsity"][l] += float((x.abs() < 0.1 * sd).float().mean())
                laym["gini"][l] += _gini(h.abs().float().cpu().numpy())
                cmax = h.abs().max(dim=0).values; med = cmax.median() + 1e-6
                laym["dead_frac"][l] += float((cmax < 0.05 * med).float().mean())
                laym["massive_max"][l] = max(laym["massive_max"][l], float((cmax / med).max()))
                laym["massive_count"][l] = max(laym["massive_count"][l], float((cmax >= 6 * med).sum()))
                ridge_Hl[l].append(h.cpu())
                if nrm is not None and lm_head is not None:
                    try:
                        pl_ = torch.softmax(lm_head(nrm(h.to(hs[-1].dtype))).float(), -1)
                        laym["ll_entropy"][l] += float((-(pl_ * (pl_ + 1e-12).log()).sum(-1)).mean())
                        laym["ll_kl_to_final"][l] += float((pl_ * ((pl_ + 1e-12).log() - (final_p + 1e-12).log())).sum(-1).mean())
                        laym["ll_top1_depth"][l] += float((pl_.argmax(-1) == final_p.argmax(-1)).float().mean())
                    except Exception: pass
            ns += 1
    for h in hooks: h.remove()
    for k in headm: headm[k] /= max(ns, 1)
    for k in laym:
        if k not in ("massive_max", "massive_count"): laym[k] /= max(ns, 1)
    SC["surprisal"] = nll_sum / max(nll_tok, 1)

    # retrieval + induction (LC site detectors)
    headm["retrieval"] = _retrieval_scores(model, tok, L, H, attn_idx)
    headm["induction"] = _induction_scores(model, tok, L, H, attn_idx)

    # tuned-lens-lite (ridge h_l -> h_final)
    laym["tuned_lens_kl"] = np.zeros(L); laym["tuned_lens_depth"] = np.zeros(L)
    if nrm is not None and lm_head is not None:
        try:
          with torch.no_grad():
            Hf = torch.cat(ridge_Hf, 0).cuda(); d = Hf.shape[1]; I = torch.eye(d, device=Hf.device)
            pf = torch.softmax(lm_head(nrm(Hf.to(hs[-1].dtype))).float(), -1)
            for l in range(L):
                Hl = torch.cat(ridge_Hl[l], 0).cuda()
                A = torch.linalg.solve(Hl.t() @ Hl + 1e-2 * I, Hl.t() @ Hf)
                pl_ = torch.softmax(lm_head(nrm((Hl @ A).to(hs[-1].dtype))).float(), -1)
                laym["tuned_lens_kl"][l] = float((pl_ * ((pl_ + 1e-12).log() - (pf + 1e-12).log())).sum(-1).mean())
                laym["tuned_lens_depth"][l] = float((pl_.argmax(-1) == pf.argmax(-1)).float().mean())
                del Hl, A, pl_
        except Exception as e: log("tuned-lens failed", e)
    laym["lens_gap"] = laym["tuned_lens_kl"] - laym["ll_kl_to_final"]

    # ---- Pass C: a-priori Fisher/grad/grad-noise on this dataset (CF frontier) ----
    big = SIZE_B.get(args.model, 0) >= 20; ml = 256 if big else 512
    if pairs:
        try:
            gm, fi, gns = _grad_fisher_pairs(model, tok, pairs, L, H, hd, n_kv, grp,
                                             n_batches=args.fisher_batches, use_ckpt=big, maxlen=ml)
            headm["grad_mag"], headm["fisher"], headm["grad_noise"] = gm, fi, gns
        except Exception as e: log("grad/fisher failed", e)

    # ---- Pass D: short SFT on this dataset -> drift (CF frontier) ----
    if pairs and args.sft_steps > 0:
        try:
            snap = snapshot_proj(model); base_out = _probe_head_acts(model, tok)
            _quick_sft_pairs(model, tok, pairs, args.sft_steps, args.lr, use_ckpt=big, maxlen=ml)
            model.eval()
            headm["dW_drift"] = _head_drift_from_snap(model, snap, L, H, hd)
            headm["act_drift"] = _act_drift(base_out, _probe_head_acts(model, tok))
        except Exception as e: log("drift failed", e)

    # ---- save unified record ----
    out = {"model": args.model, "dataset": ds, "ctxlen": args.ctxlen, "layers": L, "heads": H,
           "is_long": GRID_DATASETS.get(ds, (False, ""))[0], "domain": GRID_DATASETS.get(ds, (False, "?"))[1],
           "scalars": SC, "frontier": FRONTIER}
    np.savez(tag_path(args.model, f"grid_{ds}.npz"),
             **{"H_" + k: v for k, v in headm.items()}, **{"L_" + k: v for k, v in laym.items()},
             **{"P_" + k: v for k, v in PL.items()}, **{"PH_" + k: v for k, v in PH.items()})
    json.dump(out, open(tag_path(args.model, f"grid_{ds}.json"), "w"), indent=2)
    log(f"GRID done {args.model} x {ds}: heads={list(headm)} surprisal={SC['surprisal']:.3f}")
    free(model)

def _grad_fisher_pairs(model, tok, pairs, L, H, hd, n_kv, grp, n_batches=8, use_ckpt=False, maxlen=512):
    feats = _pairs_to_feats(tok, pairs, maxlen)
    return _grad_fisher_feats(model, feats, tok, L, H, hd, n_batches, use_ckpt)

def _pairs_to_feats(tok, pairs, maxlen):
    feats = []
    for p, c in pairs:
        pi = tok(p, add_special_tokens=False).input_ids
        ci = tok(c, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (pi + ci)[:maxlen]; labels = ([-100] * len(pi) + ci)[:maxlen]
        if len(ids) > len(pi): feats.append((ids, labels))
    return feats

def _grad_fisher_feats(model, feats, tok, L, H, hd, n_batches, use_ckpt):
    for p in model.parameters(): p.requires_grad_(False)
    attn = _attn_modules(model)
    for a in attn:
        if a is None: continue
        for nm in ["q_proj", "o_proj"]: getattr(a, nm).weight.requires_grad_(True)
    if use_ckpt: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    grad_mag = np.zeros((L, H)); fisher = np.zeros((L, H)); per_batch = []; nb = 0
    model.train()
    for bi in range(min(n_batches, len(feats))):
        ids, labels = collate([feats[bi]], tok.pad_token_id)
        out = model(ids.cuda(), labels=labels.cuda()); model.zero_grad(set_to_none=True); out.loss.backward()
        pb = np.zeros((L, H))
        for l, a in enumerate(attn):
            if a is None: continue
            for nm in ["q_proj", "o_proj"]:
                g = getattr(a, nm).weight.grad
                if g is None: continue
                g = g.detach().float()
                gh = (g.view(H, hd, -1) if nm == "q_proj" else g.view(-1, H, hd).permute(1, 0, 2)).reshape(H, -1)
                grad_mag[l] += gh.abs().mean(dim=1).cpu().numpy(); fisher[l] += (gh * gh).sum(dim=1).cpu().numpy()
                pb[l] += gh.norm(dim=1).cpu().numpy()
        per_batch.append(pb); nb += 1
    model.zero_grad(set_to_none=True); model.eval()
    if use_ckpt: model.gradient_checkpointing_disable()
    grad_mag /= max(nb, 1); fisher /= max(nb, 1)
    pb = np.stack(per_batch, 0) if per_batch else np.zeros((1, L, H))
    gns = pb.var(axis=0) / (pb.mean(axis=0) ** 2 + 1e-9)
    for p in model.parameters(): p.requires_grad_(True)
    return grad_mag, fisher, gns

def _quick_sft_pairs(model, tok, pairs, steps, lr, use_ckpt=False, maxlen=512):
    for p in model.parameters(): p.requires_grad_(False)
    proj_set = ["q_proj", "o_proj"] if use_ckpt else ["q_proj", "k_proj", "v_proj", "o_proj"]
    tp = []
    for a in _attn_modules(model):
        if a is None: continue
        for nm in proj_set:
            w = getattr(a, nm).weight; w.requires_grad_(True); tp.append(w)
    if use_ckpt: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    feats = _pairs_to_feats(tok, pairs, maxlen)
    if not feats: return
    opt = torch.optim.SGD(tp, lr=lr * 10) if use_ckpt else torch.optim.AdamW(tp, lr=lr)
    model.train()
    for s in range(steps):
        b = feats[s % len(feats):s % len(feats) + 1]
        ids, labels = collate(b, tok.pad_token_id)
        out = model(ids.cuda(), labels=labels.cuda()); opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(tp, 1.0); opt.step()
    if use_ckpt: model.gradient_checkpointing_disable()

# ================================================================ H3 CAUSAL TEST
# Hypothesis: protecting the long-context-coupled heads during SFT preserves
# long-context (NIAH) + general (MMLU) better than no/random/post-hoc protection,
# at matched new-domain learning. Protection = freeze (grad-mask) the heads' q/o.
def _zscore(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s + 1e-9)

def _h3_select(model_name, kind, k, L, H):
    """Pick protected (layer,head) set. lc = top by combined LC-coupling z-score
    (retrieval+attn_distance+v_norm), detected on generic text (data-agnostic)."""
    gridf = os.path.join(ART, model_name, "grid_wikitext.npz")
    rng = np.random.RandomState(0)
    if kind == "none":
        return []
    if kind == "random" or not os.path.exists(gridf):
        flat = [(l, h) for l in range(L) for h in range(H)]
        idx = rng.choice(len(flat), size=min(k, len(flat)), replace=False)
        return [flat[i] for i in idx]
    z = np.load(gridf, allow_pickle=True)
    def g(key): return z["H_" + key] if "H_" + key in z.files else np.zeros((L, H))
    if kind == "lc":
        score = _zscore(g("retrieval")) + _zscore(g("attn_distance")) + _zscore(g("v_norm"))
    elif kind == "retrieval":
        score = g("retrieval")
    elif kind == "attn_distance":
        score = g("attn_distance")
    elif kind == "deltaw":
        score = g("dW_drift")  # post-hoc oracle ([mech-forget]-style): heads that DID drift
    else:
        score = g(kind)
    idx = np.dstack(np.unravel_index(np.argsort(score.ravel())[::-1][:k], score.shape))[0]
    return [tuple(map(int, x)) for x in idx]

@torch.no_grad()
def _niah_acc(model, tok, lengths=(2000, 6000), samples=6):
    ok = tot = 0
    for Ln in lengths:
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            for s in range(samples):
                code = f"{random.randint(10000,99999)}"; key = f"K{s}"
                body = filler_to_len(tok, Ln); ins = int(len(body) * d)
                ctx = body[:ins] + f" The special {key} value is {code}. " + body[ins:]
                prompt = ("Read the text and answer.\n" + ctx +
                          f"\nQuestion: What is the special {key} value? Answer with the number only:")
                ids = tok(prompt, return_tensors="pt", truncation=True, max_length=Ln + 200).input_ids.cuda()
                g = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.pad_token_id)
                ans = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
                tot += 1; ok += int(code in ans)
    return round(ok / max(tot, 1), 4)

@torch.no_grad()
def _gsm8k_acc(model, tok, n=80):
    from datasets import load_dataset
    try: ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception: return None
    idx = list(range(len(ds))); random.Random(7).shuffle(idx)
    ok = 0; tot = 0
    for i in idx[:n]:
        r = ds[i]; gold = r["answer"].split("####")[-1].strip().replace(",", "")
        ids = tok(f"Question: {r['question']}\nAnswer:", return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        g = model.generate(ids, max_new_tokens=180, do_sample=False, pad_token_id=tok.pad_token_id)
        out = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        nums = [c.replace(",", "") for c in __import__("re").findall(r"-?[\d,]+", out)]
        tot += 1; ok += int(bool(nums) and nums[-1] == gold)
    return round(ok / max(tot, 1), 4)

def cmd_h3(args):
    model_id = MODELS.get(args.model, args.model)
    log(f"H3 {args.model} protect={args.protect} k={args.protect_k} domain={args.domain} steps={args.steps}")
    model, tok = load_model(model_id, eager=False)
    cfg = model.config; L, H, hd = cfg.num_hidden_layers, cfg.num_attention_heads, head_dim_of(cfg)
    res = {"model": args.model, "protect": args.protect, "k": args.protect_k, "domain": args.domain, "steps": args.steps}
    # ---- before ----
    try: res["before_mmlu"] = eval_mmlu(model, tok, args.n_eval)
    except Exception as e: log("mmlu pre fail", e); res["before_mmlu"] = None
    try: res["before_niah"] = _niah_acc(model, tok)
    except Exception as e: log("niah pre fail", e); res["before_niah"] = None
    try: res["before_dom"] = _gsm8k_acc(model, tok, args.n_eval) if args.domain == "gsm8k" else None
    except Exception as e: res["before_dom"] = None
    prot = _h3_select(args.model, args.protect, args.protect_k, L, H)
    res["n_protected"] = len(prot)
    log(f"protecting {len(prot)} heads ({args.protect})")
    # ---- SFT (attn q/k/v/o proj only; mask protected q/o grads) ----
    pairs = load_grid_dataset(tok, args.domain, n=args.steps + 64)[1]
    feats = _pairs_to_feats(tok, pairs, 512)
    if not feats: log("no sft data; abort"); free(model); return
    for p in model.parameters(): p.requires_grad_(False)
    tp = []
    for a in _attn_modules(model):
        if a is None: continue
        for nm in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w = getattr(a, nm).weight; w.requires_grad_(True); tp.append(w)
    opt = torch.optim.AdamW(tp, lr=args.lr)
    model.train()
    for s in range(args.steps):
        b = feats[s % len(feats):s % len(feats) + 1]
        ids, labels = collate(b, tok.pad_token_id)
        out = model(ids.cuda(), labels=labels.cuda()); opt.zero_grad(); out.loss.backward()
        if prot: mask_head_grads(model, prot, hd)
        torch.nn.utils.clip_grad_norm_(tp, 1.0); opt.step()
        if s % 100 == 0: log(f"  h3sft {s}/{args.steps} loss {float(out.loss):.4f}")
    model.eval()
    # ---- after ----
    try: res["after_mmlu"] = eval_mmlu(model, tok, args.n_eval)
    except Exception as e: res["after_mmlu"] = None
    try: res["after_niah"] = _niah_acc(model, tok)
    except Exception as e: res["after_niah"] = None
    try: res["after_dom"] = _gsm8k_acc(model, tok, args.n_eval) if args.domain == "gsm8k" else None
    except Exception as e: res["after_dom"] = None
    # deltas
    for m in ["mmlu", "niah", "dom"]:
        b, a = res.get("before_" + m), res.get("after_" + m)
        res["d_" + m] = round(a - b, 4) if (a is not None and b is not None) else None
    json.dump(res, open(tag_path(args.model, f"h3_{args.protect}_{args.domain}.json"), "w"), indent=2)
    log(f"H3 done {args.protect}: dNIAH={res.get('d_niah')} dMMLU={res.get('d_mmlu')} dDOM={res.get('d_dom')}")
    free(model)

# ---------------------------------------------------------------- collate/util
def collate(batch, pad):
    m = max(len(x[0]) for x in batch)
    ids = []; labels = []
    for i, l in batch:
        ids.append(i + [pad] * (m - len(i)))
        labels.append(l + [-100] * (m - len(l)))
    return torch.tensor(ids), torch.tensor(labels)

def free(model):
    del model; gc.collect(); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect"); p.add_argument("--model", required=True)
    p = sub.add_parser("niah"); p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="1500,4000"); p.add_argument("--samples", type=int, default=6)
    p.add_argument("--ablate", action="store_true"); p.add_argument("--ablate_k", type=int, default=20)
    p.add_argument("--ablate_site", default="retr", choices=["retr", "sink", "induction"])
    p.add_argument("--ckpt", default="")
    p = sub.add_parser("sft"); p.add_argument("--model", required=True)
    p.add_argument("--domain", default="math"); p.add_argument("--mode", default="full")
    p.add_argument("--protect", default="none"); p.add_argument("--protect_k", type=int, default=20)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--maxlen", type=int, default=512)
    p.add_argument("--run_tag", default="")
    p = sub.add_parser("capeval"); p.add_argument("--model", required=True)
    p.add_argument("--ckpt", default=""); p.add_argument("--n", type=int, default=200)
    p = sub.add_parser("intrinsic"); p.add_argument("--model", required=True)
    p.add_argument("--domain", default="math"); p.add_argument("--sft_steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--fisher_batches", type=int, default=8)
    p = sub.add_parser("facts"); p.add_argument("--model", required=True)
    p = sub.add_parser("grid"); p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True); p.add_argument("--ctxlen", type=int, default=1024)
    p.add_argument("--n_texts", type=int, default=12); p.add_argument("--fisher_batches", type=int, default=8)
    p.add_argument("--sft_steps", type=int, default=80); p.add_argument("--lr", type=float, default=2e-5)
    p = sub.add_parser("h3"); p.add_argument("--model", required=True)
    p.add_argument("--protect", default="none"); p.add_argument("--protect_k", type=int, default=40)
    p.add_argument("--domain", default="gsm8k"); p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--n_eval", type=int, default=120)
    args = ap.parse_args()
    {"detect": cmd_detect, "niah": cmd_niah, "sft": cmd_sft, "capeval": cmd_capeval,
     "intrinsic": cmd_intrinsic, "facts": cmd_facts, "grid": cmd_grid, "h3": cmd_h3}[args.cmd](args)

if __name__ == "__main__":
    main()
