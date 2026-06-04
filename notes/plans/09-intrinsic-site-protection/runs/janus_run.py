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
}
SIZE_B = {"qwen3-0.6b": 0.6, "qwen3-1.7b": 1.7, "qwen3-4b": 4.0, "qwen3-8b": 8.0,
          "qwen3-14b": 14.0, "qwen2.5-1.5b": 1.5, "qwen2.5-1.5b-instruct": 1.5,
          "qwen2.5-7b-instruct": 7.0}

def log(*a):
    print(f"[janus {time.strftime('%H:%M:%S')}]", *a, flush=True)

def tag_path(tag, name):
    d = os.path.join(ART, tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)

# ---------------------------------------------------------------- model utils
def load_model(model_id, eager=False, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=dtype)
    if eager:
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).cuda().eval()
    return model, tok

def head_dim_of(cfg):
    return getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

def layers_of(model):
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
        a = lyr.self_attn
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
    args = ap.parse_args()
    {"detect": cmd_detect, "niah": cmd_niah, "sft": cmd_sft, "capeval": cmd_capeval}[args.cmd](args)

if __name__ == "__main__":
    main()
