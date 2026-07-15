                def _fulldoc_bm25(it, submode):   # FAIR-vs-RAG fix: chunk/BM25 family needs no forward -> retrieve over the FULL (untruncated) doc, like RAG. Fixes the ctx>MAXCTX truncation asymmetry (F44).
                    import math as _m
                    from collections import Counter as _C
                    toks = rt.tok("".join(map(str, getattr(it, "chunks", []) or [])), add_special_tokens=False).input_ids
                    L = len(toks)
                    if L == 0:
                        return embed(rt.query_ids(it))
                    W = 256 if submode in ("chunk", "hier", "auto") else 32
                    budget = max(8, int(LL_RATE * min(L, MAXCTX)))
                    qset = set(rt.query_ids(it)[0].tolist())
                    units = [(i, min(i + W, L)) for i in range(0, L, W)]
                    segs = [toks[a:b] for a, b in units]; N = len(segs); df = {}
                    for sg in segs:
                        for t in set(sg): df[t] = df.get(t, 0) + 1
                    avgdl = max(1.0, L / max(1, N)); k1, bb = 1.5, 0.75
                    def _sc(sg):
                        tf = _C(sg); v = 0.0
                        for t in qset:
                            if t in tf:
                                idf = _m.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                                v += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - bb + bb * len(sg) / avgdl))
                        return v
                    ranked = sorted(range(N), key=lambda i: _sc(segs[i]), reverse=True)
                    sel, tot = [], 0
                    for i in ranked:
                        if tot + len(segs[i]) > budget and sel: break
                        sel.append(i); tot += len(segs[i])
                    sel.sort()
                    keep_ids = [t for i in sel for t in segs[i]][:MAXCTX]
                    return embed(torch.tensor([keep_ids], device=dev))
                def _imp(it):
                    ids = rt.tok("".join(map(str, getattr(it, "chunks", []) or [])), add_special_tokens=False,
                                 truncation=True, max_length=MAXCTX, return_tensors="pt").input_ids.to(dev)
                    if ids.shape[1] == 0:
                        return embed(rt.query_ids(it))
                    _FD = env("GCM_IMP_FULLDOC", "1") == "1"
                    _M0 = env("GCM_IMP_MODE", "span")
                    _doc_too_long = ids.shape[1] >= MAXCTX   # doc does not fit -> span (needs a forward) cannot see it all -> must retrieve over full doc
                    _go_fd = _M0 in ("chunk", "bm25span", "hier") or (_M0 == "auto" and (_doc_too_long or not bool(options_for(it))))
                    if _FD and _go_fd:
                        return _fulldoc_bm25(it, _M0)
                    L = ids.shape[1]
                    with torch.no_grad():
                        E = embed(ids)[0].float()
                        qe = embed(rt.query_ids(it))[0].float().mean(0, keepdim=True)
                        En = E / (E.norm(dim=-1, keepdim=True) + 1e-6); qn = qe / (qe.norm() + 1e-6)
                        qdot = (En @ qn.T).squeeze(-1)                       # query relevance (F20: 0.95 word-needle)
                        lg = base(inputs_embeds=embed(ids), use_cache=False).logits[0].float()
                        lp = lg.log_softmax(-1); surp = torch.zeros(L, device=dev)
                        surp[1:] = -lp[:-1].gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)   # surprisal (F20: 0.84 numeric)
                        z = lambda t: (t - t.mean()) / (t.std() + 1e-6)
                        _sig = env("GCM_IMP_SIGNAL", "both")                 # query/surprisal/both/lex/qlex/all
                        import math as _m
                        _qset = set(rt.query_ids(it)[0].tolist())
                        _tf = {}
                        for _t in ids[0].tolist(): _tf[_t] = _tf.get(_t, 0) + 1
                        lex = torch.tensor([(_m.log((L + 1) / (_tf[_t] + 0.5)) if _t in _qset else 0.0) for _t in ids[0].tolist()], device=dev, dtype=torch.float)  # IDF-weighted query-term match (BM25-style; downweights boilerplate)
                        if _sig == "query": score = z(qdot)
                        elif _sig == "surprisal": score = z(surp)
                        elif _sig == "lex": score = z(lex)
                        elif _sig == "qlex": score = z(qdot) + z(lex)
                        elif _sig == "all": score = z(qdot) + z(surp) + z(lex)
                        else: score = z(qdot) + z(surp)
                    keep = max(8, int(LL_RATE * L))
                    MODE = env("GCM_IMP_MODE", "span")
                    W = max(1, int(env("GCM_IMP_SPAN", "32")))
                    CW = int(env("GCM_IMP_CHUNK", "256"))
                    toks = ids[0].tolist()
                    def _spanmax(sc, w):
                        nb = (L + w - 1) // w
                        pad = torch.full((nb * w - L,), float("-inf"), device=dev)
                        return torch.cat([sc, pad]).view(nb, w).max(1).values
                    def _keep_spans(sc, w, budget):
                        bs = _spanmax(sc, w)
                        blocks = bs.topk(min(max(1, budget // w), (L + w - 1) // w)).indices.sort().values
                        return torch.cat([torch.arange(int(b) * w, min(int(b) * w + w, L), device=dev) for b in blocks])
                    def _units(w):
                        return [(i, min(i + w, L)) for i in range(0, L, w)]
                    def _bm25(units):
                        k1, bb = 1.5, 0.75; udf = {}
                        for a, b_ in units:
                            for t in set(toks[a:b_]): udf[t] = udf.get(t, 0) + 1
                        Nu = len(units); avgdl = max(1.0, L / Nu); out = []
                        for a, b_ in units:
                            seg = toks[a:b_]; tfc = {}
                            for t in seg: tfc[t] = tfc.get(t, 0) + 1
                            sm = 0.0
                            for t in _qset:
                                if t in tfc:
                                    idf = _m.log(1 + (Nu - udf.get(t, 0) + 0.5) / (udf.get(t, 0) + 0.5))
                                    sm += idf * tfc[t] * (k1 + 1) / (tfc[t] + k1 * (1 - bb + bb * len(seg) / avgdl))
                            out.append(sm)
                        return torch.tensor(out, device=dev, dtype=torch.float)
                    def _sel_units(us, units, budget):
                        order = us.argsort(descending=True); sel = []; tot = 0
                        for j in order.tolist():
                            a, b_ = units[j]
                            if tot + (b_ - a) > budget and sel: continue
                            sel.append(j); tot += (b_ - a)
                            if tot >= budget: break
                        sel.sort(); return [i for j in sel for i in range(units[j][0], units[j][1])]
                    if MODE == "chunk":
                        units = _units(CW); bm = _bm25(units)
                        ms = torch.stack([surp[a:b_].mean() for a, b_ in units])
                        mq = torch.stack([qdot[a:b_].mean() for a, b_ in units])
                        idx = torch.tensor(_sel_units(z(bm) + z(ms) + z(mq), units, keep), device=dev, dtype=torch.long)
                    elif MODE == "bm25span":
                        units = _units(W); bm = _bm25(units)
                        ms = torch.stack([surp[a:b_].mean() for a, b_ in units])
                        idx = torch.tensor(_sel_units(z(bm) + z(ms), units, keep), device=dev, dtype=torch.long)
                    elif MODE == "hier":
                        units = _units(CW); bm = _bm25(units)
                        toks1 = _sel_units(bm, units, min(2 * keep, L))
                        sub = torch.tensor(sorted(toks1), device=dev, dtype=torch.long)
                        subsc = score[sub]; ns = len(sub); nb = (ns + W - 1) // W
                        pad = torch.full((nb * W - ns,), float("-inf"), device=dev)
                        bs = torch.cat([subsc, pad]).view(nb, W).max(1).values
                        blocks = bs.topk(min(max(1, keep // W), nb)).indices.sort().values
                        idx = torch.cat([sub[int(b) * W:min(int(b) * W + W, ns)] for b in blocks])
                    elif MODE == "qfree":
                        idx = _keep_spans(z(surp), W, keep) if W > 1 else z(surp).topk(min(keep, L)).indices
                    elif MODE == "auto":  # input-driven route chunk<->span; routers {peak,mc,qover}; optional adaptive budget (F39)
                        _router = env("GCM_IMP_AUTO_ROUTER", "mc")
                        units = _units(CW); bm = _bm25(units)
                        peak = float(bm.max() / (bm.mean() + 1e-6)) if bm.numel() > 1 else 0.0
                        if _router == "mc":
                            go_chunk = not bool(options_for(it))            # options present (MC/reasoning) -> span; extractive -> chunk
                        elif _router == "qover":
                            ov = len(_qset & set(toks)) / max(1, len(_qset))
                            go_chunk = ov >= float(env("GCM_IMP_AUTO_TAU", "0.5"))
                        else:  # peak (BM25 max/mean)
                            go_chunk = peak >= float(env("GCM_IMP_AUTO_TAU", "3.0"))
                        kk = keep
                        if env("GCM_IMP_ADAPT_BUDGET", "0") == "1" and go_chunk:
                            kk = max(8, int(0.25 * L))                      # tighten budget when a lexical anchor localizes the answer (targets F39)
                        if go_chunk:
                            ms = torch.stack([surp[a:b_].mean() for a, b_ in units])
                            mq = torch.stack([qdot[a:b_].mean() for a, b_ in units])
                            idx = torch.tensor(_sel_units(z(bm) + z(ms) + z(mq), units, kk), device=dev, dtype=torch.long)
                        else:
                            idx = _keep_spans(score, W, kk) if W > 1 else score.topk(min(kk, L)).indices
                    else:  # span (default): token score -> top-p spans
                        idx = _keep_spans(score, W, keep) if W > 1 else score.topk(min(keep, L)).indices
                    if idx.numel() == 0:
                        idx = torch.arange(min(keep, L), device=dev)
                    idx = idx.sort().values
                    return embed(ids[:, idx])
                mpref = [_imp(it) for it in its]
