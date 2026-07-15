from mem_embedding.gcm.data import load_items
HEAD = ["longbench_v2","infbench_choice","lb_2wikimqa","lb_hotpotqa","lb_musique",
        "lb_multifieldqa","lb_qasper","lb_narrativeqa","quality","quality_hard","musr_mm"]
SHORT = ["squad_v2","hotpot_qa","trivia_qa","ms_marco"]
for b in HEAD:
    try:
        n = len(load_items(b, 100000, 8, 1, "validation"))
        print("FULL  %-18s N=%d" % (b, n))
    except Exception as e:
        print("ERR   %-18s %s" % (b, str(e)[:80]))
for b in SHORT:
    try:
        nfull = len(load_items(b, 100000, 8, 1, "validation"))
        print("SHORT %-18s full=%d  (grid ran subset N=500)" % (b, nfull))
    except Exception as e:
        print("ERR   %-18s %s" % (b, str(e)[:80]))
