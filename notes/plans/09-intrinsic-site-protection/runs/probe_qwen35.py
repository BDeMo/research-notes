import sys, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
mid = "Qwen/Qwen3.5-4B"
print("=== config ===")
cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
print("type:", cfg.model_type, "| arch:", getattr(cfg, "architectures", None))
tc = getattr(cfg, "text_config", cfg)
for k in ["num_hidden_layers", "num_attention_heads", "num_key_value_heads",
          "hidden_size", "head_dim", "full_attention_interval", "layer_types"]:
    if hasattr(tc, k): print(" ", k, "=", getattr(tc, k))
print("=== load (text) ===")
try:
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="eager").cuda().eval()
    print("loaded:", type(m).__name__)
    base = m.model if hasattr(m, "model") else m
    layers = base.layers if hasattr(base, "layers") else base.language_model.layers
    print("n layers:", len(layers))
    for i in [0, 1, 2, 3]:
        l = layers[i]
        attn = getattr(l, "self_attn", None)
        names = [n for n, _ in l.named_children()]
        sub = [n for n, _ in attn.named_children()] if attn is not None else None
        print(f"  layer{i}: children={names} | self_attn_sub={sub}")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = m(ids, output_attentions=True)
    att = out.attentions
    print("attentions len:", None if att is None else len(att))
    if att is not None:
        present = [i for i in range(len(att)) if att[i] is not None]
        print("layers WITH softmax attn:", present[:20], "...total", len(present))
except Exception as e:
    import traceback; traceback.print_exc()
    print("LOAD FAILED:", str(e)[:200])
