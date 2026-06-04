from huggingface_hub import HfApi
api = HfApi()
cands = ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-8B",
         "THUDM/GLM-4-9B-0414", "zai-org/GLM-4-9B-0414", "THUDM/glm-4-9b",
         "THUDM/glm-4-9b-chat-hf", "zai-org/GLM-4-9B", "THUDM/GLM-4-32B-0414",
         "zai-org/GLM-4-32B-0414"]
for c in cands:
    try:
        info = api.model_info(c)
        sz = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors"))
        arch = info.config.get("architectures") if info.config else "?"
        print("OK   {:32s} ~{:.1f}GB  arch={}".format(c, sz / 1e9, arch))
    except Exception as e:
        print("MISS {:32s} {}".format(c, str(e)[:55]))
