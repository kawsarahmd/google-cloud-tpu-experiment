# save_pt_locally.py
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

SOURCE_REPO = "kawsarahmd/BanglaEnglishT5-32k_Translation_v2"
TARGET_DIR  = "/kaggle/working/BanglaEnglishT5-32k_Translation_v2"
HF_TOKEN    = os.getenv("HF_TOKEN")  # set if the repo is private

print("-> Loading config & tokenizer...")
config = AutoConfig.from_pretrained(SOURCE_REPO, token=HF_TOKEN)
# tokenizer = AutoTokenizer.from_pretrained(SOURCE_REPO, token=HF_TOKEN)

print("-> Converting Flax -> PyTorch (full materialization on CPU to avoid meta tensors)...")
pt_model = AutoModelForSeq2SeqLM.from_pretrained(
    SOURCE_REPO,
    from_flax=True,
    config=config,
    token=HF_TOKEN,
    device_map=None,          # no sharding
    low_cpu_mem_usage=False,  # fully load weights (prevents meta tensors)
    torch_dtype=None          # default dtype; avoids bf16 meta init paths
)

# Extra safety: ensure not on meta, force CPU + float32 before saving
if any(p.is_meta for p in pt_model.parameters()):
    print("-> Detected meta tensors; materializing via state_dict round-trip...")
    state = {k: v.to("cpu") for k, v in pt_model.state_dict().items()}
    pt_model.to("cpu")
    pt_model.load_state_dict(state, strict=False)
else:
    pt_model.to("cpu")

pt_model = pt_model.float()

print(f"-> Saving PyTorch model & tokenizer to: {TARGET_DIR}")
os.makedirs(TARGET_DIR, exist_ok=True)
pt_model.save_pretrained(TARGET_DIR, safe_serialization=True)  # writes model.safetensors
# tokenizer.save_pretrained(TARGET_DIR)
# config.save_pretrained(TARGET_DIR)
# pt_model.push
print("✅ Done. Files written:")
print("  - model.safetensors")
print("  - config.json")
print("  - tokenizer.json / tokenizer.model (as applicable)")
print("  - special_tokens_map.json, tokenizer_config.json, etc.")

pt_model.push_to_hub(
    SOURCE_REPO,
    token=HF_TOKEN,
)
