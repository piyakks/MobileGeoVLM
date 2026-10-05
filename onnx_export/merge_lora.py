#
# Merge a LoRA run (adapter + non_lora_trainables.bin) into its base FastVLM checkpoint and save a
# standalone fp32 checkpoint that export_onnx.py can read.
#
# Usage:
#   python onnx_export/merge_lora.py --lora-path checkpoints/llava-fastvithd_1.5b_stage3-geochat-lora \
#       --model-base checkpoints/llava-fastvithd_1.5b_stage3 --out checkpoints/fastvlm-1.5b-geochat-merged
#
import os
import sys
import shutil
import argparse

import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Same steps as llava/model/builder.py's LoRA branch, but on CPU in fp32 for an accurate export
    cfg = AutoConfig.from_pretrained(args.lora_path)
    model = LlavaQwen2ForCausalLM.from_pretrained(args.model_base, config=cfg, torch_dtype=torch.float32,
                                                  low_cpu_mem_usage=True)

    non_lora = torch.load(os.path.join(args.lora_path, "non_lora_trainables.bin"), map_location="cpu")
    non_lora = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora.items()}
    if any(k.startswith("model.model.") for k in non_lora):
        non_lora = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora.items()}
    missing = [k for k in non_lora if k not in model.state_dict()]
    if missing:
        raise RuntimeError(f"non_lora_trainables keys not in model: {missing[:5]}")
    model.load_state_dict({k: v.float() for k, v in non_lora.items()}, strict=False)
    print(f"loaded {len(non_lora)} non-LoRA tensors: {sorted({k.rsplit('.', 2)[0] for k in non_lora})}")

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()

    # Inference-friendly config: stop at <|im_end|> and don't force eager attention
    model.config.use_cache = True
    for k in ("_attn_implementation_autoset",):
        model.config.__dict__.pop(k, None)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    model.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    for f in ("added_tokens.json", "special_tokens_map.json", "merges.txt", "vocab.json"):
        src = os.path.join(args.model_base, f)
        if os.path.exists(src) and not os.path.exists(os.path.join(args.out, f)):
            shutil.copy(src, args.out)
    print("saved merged model to", args.out)


if __name__ == "__main__":
    main()
