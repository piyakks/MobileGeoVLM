#
# Export a FastVLM (LLaVA-Qwen2) checkpoint to three ONNX graphs for on-device inference:
#   vision_encoder.onnx : pixel_values (1,3,S,S)          -> image_features (1,N,H)   [FastViTHD + mm_projector]
#   embed_tokens.onnx   : input_ids (1,T)                 -> inputs_embeds (1,T,H)
#   decoder.onnx        : inputs_embeds + mask + pos + KV -> logits of last token + present KV
#
# Usage:
#   python onnx_export/export_onnx.py --model-path checkpoints/llava-fastvithd_0.5b_stage3 --out onnx_export/fastvlm_0.5b
#
import os
import sys
import json
import shutil
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM


class VisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.get_model().get_vision_tower()
        self.mm_projector = model.get_model().mm_projector

    def forward(self, pixel_values):
        features = self.vision_tower(pixel_values)
        return self.mm_projector(features)


class EmbedTokens(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.get_model().embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class Decoder(nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model.get_model()
        self.lm_head = model.lm_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, *past):
        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.num_layers))
        cache = DynamicCache.from_legacy_cache(legacy)
        out = self.model(inputs_embeds=inputs_embeds,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=cache,
                         use_cache=True,
                         return_dict=True)
        # Only the last position is needed for generation
        logits = self.lm_head(out.last_hidden_state[:, -1:, :])
        present = []
        for k, v in out.past_key_values.to_legacy_cache():
            present += [k, v]
        return (logits, *present)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    model = LlavaQwen2ForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32,
                                                  attn_implementation="eager", low_cpu_mem_usage=True)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    model.eval()

    cfg = model.config
    image_size = vision_tower.input_image_size
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    # 1) Vision encoder + projector
    print("Exporting vision_encoder.onnx ...")
    pixel_values = torch.rand(1, 3, image_size, image_size)
    vision = VisionEncoder(model).eval()
    image_features = vision(pixel_values)
    torch.onnx.export(vision, (pixel_values,), os.path.join(args.out, "vision_encoder.onnx"),
                      input_names=["pixel_values"], output_names=["image_features"],
                      opset_version=args.opset, dynamo=False)
    num_image_tokens = image_features.shape[1]

    # 2) Token embeddings
    print("Exporting embed_tokens.onnx ...")
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    torch.onnx.export(EmbedTokens(model).eval(), (input_ids,), os.path.join(args.out, "embed_tokens.onnx"),
                      input_names=["input_ids"], output_names=["inputs_embeds"],
                      dynamic_axes={"input_ids": {1: "seq"}, "inputs_embeds": {1: "seq"}},
                      opset_version=args.opset, dynamo=False)

    # 3) Decoder with KV cache (trace with a non-empty past so the cache path is captured)
    print("Exporting decoder.onnx ...")
    past_len, seq_len = 4, 3
    inputs_embeds = torch.randn(1, seq_len, cfg.hidden_size)
    attention_mask = torch.ones(1, past_len + seq_len, dtype=torch.long)
    position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long).unsqueeze(0)
    past = []
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {"inputs_embeds": {1: "seq"},
                    "attention_mask": {1: "total_seq"},
                    "position_ids": {1: "seq"}}
    for i in range(num_layers):
        for kv in ("key", "value"):
            past.append(torch.randn(1, num_kv_heads, past_len, head_dim))
            input_names.append(f"past_key_values.{i}.{kv}")
            output_names.append(f"present.{i}.{kv}")
            dynamic_axes[f"past_key_values.{i}.{kv}"] = {2: "past_seq"}
            dynamic_axes[f"present.{i}.{kv}"] = {2: "total_seq"}

    decoder_dir = os.path.join(args.out, "decoder_tmp")
    os.makedirs(decoder_dir, exist_ok=True)
    torch.onnx.export(Decoder(model, num_layers).eval(),
                      (inputs_embeds, attention_mask, position_ids, *past),
                      os.path.join(decoder_dir, "decoder.onnx"),
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                      opset_version=args.opset, dynamo=False)

    # Re-save the >2GB decoder with a single external data file
    import onnx
    m = onnx.load(os.path.join(decoder_dir, "decoder.onnx"))
    onnx.save_model(m, os.path.join(args.out, "decoder.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="decoder.onnx.data")
    shutil.rmtree(decoder_dir)

    # Tokenizer + metadata the app needs
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.save_pretrained(args.out)
    meta = {
        "image_size": image_size,
        "num_image_tokens": num_image_tokens,
        "hidden_size": cfg.hidden_size,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "pad_color": 0,
        "rescale_factor": 1 / 255.0,
        "system_prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "assistant_prefix": "<|im_end|>\n<|im_start|>assistant\n",
    }
    with open(os.path.join(args.out, "fastvlm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done:", args.out, meta)


if __name__ == "__main__":
    main()
