#
# Reference ONNX Runtime pipeline for exported FastVLM — the Android app mirrors this logic.
# Optionally compares against the PyTorch checkpoint (greedy decoding).
#
# Usage:
#   python onnx_export/run_onnx.py --onnx-dir onnx_export/fastvlm_0.5b --image 00029.png \
#       --prompt "Describe the image." [--compare checkpoints/llava-fastvithd_0.5b_stage3]
#
import os
import sys
import json
import time
import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer


def preprocess(image, size, pad_color=0):
    """Pad to square, resize to (size, size) bicubic, scale to [0,1], CHW float32."""
    image = image.convert("RGB")
    w, h = image.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (pad_color,) * 3)
    square.paste(image, ((side - w) // 2, (side - h) // 2))
    square = square.resize((size, size), Image.BICUBIC)
    x = np.asarray(square, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


class FastVLMOnnx:
    def __init__(self, onnx_dir, vision="vision_encoder.onnx", embed="embed_tokens.onnx", decoder="decoder.onnx"):
        with open(os.path.join(onnx_dir, "fastvlm_meta.json")) as f:
            self.meta = json.load(f)
        opts = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        self.vision = ort.InferenceSession(os.path.join(onnx_dir, vision), opts, providers=providers)
        self.embed = ort.InferenceSession(os.path.join(onnx_dir, embed), opts, providers=providers)
        self.decoder = ort.InferenceSession(os.path.join(onnx_dir, decoder), opts, providers=providers)
        self.tokenizer = Tokenizer.from_file(os.path.join(onnx_dir, "tokenizer.json"))

    def tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def generate(self, image, prompt, max_new_tokens=128):
        m = self.meta
        t0 = time.time()
        pixel_values = preprocess(image, m["image_size"], m["pad_color"])
        image_features = self.vision.run(None, {"pixel_values": pixel_values})[0]
        t_vision = time.time() - t0

        # <system><user_prefix> [image] \n<prompt><assistant_prefix>
        pre_ids = self.tokens(m["system_prompt"] + m["user_prefix"])
        post_ids = self.tokens("\n" + prompt + m["assistant_prefix"])
        embed = lambda ids: self.embed.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]
        inputs_embeds = np.concatenate([embed(pre_ids), image_features.astype(np.float32), embed(post_ids)], axis=1)

        L, kvh, hd = m["num_layers"], m["num_kv_heads"], m["head_dim"]
        past = {f"past_key_values.{i}.{kv}": np.zeros((1, kvh, 0, hd), dtype=np.float32)
                for i in range(L) for kv in ("key", "value")}
        seq = inputs_embeds.shape[1]
        total = seq
        position_ids = np.arange(seq, dtype=np.int64)[None]

        out_ids = []
        t_first = None
        for step in range(max_new_tokens):
            feeds = {"inputs_embeds": inputs_embeds,
                     "attention_mask": np.ones((1, total), dtype=np.int64),
                     "position_ids": position_ids, **past}
            outputs = self.decoder.run(None, feeds)
            if t_first is None:
                t_first = time.time() - t0
            next_id = int(np.argmax(outputs[0][0, -1]))
            if next_id == m["eos_token_id"]:
                break
            out_ids.append(next_id)
            for j, name in enumerate(past):
                past[name] = outputs[1 + j]
            inputs_embeds = embed([next_id])
            position_ids = np.array([[total]], dtype=np.int64)
            total += 1

        text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        return text, {"vision_s": round(t_vision, 2), "ttft_s": round(t_first, 2),
                      "total_s": round(time.time() - t0, 2), "new_tokens": len(out_ids)}


def run_torch(model_path, image, prompt, max_new_tokens):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llava.conversation import conv_templates
    from llava.model.language_model.llava_qwen import LlavaQwen2ForCausalLM
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaQwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    vt = model.get_vision_tower()
    if not vt.is_loaded:
        vt.load_model()
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    images = process_images([image.convert("RGB")], vt.image_processor, model.config)
    with torch.inference_mode():
        out = model.generate(input_ids, images=images, image_sizes=[image.size], do_sample=False,
                             num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                             pad_token_id=tokenizer.pad_token_id)
    return full_prompt, tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--vision", default="vision_encoder.onnx")
    parser.add_argument("--embed", default="embed_tokens.onnx")
    parser.add_argument("--decoder", default="decoder.onnx")
    parser.add_argument("--compare", default=None, help="PyTorch checkpoint dir to compare against")
    args = parser.parse_args()

    image = Image.open(args.image)
    vlm = FastVLMOnnx(args.onnx_dir, args.vision, args.embed, args.decoder)
    text, stats = vlm.generate(image, args.prompt, args.max_new_tokens)
    print("[ONNX]", text)
    print("      ", stats)

    if args.compare:
        full_prompt, ref = run_torch(args.compare, image, args.prompt, args.max_new_tokens)
        print("[Torch]", ref)
        print("match:", text == ref)
        print("prompt template:", repr(full_prompt))
