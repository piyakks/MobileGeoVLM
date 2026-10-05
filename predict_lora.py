#
# Single-image check for a LoRA-finetuned FastVLM: runs the same image + prompt through
# the base model and the LoRA model and prints both answers side by side.
#
# Usage:
# python predict_lora.py --lora_path ./checkpoints/llava-fastvithd_1.5b_stage3-geochat-lora \
#                          --model_base ./checkpoints/llava-fastvithd_1.5b_stage3 \
#                          --image_file 0024.png --prompt "Describe the image." [--no_base]
#
import os
import gc
import time
import argparse

import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load(model_path, model_base=None):
    model_name = get_model_name_from_path(model_path)
    if model_base is not None and "lora" not in model_name.lower():
        raise ValueError(f"builder.py only takes the LoRA path when 'lora' is in the folder name: {model_name}")
    # Force SDPA: the LoRA config saved by training makes HF pick eager attention, and Qwen2's large k_proj
    # bias overflows eager fp16 attention (NaN -> "!!!!" output). SDPA accumulates safely in fp16.
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name, device="cuda",
                                                                 attn_implementation="sdpa")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # The checkpoint's generation_config.json has no eos_token_id, so generate() would not stop at
    # <|im_end|> and keeps inventing extra Question/Answer turns. Stop at the chat end token.
    model.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return tokenizer, model, image_processor


def answer(tokenizer, model, image_processor, image, prompt, args):
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).to(model.device)
    image_tensor = process_images([image], image_processor, model.config)[0]

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(model.device, dtype=torch.float16),
            image_sizes=[image.size],
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=None,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text, time.time() - start


def run(label, model_path, model_base, image, args):
    print(f"\n[{label}] loading {model_path}" + (f" (+ base {model_base})" if model_base else ""))
    tokenizer, model, image_processor = load(model_path, model_base)
    text, sec = answer(tokenizer, model, image_processor, image, args.prompt, args)
    print(f"[{label}] ({sec:.2f}s)\n{text}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA output dir (adapter + non_lora_trainables.bin)")
    parser.add_argument("--model_base", type=str, required=True, help="Base checkpoint the LoRA was trained on")
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Describe the image.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (reproducible)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no_base", action="store_true", help="Only run the LoRA model")
    args = parser.parse_args()

    disable_torch_init()
    image = Image.open(args.image_file).convert('RGB')
    print(f"image: {args.image_file} {image.size}\nprompt: {args.prompt}")

    if not args.no_base:
        run("BASE", os.path.expanduser(args.model_base), None, image, args)
    run("LoRA", os.path.expanduser(args.lora_path), os.path.expanduser(args.model_base), image, args)
