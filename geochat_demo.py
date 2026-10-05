#
# GeoChat-style Gradio demo for FastVLM (LLaVA: FastViTHD + Qwen2).
# Ported from GeoChat/geochat_demo.py to the llava package in this repo and Gradio 6.
#
# Usage:
#   python geochat_demo.py --model-path ./checkpoints/llava-fastvithd_1.5b_stage3-geochat-lora \
#                          --model-base ./checkpoints/llava-fastvithd_1.5b_stage3
#
import argparse
import copy
import hashlib
import html
import random
import re
from threading import Thread

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser(description="FastVLM GeoChat Demo")
    parser.add_argument("--model-path", type=str, default="./checkpoints/llava-fastvithd_1.5b_stage3-geochat-lora")
    parser.add_argument("--model-base", type=str, default="./checkpoints/llava-fastvithd_1.5b_stage3")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

args = parse_args()
device = f"cuda:{args.gpu_id}"

print("Initializing FastVLM")
disable_torch_init()
model_name = get_model_name_from_path(args.model_path)
# SDPA: the LoRA config saved by training makes HF pick eager attention, which overflows to NaN in fp16
# on Qwen2 (large k_proj bias) and prints "!!!!".
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device,
    attn_implementation="sdpa")
model = model.eval()
# The released checkpoints ship do_sample=True in generation_config.json; sampling is controlled from the UI instead.
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.pad_token_id = tokenizer.pad_token_id
# That generation_config.json also has no eos_token_id, so generate() would run past <|im_end|> and
# keep inventing extra Question/Answer turns. Stop at the chat end token.
model.generation_config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

BOX_SCALE = 100  # GeoChat boxes are normalized to [0, 100]
BOX_PATTERN = re.compile(r"\{<(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)>(?:\|<(-?\d+)>)?\}")
TOKEN_PATTERN = re.compile(r"<p>(.*?)</p>|" + BOX_PATTERN.pattern)

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (210, 210, 0), (255, 0, 255), (0, 255, 255),
    (250, 128, 114), (255, 165, 0), (0, 128, 0), (144, 238, 144), (175, 238, 238), (0, 191, 255),
    (138, 43, 226), (255, 215, 0),
]


# ----------------------------------------------------------------------------- boxes

def parse_boxes(text):
    """Return [(label, x1, y1, x2, y2, angle)] in GeoChat's 0-100 coordinates.

    Each box takes the label of the closest preceding <p>...</p> phrase (empty for [refer] answers).
    """
    boxes, label = [], ""
    for m in TOKEN_PATTERN.finditer(text):
        if m.group(1) is not None:
            label = m.group(1).strip()
        else:
            x1, y1, x2, y2 = (int(v) for v in m.group(2, 3, 4, 5))
            angle = int(m.group(6)) if m.group(6) is not None else 0
            boxes.append((label, x1, y1, x2, y2, angle))
    return boxes


def rotated_polygon(x1, y1, x2, y2, angle):
    # Same convention as GeoChat's rotate_bbox (cv2 rotation around the box center).
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    rot = cv2.getRotationMatrix2D(center, angle, 1)
    rect = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    return cv2.transform(rect, rot)[0].astype(np.int32)


def draw_boxes(image, boxes):
    if image is None or not boxes:
        return None
    img = image.convert("RGB")
    scale = 800 / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    canvas = np.array(img)
    w, h = img.size
    thickness = max(2, round(max(w, h) / 300))
    font_scale = max(0.4, max(w, h) / 1200)

    label_colors = {}
    for label, x1, y1, x2, y2, angle in boxes:
        color = label_colors.setdefault(label, COLORS[len(label_colors) % len(COLORS)])
        poly = rotated_polygon(x1 / BOX_SCALE * w, y1 / BOX_SCALE * h,
                               x2 / BOX_SCALE * w, y2 / BOX_SCALE * h, angle)
        cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=thickness)
        if label:
            tx, ty = int(poly[:, 0].min()), int(poly[:, 1].min())
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            ty = max(ty, th + 4)
            cv2.rectangle(canvas, (tx, ty - th - 4), (tx + tw + 4, ty), color, -1)
            cv2.putText(canvas, label, (tx + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)
    return Image.fromarray(canvas)


def mask_to_box(editor_value):
    """Bounding box of the strokes drawn on the image editor, as a GeoChat box string."""
    background = editor_value.get("background")
    layers = [l for l in (editor_value.get("layers") or []) if l is not None]
    if background is None or not layers:
        return ""
    alpha = np.zeros((background.height, background.width), dtype=bool)
    for layer in layers:
        layer = layer.convert("RGBA").resize(background.size, Image.NEAREST)
        alpha |= np.array(layer)[:, :, 3] > 0
    if not alpha.any():
        return ""
    rows, cols = np.where(alpha.any(axis=1))[0], np.where(alpha.any(axis=0))[0]
    x1, x2 = cols[0] * BOX_SCALE // background.width, cols[-1] * BOX_SCALE // background.width
    y1, y2 = rows[0] * BOX_SCALE // background.height, rows[-1] * BOX_SCALE // background.height
    return f"{{<{x1}><{y1}><{x2}><{y2}>|<0>}}"


def colorize(text):
    """Escape the raw answer for the chatbot and color each <p>phrase</p>."""
    label_colors = {}

    def repl(m):
        label = m.group(1).strip()
        r, g, b = label_colors.setdefault(label, COLORS[len(label_colors) % len(COLORS)])
        return f'<b style="color:rgb({r},{g},{b})">{html.escape(m.group(1))}</b>'

    parts, last = [], 0
    for m in re.finditer(r"<p>(.*?)</p>", text):
        parts.append(html.escape(text[last:m.start()]))
        parts.append(repl(m))
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


# ----------------------------------------------------------------------------- chat

def new_state():
    return {"conv": conv_templates[args.conv_mode].copy(), "image_hash": None,
            "image_tensor": None, "image_size": None}


def image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()


def gradio_ask(user_message, chatbot, state, editor_value):
    if not user_message.strip():
        raise gr.Error("Input should not be empty!")
    image = (editor_value or {}).get("background")
    if image is None:
        raise gr.Error("Upload an image first.")
    image = image.convert("RGB")

    # A new image starts a new conversation (same behaviour as GeoChat's replace_flag).
    h = image_hash(image)
    if state is None or state["image_hash"] != h:
        state = new_state()
        state["image_hash"] = h
        state["image_tensor"] = process_images([image], image_processor, model.config)[0]
        state["image_size"] = image.size
        chatbot = []

    if "[identify]" in user_message and not BOX_PATTERN.search(user_message):
        box = mask_to_box(editor_value)
        if not box:
            raise gr.Error("[identify] needs a box: draw on the image or type {<x1><y1><x2><y2>|<angle>}.")
        user_message = f"{user_message.rstrip()} {box}"

    conv = state["conv"]
    if len(conv.messages) == 0:
        image_token = DEFAULT_IMAGE_TOKEN
        if model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conv.append_message(conv.roles[0], image_token + "\n" + user_message)
    else:
        conv.append_message(conv.roles[0], user_message)
    conv.append_message(conv.roles[1], None)

    chatbot = chatbot + [{"role": "user", "content": html.escape(user_message)}]
    if "[identify]" in user_message:
        visual = draw_boxes(image, parse_boxes(user_message))
        if visual is not None:
            chatbot.append({"role": "user", "content": gr.Image(value=visual)})
    chatbot.append({"role": "assistant", "content": ""})
    return "", chatbot, state


def gradio_stream_answer(chatbot, state, temperature):
    conv = state["conv"]
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    images = state["image_tensor"].unsqueeze(0).to(device=device, dtype=torch.float16)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)
    gen_kwargs = dict(inputs=input_ids, images=images, image_sizes=[state["image_size"]],
                      max_new_tokens=args.max_new_tokens, streamer=streamer, use_cache=True)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    def generate():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    Thread(target=generate).start()

    output = ""
    for new_text in streamer:
        output += new_text
        chatbot[-1]["content"] = html.escape(output)
        yield chatbot, state

    output = output.strip()
    print(output)
    conv.messages[-1][1] = output
    chatbot[-1]["content"] = colorize(output)
    yield chatbot, state


def gradio_visualize(chatbot, state, editor_value):
    image = (editor_value or {}).get("background")
    answer = state["conv"].messages[-1][1] or ""
    visual = draw_boxes(image, parse_boxes(answer))
    if visual is not None:
        chatbot = chatbot + [{"role": "assistant", "content": gr.Image(value=visual)}]
    return chatbot


def gradio_reset():
    return [], None, None, gr.update(value="", placeholder="Upload your image and chat")


TASKS = {
    "No Tag": ("", "**Hint:** Type in whatever you want"),
    "Scene Classification": ("Classify the image in the following classes: ",
                             "**Hint:** Type in the classes you want the model to classify in"),
    "Identify": ("[identify] what is this ",
                 "**Hint:** Draw on the object with the brush, then send. Clear the drawing before redrawing."),
    "Refer": ("[refer] where is <p></p> ?", "**Hint:** Put the object description inside <p></p>"),
    "Grounding": ("[grounding] describe this image in detail",
                  "**Hint:** The model describes the image and boxes every object it mentions"),
}


def gradio_taskselect(task):
    return TASKS[task]


# ----------------------------------------------------------------------------- UI

title = """<h1 align="center">FastVLM × GeoChat Demo</h1>"""
introduction = f"""
Model: `{args.model_path}`

1. **Identify**: draw on the object with the brush and click **Send**, or type a box `{{<x1><y1><x2><y2>|<angle>}}` (0-100).
2. **Refer**: `[refer] where is <p>object</p> ?` → the predicted box is drawn on the image.
3. **Grounding**: `[grounding] describe this image in detail` → every box in the answer is drawn.
4. **No Tag**: chat freely. Uploading a new image starts a new conversation.
"""

with gr.Blocks(title="FastVLM GeoChat Demo") as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.ImageEditor(
                type="pil", label="Image", sources=("upload", "clipboard"), transforms=(), layers=False,
                brush=gr.Brush(default_size=20, colors=["#ff0000"], color_mode="fixed"),
            )
            temperature = gr.Slider(minimum=0.0, maximum=1.5, value=0.0, step=0.1, interactive=True,
                                    label="Temperature (0 = greedy)")
            clear = gr.Button("Restart")
            gr.Markdown(introduction)

        with gr.Column(scale=2):
            state = gr.State(value=None)
            chatbot = gr.Chatbot(label="FastVLM", height=600, sanitize_html=False)
            task = gr.Radio(choices=list(TASKS), value="No Tag", label="Task Shortcuts")
            task_inst = gr.Markdown(TASKS["No Tag"][1])
            with gr.Row():
                text_input = gr.Textbox(placeholder="Upload your image and chat", show_label=False,
                                        container=False, scale=12)
                send = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Examples(examples=[
                ["demo_images/train_2956_0001.png", "Where are the airplanes located and what is their type?"],
                ["demo_images/7292.JPG", "How many buildings are flooded?"],
            ], inputs=[image, text_input])
        with gr.Column():
            gr.Examples(examples=[
                ["demo_images/church_183.png",
                 "Classify the image in the following classes: Church, Beach, Dense Residential, Storage Tanks."],
                ["demo_images/04444.png", "[identify] what is this {<8><26><22><37>|<0>}"],
                ["demo_images/train_2956_0001.png", "[grounding] describe this image in detail"],
            ], inputs=[image, text_input])

    task.change(gradio_taskselect, [task], [text_input, task_inst], queue=False)

    for trigger in (text_input.submit, send.click):
        trigger(
            gradio_ask, [text_input, chatbot, state, image], [text_input, chatbot, state], queue=False,
        ).success(
            gradio_stream_answer, [chatbot, state, temperature], [chatbot, state],
        ).success(
            gradio_visualize, [chatbot, state, image], [chatbot], queue=False,
        )

    clear.click(gradio_reset, None, [chatbot, image, state, text_input], queue=False)

demo.queue().launch(server_name=args.server_name, server_port=args.port, share=args.share)
