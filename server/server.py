#
# FastVLM inference server (FastAPI).
# Loads the model once and answers image + prompt requests over HTTP.
#
# Usage:
#   python server/server.py --model-path checkpoints/llava-fastvithd_0.5b_stage3 --port 8000
#
import io
import os
import sys
import time
import argparse
import threading

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class FastVLM:
    def __init__(self, model_path, model_base=None, conv_mode="qwen_2", load_4bit=False, load_8bit=False):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, model_name,
            load_8bit=load_8bit, load_4bit=load_4bit, device="cuda")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        # The checkpoint's generation_config.json has no eos_token_id, so generate() would not stop at
        # <|im_end|> and keeps inventing extra Question/Answer turns. Stop at the chat end token.
        self.model.generation_config.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.conv_mode = conv_mode
        self.model_name = model_name
        # GPU is shared, so run one generation at a time
        self.lock = threading.Lock()

    def build_prompt(self, question):
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def predict(self, image, question, temperature=0.2, top_p=None, max_new_tokens=256):
        prompt = self.build_prompt(question)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).to(self.model.device)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(self.model.device, dtype=torch.float16)

        with self.lock, torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


app = FastAPI(title="FastVLM Server")
vlm: FastVLM = None
MAX_NEW_TOKENS = 128

# Minimal mobile web UI so a phone browser can use the server without the Android app
INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FastVLM</title>
<style>
  body { font-family: sans-serif; max-width: 640px; margin: 0 auto; padding: 16px; }
  video, img { width: 100%; max-height: 420px; object-fit: contain; background: #000; margin-top: 8px; border-radius: 6px; }
  input[type=text], button, label.btn { width: 100%; box-sizing: border-box; font-size: 16px; padding: 12px; margin-top: 8px; }
  label.btn { display: block; text-align: center; background: #444; color: #fff; border-radius: 6px; }
  button { background: #0a7; color: #fff; border: none; border-radius: 6px; }
  button.stop { background: #c33; }
  button:disabled { background: #999; }
  .row { display: flex; gap: 8px; }
  .row > * { flex: 1; }
  .tab { background: #ddd; color: #000; }
  .tab.on { background: #333; color: #fff; }
  #answer { white-space: pre-wrap; margin-top: 12px; font-size: 18px; min-height: 2em; }
  #meta { color: #888; font-size: 13px; margin-top: 4px; }
  .hidden { display: none; }
</style>
</head>
<body>
  <div class="row">
    <button id="tabLive" class="tab on">실시간</button>
    <button id="tabPhoto" class="tab">사진</button>
  </div>

  <div id="live">
    <video id="video" autoplay playsinline muted></video>
    <button id="liveBtn">시작</button>
  </div>

  <div id="photo" class="hidden">
    <div class="row">
      <label class="btn">카메라<input id="cam" type="file" accept="image/*" capture="environment" hidden></label>
      <label class="btn">갤러리<input id="gal" type="file" accept="image/*" hidden></label>
    </div>
    <img id="preview" alt="">
    <button id="send">Ask</button>
  </div>

  <input id="prompt" type="text" value="Describe the image in one sentence in Korean language.">
  <div id="answer"></div>
  <div id="meta"></div>

<script>
  const $ = id => document.getElementById(id);
  const answer = $('answer'), meta = $('meta'), video = $('video');
  const prompt = () => $('prompt').value || 'Describe the image in one sentence in Korean language.';

  async function ask(blob) {
    const form = new FormData();
    form.append('image', blob, 'frame.jpg');
    form.append('prompt', prompt());
    form.append('temperature', '0');  // greedy decoding: same frame -> same answer
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(data));
    return data;
  }

  // ---- tabs ----
  $('tabLive').onclick = () => { $('live').classList.remove('hidden'); $('photo').classList.add('hidden');
    $('tabLive').classList.add('on'); $('tabPhoto').classList.remove('on'); };
  $('tabPhoto').onclick = () => { stopLive(); $('photo').classList.remove('hidden'); $('live').classList.add('hidden');
    $('tabPhoto').classList.add('on'); $('tabLive').classList.remove('on'); };

  // ---- live mode: send the latest frame, wait for the answer, repeat ----
  let stream = null, running = false, lastSig = null, lastPrompt = null;
  const canvas = document.createElement('canvas');
  const small = document.createElement('canvas');
  small.width = small.height = 32;
  const SCENE_THRESHOLD = 12;  // mean abs pixel diff (0-255) needed to count as a new scene
  const INTERVAL_MS = 5000;    // send at most one frame every 5 seconds

  // Tiny grayscale thumbnail used to detect whether the scene actually changed
  function signature() {
    const ctx = small.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, 32, 32);
    const d = ctx.getImageData(0, 0, 32, 32).data;
    const g = new Uint8Array(32 * 32);
    for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
    return g;
  }

  function sceneChanged(sig) {
    if (!lastSig || prompt() !== lastPrompt) return true;
    let sum = 0;
    for (let i = 0; i < sig.length; i++) sum += Math.abs(sig[i] - lastSig[i]);
    return sum / sig.length > SCENE_THRESHOLD;
  }

  function grabFrame(maxSide) {
    const w = video.videoWidth, h = video.videoHeight;
    const s = Math.min(1, maxSide / Math.max(w, h));
    canvas.width = Math.round(w * s); canvas.height = Math.round(h * s);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.8));
  }

  async function loop() {
    while (running) {
      if (!video.videoWidth) { await new Promise(r => setTimeout(r, 100)); continue; }
      const t0 = performance.now();
      const sig = signature();
      if (sceneChanged(sig)) {
        try {
          const data = await ask(await grabFrame(768));
          if (!running) break;
          lastSig = sig; lastPrompt = prompt();
          answer.textContent = data.answer;
          meta.textContent = '모델 ' + data.latency_ms + ' ms · 왕복 ' + Math.round(performance.now() - t0) + ' ms';
        } catch (err) {
          meta.textContent = '요청 실패: ' + err.message;
        }
      }
      await new Promise(r => setTimeout(r, Math.max(0, INTERVAL_MS - (performance.now() - t0))));
    }
  }

  async function startLive() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      answer.textContent = '카메라를 쓸 수 없습니다. https:// 로 접속했는지 확인하세요.';
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    } catch (err) {
      answer.textContent = '카메라 권한 오류: ' + err.message;
      return;
    }
    video.srcObject = stream;
    running = true;
    $('liveBtn').textContent = '정지'; $('liveBtn').classList.add('stop');
    loop();
  }

  function stopLive() {
    running = false;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null;
    lastSig = null;
    $('liveBtn').textContent = '시작'; $('liveBtn').classList.remove('stop');
  }

  $('liveBtn').onclick = () => running ? stopLive() : startLive();

  // ---- photo mode ----
  let file = null;
  function pick(e) {
    file = e.target.files[0];
    if (file) { $('preview').src = URL.createObjectURL(file); answer.textContent = ''; meta.textContent = ''; }
  }
  $('cam').onchange = pick;
  $('gal').onchange = pick;
  $('send').onclick = async () => {
    if (!file) { answer.textContent = '이미지를 먼저 선택하세요.'; return; }
    $('send').disabled = true; answer.textContent = '생각 중...';
    try {
      const data = await ask(file);
      answer.textContent = data.answer;
      meta.textContent = data.latency_ms + ' ms';
    } catch (err) {
      answer.textContent = '요청 실패: ' + err.message;
    }
    $('send').disabled = false;
  };
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok", "model": vlm.model_name if vlm else None}


@app.post("/predict")
def predict(image: UploadFile = File(...),
            prompt: str = Form("Describe the image."),
            temperature: float = Form(0.2),
            max_new_tokens: int = Form(None)):
    try:
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    max_new_tokens = max(1, min(max_new_tokens or MAX_NEW_TOKENS, 1024))
    start = time.time()
    answer = vlm.predict(img, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    return {"answer": answer, "latency_ms": int((time.time() - start) * 1000)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ssl-certfile", type=str, default=None, help="HTTPS cert (needed for live camera in phone browsers)")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    args = parser.parse_args()

    MAX_NEW_TOKENS = args.max_new_tokens
    vlm = FastVLM(args.model_path, args.model_base, args.conv_mode, args.load_4bit, args.load_8bit)
    uvicorn.run(app, host=args.host, port=args.port,
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
