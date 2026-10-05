# MobileGeoVLM

위성·항공 영상 분석용 Vision-Language Model을 **안드로이드 기기에서 오프라인으로** 실행하는 프로젝트입니다.
Apple의 [FastVLM](https://github.com/apple/ml-fastvlm) (FastViTHD + Qwen2) 1.5B 모델을
[GeoChat](https://github.com/mbzuai-oryx/GeoChat) 원격탐사 데이터로 LoRA 파인튜닝하고, ONNX로 변환·양자화해 Galaxy Tab S9에서 구동합니다.

## 주요 기능
- **원격탐사 특화 VLM**: 장면 분류, VQA, 물체 탐지(`[grounding]`), 물체 찾기(`[refer]`), 영역 식별(`[identify]`)
- **온디바이스 추론**: 인터넷/서버 없이 ONNX Runtime(CPU)으로 실행
- **실시간 카메라 / 갤러리** 입력, 5초 간격 추론 + 장면 변화 감지
- **박스 자동 시각화**: 답변 속 `<p>물체</p>{<x1><y1><x2><y2>|<각도>}`를 회전 박스·라벨로 표시
- **영역 지정**: 화면에 그린 영역을 `[identify]` 박스 좌표로 변환
- 지연 시간 · RAM 사용량 표시

## 파이프라인
```
FastVLM-1.5B (stage3) ──LoRA 파인튜닝 (GeoChat_Instruct 308k)──▶ LoRA adapter
        │                                                           │
        └────────────── merge_lora.py ◀─────────────────────────────┘
                            │
                     export_onnx.py ──▶ vision_encoder / embed_tokens / decoder(+KV cache) .onnx
                            │
                    quantize_onnx.py ──▶ decoder 4bit (MatMulNBits), embedding int8, vision fp32
                            │
                Android 앱 (Kotlin + ONNX Runtime + CameraX)
```

## 결과 (Galaxy Tab S9, Snapdragon 8 Gen 2, CPU)
| 항목 | 값 |
|---|---|
| 모델 크기 (기기) | 약 1.7 GB |
| 지연 (한 문장 답변) | 약 8 초 (비전 인코더 약 5초) |
| 지연 (Grounding) | 약 9~10 초 |
| 생성 속도 | 약 19 tok/s |
| RAM (추론 중 최대) | 약 2.6 GB |

양자화 영향: 디코더 4bit·임베딩 int8은 FP32와 거의 동일한 답을 냈고, 비전 인코더 int8은 출력이 망가져 FP32를 유지했습니다.
FP32 ONNX는 PyTorch와 greedy 출력이 토큰 단위로 일치합니다.

## 폴더 구조
| 경로 | 내용 |
|---|---|
| `scripts/finetune_geochat_lora.sh` | GeoChat LoRA 학습 (DeepSpeed ZeRO-2, RTX 3090 x2) |
| `llava/train/train_qwen.py` | 학습 코드 (`--max_samples`: 데이터 비율을 유지한 서브샘플링) |
| `predict_lora.py`, `geochat_demo.py` | PC 추론 (원본 vs LoRA 비교), Gradio 데모 |
| `onnx_export/` | `merge_lora.py` → `export_onnx.py` → `quantize_onnx.py`, 검증용 `run_onnx.py` |
| `android_ondevice/` | 온디바이스 안드로이드 앱 |
| `server/`, `android_client/` | (선택) PC GPU 서버 + 웹/안드로이드 클라이언트 |

## 실행 방법
```bash
# 1. 환경 (학습/추론: transformers 4.48.3)
conda create -n fastvlm-train python=3.10 && conda activate fastvlm-train
pip install -e ".[train]" && pip install flash-attn --no-build-isolation
bash get_models.sh                                   # FastVLM 체크포인트

# 2. LoRA 학습
DATA_DIR=/path/to/GeoChat_data bash scripts/finetune_geochat_lora.sh

# 3. 추론 확인
python predict_lora.py --lora_path checkpoints/llava-fastvithd_1.5b_stage3-geochat-lora \
    --model_base checkpoints/llava-fastvithd_1.5b_stage3 --image_file demo_images/church_183.png \
    --prompt "[grounding] describe this image in detail"

# 4. 병합 → ONNX → 양자화
python onnx_export/merge_lora.py --lora-path checkpoints/llava-fastvithd_1.5b_stage3-geochat-lora \
    --model-base checkpoints/llava-fastvithd_1.5b_stage3 --out checkpoints/fastvlm-1.5b-geochat-merged
python onnx_export/export_onnx.py --model-path checkpoints/fastvlm-1.5b-geochat-merged --out onnx_export/fastvlm_1.5b_geochat
python onnx_export/quantize_onnx.py --onnx-dir onnx_export/fastvlm_1.5b_geochat

# 5. 앱 설치 및 모델 복사 → android_ondevice/README.md 참고
```

## 구현 메모
- **3분할 ONNX**: 이미지는 한 번만 인코딩하고, 디코더만 KV 캐시와 함께 토큰마다 반복 실행
- **Qwen2 BPE 토크나이저를 Kotlin으로 구현** (HF tokenizers와 결과 일치 테스트)
- **fp16 NaN 이슈**: 학습 후 저장된 config로 eager attention이 선택되면 Qwen2의 큰 `k_proj` bias 때문에 fp16에서 오버플로 → SDPA 강제
- **생성 미종료 이슈**: 체크포인트의 `generation_config`에 `eos_token_id`가 없어 `<|im_end|>`에서 멈추도록 지정

## 라이선스 / 출처
- 코드: 원본 FastVLM의 [LICENSE](LICENSE) (수정 및 추가 코드 포함)
- 모델 가중치 및 파생 모델: Apple [LICENSE_MODEL](LICENSE_MODEL) — **연구 목적으로만 사용 가능**. 이 저장소에는 가중치가 포함되어 있지 않습니다.
- 학습 데이터: [GeoChat_Instruct](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct) (MBZUAI)
- 기반 코드: [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm), [LLaVA](https://github.com/haotian-liu/LLaVA), [GeoChat](https://github.com/mbzuai-oryx/GeoChat)

---

<details>
<summary>원본 FastVLM README</summary>

# FastVLM: Efficient Vision Encoding for Vision Language Models

This is the official repository of
**[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303). (CVPR 2025)**

[//]: # (![FastViTHD Performance]&#40;docs/acc_vs_latency_qwen-2.png&#41;)
<p align="center">
<img src="docs/acc_vs_latency_qwen-2.png" alt="Accuracy vs latency figure." width="400"/>
</p>

### Highlights
* We introduce FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images.  
* Our smallest variant outperforms LLaVA-OneVision-0.5B with 85x faster Time-to-First-Token (TTFT) and 3.4x smaller vision encoder.
* Our larger variants using Qwen2-7B LLM outperform recent works like Cambrian-1-8B while using a single image encoder with a 7.9x faster TTFT.
* Demo iOS app to demonstrate the performance of our model on a mobile device.

<table>
<tr>
    <td><img src="docs/fastvlm-counting.gif" alt="FastVLM - Counting"></td>
    <td><img src="docs/fastvlm-handwriting.gif" alt="FastVLM - Handwriting"></td>
    <td><img src="docs/fastvlm-emoji.gif" alt="FastVLM - Emoji"></td>
</tr>
</table>

## Getting Started
We use LLaVA codebase to train FastVLM variants. In order to train or finetune your own variants, 
please follow instructions provided in [LLaVA](https://github.com/haotian-liu/LLaVA) codebase. 
We provide instructions for running inference with our models.   

### Setup
```bash
conda create -n fastvlm python=3.10
conda activate fastvlm
pip install -e .
```

### Usage Example
To run inference of PyTorch checkpoint, follow the instruction below
```bash
python predict.py --model-path /path/to/checkpoint-dir \
                  --image-file /path/to/image.png \
                  --prompt "Describe the image."
```

### Inference on Apple Silicon
To run inference on Apple Silicon, pytorch checkpoints have to be exported to format 
suitable for running on Apple Silicon, detailed instructions and code can be found [`model_export`](model_export/) subfolder.
Please see the README there for more details.

For convenience, we provide 3 models that are in Apple Silicon compatible format: [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3_llm.fp16.zip), 
[fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3_llm.int8.zip), 
[fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3_llm.int4.zip). 
We encourage developers to export the model of their choice with the appropriate quantization levels following 
the instructions in [`model_export`](model_export/).

### Inference on Apple Devices
To run inference on Apple devices like iPhone, iPad or Mac, see [`app`](app/) subfolder for more details.

## Citation
If you found this code useful, please cite the following paper:
```
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
}
```

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.

</details>
