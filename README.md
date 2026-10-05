# MobileGeoVLM

위성·항공 영상 분석용 Vision-Language Model을 **안드로이드 기기에서 오프라인으로** 실행하는 프로젝트입니다.

기존 학습된 VLM에 LORA기반 Finetuning을 통한 모델 설계  

모델: FastVLM 1.5B [FastVLM](https://github.com/apple/ml-fastvlm)

학습데이터:[GeoChat](https://github.com/mbzuai-oryx/GeoChat)

구동기기: Galaxy Tab S9

## 주요 기능
- **원격탐사 특화 VLM**: 장면 분류, VQA, 물체 탐지(`[grounding]`), 물체 찾기(`[refer]`), 영역 식별(`[identify]`)
- **온디바이스 추론**: 인터넷/서버 없이 CPU로 실행
- **박스 자동 시각화**: 답변 속 `<p>물체</p>{<x1><y1><x2><y2>|<각도>}`를 회전 박스·라벨로 표시
- **영역 지정**: 화면에 그린 영역을 `[identify]` 박스 좌표로 변환
- 지연 시간 · RAM 사용량 표시

## 파이프라인
```
FastVLM-1.5B (stage3) ────────────────▶ LoRA adapter
        │                                       │
        └──────────── merge_lora.py ────────────┘
                            │
                     export_onnx.py
                            │
                    quantize_onnx.py
                            │
                        Android 앱 
```

## 결과 (Galaxy Tab S9, Snapdragon 8 Gen 2, CPU)
| 항목 | 값 |
|---|---|
| 모델 크기 (기기) | 약 1.7 GB |
| 지연 (한 문장 답변) | 약 8 초 (비전 인코더 약 5초) |
| 지연 (Grounding) | 약 9~10 초 |
| 생성 속도 | 약 19 tok/s |
| RAM (추론 중 최대) | 약 2.6 GB |

양자화 영향: 디코더: int4,텍스터 임베딩:int8, 비전 인코더:FP32

## 실행 방법
```bash
# 1. 환경 
conda create -n fastvlm-train python=3.10 && conda activate fastvlm-train
pip install -e ".[train]" && pip install flash-attn --no-build-isolation
bash get_models.sh                                   # FastVLM 체크포인트

# 2. LoRA 학습
bash scripts/finetune_geochat_lora.sh

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
## Demo 영상
<img width="634" height="1010" alt="녹음 2026-10-05 161556 (1)" src="https://github.com/user-attachments/assets/0ad16e07-b090-4ca4-9040-0b07658d1150" />



## 라이선스 / 출처
- 코드: 원본 FastVLM의 [LICENSE](LICENSE) (수정 및 추가 코드 포함)
- 모델 가중치 및 파생 모델: Apple [LICENSE_MODEL](LICENSE_MODEL) — **연구 목적으로만 사용 가능**. 이 저장소에는 가중치가 포함되어 있지 않습니다.
- 학습 데이터: [GeoChat_Instruct](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct) (MBZUAI)
- 기반 코드: [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm), [LLaVA](https://github.com/haotian-liu/LLaVA), [GeoChat](https://github.com/mbzuai-oryx/GeoChat)
