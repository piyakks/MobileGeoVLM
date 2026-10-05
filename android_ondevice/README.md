# FastVLM 온디바이스 Android 앱

모델을 폰 안에서 직접 실행합니다 (인터넷/서버 불필요). ONNX Runtime + CameraX.

## 1. 모델 준비 (PC)

```bash
conda activate fastvlm-export
python onnx_export/export_onnx.py   --model-path checkpoints/llava-fastvithd_0.5b_stage3 --out onnx_export/fastvlm_0.5b
python onnx_export/quantize_onnx.py --onnx-dir onnx_export/fastvlm_0.5b
# 검증 (PyTorch와 비교)
python onnx_export/run_onnx.py --onnx-dir onnx_export/fastvlm_0.5b --image llava/serve/examples/waterview.jpg \
    --decoder decoder_q4.onnx --embed embed_tokens_int8.onnx
```

폰에 올릴 파일 (모델 폴더 하나에 아래 7개, 1.5B 기준 약 1.7GB):
`vision_encoder.onnx`(fp32), `embed_tokens_int8.onnx`, `decoder_q4.onnx` + `decoder_q4.onnx.data`, `vocab.json`, `merges.txt`, `fastvlm_meta.json`

> 비전 인코더 int8 양자화는 출력이 망가져서 사용하지 않습니다 (fp32 유지).

## 2. 설치 (Windows PowerShell, 폰 USB 디버깅 켜기)

```powershell
cd C:\tae\vlm_project\ml-fastvlm
adb install android_ondevice\fastvlm-ondevice-debug.apk
adb shell mkdir -p /sdcard/Android/data/com.example.fastvlm.ondevice/files/models/1.5b-geochat
adb push onnx_export\fastvlm_1.5b_geochat_mobile\. /sdcard/Android/data/com.example.fastvlm.ondevice/files/models/1.5b-geochat/
```

`files/models/` 아래 폴더 하나가 모델 하나이며, 앱의 모델 목록에 폴더 이름으로 표시됩니다.
앱을 한 번 실행한 뒤 push 하면 폴더 권한 문제가 덜합니다.

## 3. 사용

- **모델 / 작업 선택**: 자유 질문, 빨간 동그라미, Grounding, Refer, Identify
- **실시간 시작**: 후면 카메라 → 5초마다 한 번, 장면이 바뀌었을 때만 추론 (greedy)
- **갤러리**: 사진 추론 (그리기가 필요한 작업은 그린 뒤 **질문하기**)
- 답변 속 GeoChat 박스는 이미지 위에 자동으로 그려집니다
- 설정값은 `MainActivity.kt`의 `INTERVAL_MS`, `SCENE_THRESHOLD`, `Task`(작업별 최대 토큰)

## 빌드

Android Studio에서 `android_ondevice/` 열기 → Run. (arm64-v8a 전용)
