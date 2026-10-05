# FastVLM 서버 + Android 클라이언트

## 1. 서버 (Linux/WSL2, CUDA GPU)

```bash
conda create -n fastvlm python=3.10 -y && conda activate fastvlm
pip install -e .
pip install python-multipart

bash get_models.sh   # checkpoints/ 에 모델 다운로드

python server/server.py --model-path checkpoints/llava-fastvithd_0.5b_stage3 --port 8000
```

테스트:
```bash
curl http://localhost:8000/health
curl -F image=@docs/fastvlm-flexible_prompts.png -F prompt="Describe the image." http://localhost:8000/predict
```

옵션: `--load-4bit` / `--load-8bit` (7B 모델을 GPU 한 장에 올릴 때).

## 2. 폰에서 WSL2 서버에 접속하기

WSL2는 기본적으로 내부 NAT 뒤에 있어서 폰에서 바로 접근할 수 없습니다. 둘 중 하나를 하세요.

**A. Mirrored 네트워킹 (Windows 11 22H2+)** — `%UserProfile%\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```
PowerShell에서 `wsl --shutdown` 후 다시 실행.

**B. 포트 포워딩** — 관리자 PowerShell:
```powershell
$wslIp = (wsl hostname -I).Trim().Split(" ")[0]
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$wslIp
```

공통 — Windows 방화벽 허용 (관리자 PowerShell):
```powershell
New-NetFirewallRule -DisplayName "FastVLM 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

폰과 PC를 같은 와이파이에 연결하고, 앱의 Server URL에 `http://<Windows PC IP>:8000` 입력 (`ipconfig`로 확인).

## 3. Android 클라이언트

Android Studio에서 `android_client/` 폴더 열기 → Gradle Sync → 폰 연결 후 Run.
카메라/갤러리로 이미지 선택 → 프롬프트 입력 → Ask.

## 4. 실시간 카메라 모드 (폰 브라우저)

브라우저는 HTTPS에서만 실시간 카메라를 허용하므로 HTTPS로 실행합니다.

```bash
# 인증서 생성 (한 번만)
mkdir -p server/certs && openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=fastvlm" \
  -keyout server/certs/key.pem -out server/certs/cert.pem

python server/server.py --model-path checkpoints/llava-fastvithd_0.5b_stage3 --port 7500 \
  --ssl-certfile server/certs/cert.pem --ssl-keyfile server/certs/key.pem
```

폰에서 `https://<PC IP>:7500` 접속 → "안전하지 않음" 경고에서 고급 → 계속 → 카메라 권한 허용 → 실시간 탭에서 시작.
답변이 오면 바로 다음 프레임을 보내므로, 갱신 속도 = 모델 추론 속도입니다.
