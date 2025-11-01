# WSL2 환경에서 설치 가이드

## 1. 사전 준비

### Python 버전 확인
```bash
python3 --version  # 3.10 이상 필요
```

### 필수 시스템 패키지 설치
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev build-essential
```

---

## 2. 가상환경 생성 및 활성화

```bash
# 프로젝트 디렉토리로 이동
cd /home/sajacaros/workspace/CandlEye

# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 활성화 확인 (프롬프트 앞에 (.venv) 표시됨)
which python  # /home/sajacaros/workspace/CandlEye/.venv/bin/python
```

---

## 3. pip 업그레이드

```bash
pip install --upgrade pip setuptools wheel
```

---

## 4. Requirements 설치

### 4-1. CPU 전용 (GPU 없는 경우)

```bash
# CPU 버전 PyTorch 먼저 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
pip install -r requirements.txt
```

### 4-2. GPU (CUDA) 지원 (NVIDIA GPU 있는 경우)

**WSL2에서 CUDA 사용 가능 확인**:
```bash
nvidia-smi  # GPU 정보가 나오면 CUDA 사용 가능
```

**CUDA 11.8 버전 설치**:
```bash
# CUDA 11.8 버전 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치
pip install -r requirements.txt
```

**CUDA 12.1 버전 설치** (최신):
```bash
# CUDA 12.1 버전 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 나머지 패키지 설치
pip install -r requirements.txt
```

---

## 5. 설치 확인

### PyTorch 설치 확인
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" # GPU인 경우
```

**예상 출력**:
```
PyTorch: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
```

### 전체 패키지 확인
```bash
pip list
```

---

## 6. 트러블슈팅

### 문제 1: "No module named 'pip'"
```bash
# pip 재설치
sudo apt install --reinstall python3-pip
```

### 문제 2: mplfinance 설치 실패
```bash
# matplotlib 먼저 설치
pip install matplotlib
pip install mplfinance
```

### 문제 3: CUDA 버전 확인
```bash
# NVIDIA 드라이버 버전 확인
nvidia-smi

# 지원하는 CUDA 버전 확인 (WSL2에서 CUDA Toolkit 설치 불필요)
# PyTorch가 자체 CUDA 라이브러리를 포함하고 있음
```

### 문제 4: 가상환경 비활성화
```bash
deactivate
```

### 문제 5: 가상환경 삭제 후 재생성
```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

---

## 7. 환경 변수 설정 (선택사항)

```bash
# .env 파일 생성
cp .env.example .env

# 편집
nano .env  # 또는 vim .env

# 내용 예시:
# UPBIT_ACCESS_KEY=your_access_key
# UPBIT_SECRET_KEY=your_secret_key
```

---

## 8. 설치 후 빠른 테스트

```bash
# 데이터 수집 테스트 (3개 코인만)
python src/data_collector.py --config configs/config.yaml --max-batches 5

# DB 확인
ls -lh data/candles.db
```

---

## WSL2 특이사항

1. **가상환경은 항상 활성화 필요**
   ```bash
   source .venv/bin/activate  # 터미널 열 때마다 실행
   ```

2. **GPU 사용 시 WSL2 + NVIDIA Driver 필요**
   - Windows에 NVIDIA Driver 설치 (WSL2 지원 버전)
   - WSL2 내부에는 CUDA Toolkit 설치 불필요
   - PyTorch가 자체 CUDA 라이브러리 포함

3. **경로 주의**
   - Windows 경로: `/mnt/c/Users/...`
   - WSL2 홈: `/home/username/...`
   - 프로젝트는 WSL2 파일시스템에 위치 권장 (성능)

4. **.venv 폴더는 git에서 제외**
   - `.gitignore`에 이미 포함됨
   - 각 환경에서 독립적으로 생성

---

## 자주 사용하는 명령어

```bash
# 가상환경 활성화 (매번)
source .venv/bin/activate

# 패키지 추가 설치
pip install <package_name>

# 패키지 업그레이드
pip install --upgrade <package_name>

# 현재 설치된 패키지 저장
pip freeze > requirements_frozen.txt

# 가상환경 비활성화
deactivate
```
