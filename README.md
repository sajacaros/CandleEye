# CandlEye

## 프로젝트 개요

CandlEye는 업비트 KRW 마켓 4시간봉 데이터를 수집해 차트 이미지를 생성하고, 24시간 내 목표 수익률(5%+수수료) 달성 여부를 학습하는 FastAPI + PyTorch 기반 파이프라인입니다. ccxt로 캔들 데이터를 동기화하고 SQLite에 적재한 뒤, mplfinance로 캔들스틱 이미지를 만들고 레이블을 라벨링합니다. 이후 ResNet 등 CNN 모델로 학습해 신호 예측 및 API 서빙을 목표로 합니다.

---

## 설치 및 실행 방법

### 1️⃣ 가상환경 생성 및 의존성 설치
```bash
python3 -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
```

> `requirements.txt` 내용 예시:
```text
torch
torchvision
pandas
numpy
scikit-learn
Pillow
tqdm
ccxt
mplfinance
python-dotenv
pyyaml
requests
```

---

### 2️⃣ 환경 변수 설정
```bash
cp .env.example .env
# .env 편집 후 API 키 입력
```

---

### 3️⃣ 데이터 파이프라인 실행
#### (1) 캔들 데이터 수집
```bash
python src/data_collector.py --config configs/config.yaml
```
#### (2) 이미지 및 라벨 생성
```bash
python src/image_generator.py --config configs/config.yaml [--clean-output]
```

- 생성된 이미지: `data/images/`
- 라벨 파일: `data/processed/labels.csv`

---

### 4️⃣ 모델 학습 및 평가
#### (1) 모델 학습
```bash
python src/model_pipeline.py --mode train --data_csv data/processed/labels.csv --images_dir data/images --epochs 100 --batch_size 32 --pretrained
```
#### (2) 평가
```bash
python src/model_pipeline.py --mode eval --data_csv data/processed/labels.csv --images_dir data/images --checkpoint models/best_model.pth
```
#### (3) 단일 이미지 예측
```bash
python src/model_pipeline.py --mode predict --checkpoint models/best_model.pth --image data/images/KRW_BTC_202505190000.png
```
#### (4) focal loss
```bash
python src/model_pipeline.py --mode train --data_csv data/processed/labels.csv --images_dir data/images --epochs 80 --batch_size 64 --focal_alpha 0.7 --pretrained
```


---

### 5️⃣ FastAPI 서빙 (선택)
모델 학습이 완료되면 FastAPI를 통해 예측 API를 제공합니다.
```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 디렉토리 구조
```text
CandlEye/
├── configs/
│   └── config.yaml
├── data/
│   ├── candles.db
│   ├── images/
│   └── processed/labels.csv
├── models/
│   └── best_model.pth
├── src/
│   ├── data_collector.py
│   ├── image_generator.py
│   ├── api_server.py
│   └── model_pipeline.py
├── requirements.txt
└── README.md
```

---

## 참고
- 학습 시 GPU가 있을 경우 자동으로 CUDA를 사용합니다.
- 데이터 불균형(예: label=1 비율이 낮음)은 자동으로 `pos_weight`로 보정됩니다.
- 테스트 결과는 AUC, Accuracy, Precision, Recall, Confusion Matrix를 포함합니다.