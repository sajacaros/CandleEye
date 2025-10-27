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

**이동평균선 설정** (`configs/config.yaml`):
```yaml
data:
  moving_averages: [5, 10, 20]  # MA5, MA10, MA20 추가
  # 빈 리스트 [] 또는 null로 설정하면 이동평균선 없이 생성
```
- MA5: 단기 (20시간)
- MA10: 중기 (40시간)
- MA20: 장기 (80시간 ≈ 3.3일)
- 이동평균선이 정확히 계산되는 구간만 차트 생성 (초기 20개 캔들 제외)

---

### 4️⃣ 모델 학습 및 평가
#### (1) 모델 학습 (시간 기반 분할 - 기본값)
```bash
python src/model_pipeline.py --mode train --data_csv data/processed/labels.csv --images_dir data/images --epochs 100 --batch_size 32 --pretrained
```

**새로운 기능**: 시간 기반 데이터 분할이 기본으로 활성화되어 데이터 누수를 방지합니다.
- Train: 가장 오래된 데이터 (70%)
- Validation: 중간 데이터 (15%)
- Test: 가장 최근 데이터 (15%)

랜덤 분할을 원하는 경우:
```bash
python src/model_pipeline.py --mode train --data_csv data/processed/labels.csv --images_dir data/images --epochs 100 --batch_size 32 --pretrained --random-split
```

#### (2) 평가 (심볼별 성능 분석 포함)
```bash
python src/model_pipeline.py --mode eval --data_csv data/processed/labels.csv --images_dir data/images --checkpoint models/best_model.pth
```

**새로운 기능**: 평가 시 자동으로 심볼별(KRW-BTC, KRW-ETH 등) 성능을 분석하여 모델 편향을 감지합니다.

#### (3) 단일 이미지 예측
```bash
python src/model_pipeline.py --mode predict --checkpoint models/best_model.pth --image data/images/KRW_BTC_202505190000.png
```

#### (4) Focal Loss 파라미터 조정
```bash
python src/model_pipeline.py --mode train --data_csv data/processed/labels.csv --images_dir data/images --epochs 80 --batch_size 64 --focal_alpha 0.7 --pretrained
```


---

### 5️⃣ 백테스팅 (신규 기능 ⭐)

학습된 모델의 실전 성능을 검증하기 위한 백테스팅 모듈입니다.
- 실제 거래 수수료 (0.1% × 2)
- 슬리피지 모델링
- Stop Loss / Take Profit 전략
- Sharpe Ratio, Max Drawdown, Win Rate 계산

```bash
python src/backtester.py --model models/best_model.pth --data data/processed/labels.csv --config configs/config.yaml
```

**예측 임계값 조정**:
```bash
# 보수적 전략 (높은 확률만)
python src/backtester.py --model models/best_model.pth --threshold 0.7

# 공격적 전략 (더 많은 거래)
python src/backtester.py --model models/best_model.pth --threshold 0.5
```

백테스트 설정은 `configs/config.yaml`에서 수정 가능:
```yaml
backtest:
  initial_capital: 1000000  # 초기 자본
  position_size: 0.1        # 포지션 크기 (10%)
  stop_loss: -0.03          # 손절매 (-3%)
  take_profit: 0.05         # 목표 수익률 (5%)
```

---

### 6️⃣ FastAPI 서빙 (선택)
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

## 주요 개선사항 (v2.0)

### 1. 시간 기반 데이터 분할 ⭐
- **문제**: 기존 랜덤 분할은 미래 데이터가 학습에 유입되어 과적합 발생
- **해결**: 시간 순으로 정렬 후 train(과거) → val(중간) → test(최근) 분할
- **효과**: 실제 트레이딩 환경과 동일한 조건에서 모델 검증 가능

### 2. 심볼별 성능 분석 ⭐
- **문제**: 특정 코인에 편향된 모델인지 확인 불가
- **해결**: 각 심볼(BTC, ETH, XRP 등)별 AUC, Accuracy, Precision, Recall 계산
- **효과**: 모델이 특정 코인에 과적합되지 않았는지 검증 가능

### 3. 백테스팅 모듈 ⭐
- **문제**: 모델 성능 지표(AUC 등)가 실제 수익으로 이어지는지 불명확
- **해결**: 실전과 동일한 조건(수수료, 슬리피지, 손절매)으로 거래 시뮬레이션
- **효과**: Sharpe Ratio, Max Drawdown 등 실전 지표로 모델 평가 가능

### 4. 이동평균선 추가 ⭐
- **문제**: 캔들 패턴만으로는 추세 파악이 불충분할 수 있음
- **해결**: 트레이더들이 실제로 보는 이동평균선(MA5, MA10, MA20) 차트에 추가
- **효과**:
  - 추세 방향 및 강도 시각화
  - 골든크로스/데드크로스 패턴 학습 가능
  - 지지/저항선 역할 인식
  - 과적합 방지 (스무딩된 가격 정보)

---

## 참고
- 학습 시 GPU가 있을 경우 자동으로 CUDA를 사용합니다.
- 데이터 불균형(예: label=1 비율이 낮음)은 Focal Loss와 WeightedRandomSampler로 보정됩니다.
- 테스트 결과는 AUC, Accuracy, Precision, Recall, Confusion Matrix를 포함합니다.
- 시간 기반 분할이 기본값이며, `--random-split` 플래그로 기존 방식 사용 가능합니다.