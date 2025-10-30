# CandlEye

## 프로젝트 개요

CandlEye는 업비트 KRW 마켓 4시간봉 데이터를 수집해 차트 이미지를 생성하고, 24시간 내 목표 수익률(5%+수수료) 달성 여부를 학습하는 FastAPI + PyTorch 기반 파이프라인입니다. ccxt로 캔들 데이터를 동기화하고 SQLite에 적재한 뒤, mplfinance로 캔들스틱 이미지를 만들고 레이블을 라벨링합니다. 이후 ResNet 등 CNN 모델로 학습해 신호 예측 및 API 서빙을 목표로 합니다.

---

## 🚀 빠른 시작 (Quick Start)

처음부터 끝까지 한 번에 실행:

```bash
# 1. 환경 설정
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # API 키 설정 (선택사항)

# 2. 데이터 수집 (약 10-30분 소요)
python src/data_collector.py --config configs/config.yaml

# 3. 차트 이미지 생성 (약 5-15분 소요)
python src/image_generator.py --config configs/config.yaml --clean-output

# 4. 모델 학습 (GPU: 1-2시간, CPU: 4-8시간)
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 80 --batch_size 32 --pretrained

# 5. 백테스팅 (약 1-5분)
python src/backtester.py \
  --model models/best_model.pth \
  --data data/processed/labels.csv \
  --config configs/config.yaml
```

---

## 📋 상세 설치 및 실행 방법

### 1️⃣ 가상환경 생성 및 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**의존성 패키지**:
- `torch`, `torchvision`: 딥러닝 프레임워크
- `pandas`, `numpy`: 데이터 처리
- `scikit-learn`: 평가 지표
- `mplfinance`: 캔들 차트 생성
- `ccxt`: 거래소 API
- `pyyaml`, `python-dotenv`: 설정 관리

**검증**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

### 2️⃣ 환경 변수 설정 (선택사항)

```bash
cp .env.example .env
# .env 파일 편집 (Upbit API 키 입력)
```

**주의**: API 키가 없어도 공개 데이터 수집은 가능하지만, rate limit이 엄격합니다.

---

### 3️⃣ 데이터 수집

```bash
python src/data_collector.py --config configs/config.yaml
```

**처리 내용**:
- Upbit API에서 9개 코인 4시간봉 데이터 수집
- `data/candles.db` SQLite 데이터베이스에 저장
- Rate limit 준수 (0.15초 간격)

**예상 소요 시간**: 10-30분 (수집 기간에 따라 다름)

**성공 확인**:
```bash
# DB 파일 생성 확인
ls -lh data/candles.db

# 데이터 건수 확인 (SQLite CLI 필요)
sqlite3 data/candles.db "SELECT market, COUNT(*) FROM candles GROUP BY market;"
```

**예상 결과**:
```
KRW-BTC | 2000
KRW-ETH | 2000
...
```

**트러블슈팅**:
- `ccxt` 관련 에러: API 키 확인 또는 잠시 후 재시도
- Rate limit 에러: `config.yaml`의 `fetch_batch_size` 줄이기

---

### 4️⃣ 차트 이미지 생성

```bash
python src/image_generator.py --config configs/config.yaml --clean-output
```

**처리 내용**:
- SQLite DB에서 캔들 데이터 읽기
- 24개 캔들 윈도우로 차트 이미지 생성 (MA 포함)
- 레이블 계산 (24시간 내 5% 상승 여부)
- 3캔들씩 슬라이딩 (stride=3)

**생성 파일**:
- `data/images/*.png`: 차트 이미지 (예: KRW_BTC_202501150800.png)
- `data/processed/labels.csv`: 메타데이터 (market, image_path, label, entry_price 등)

**예상 소요 시간**: 5-15분

**성공 확인**:
```bash
# 이미지 개수 확인
ls data/images/*.png | wc -l

# labels.csv 확인
head -5 data/processed/labels.csv
wc -l data/processed/labels.csv
```

**예상 결과**:
- 이미지: 5,000~20,000개 (데이터 양에 따라)
- labels.csv: 동일한 행 수 + 헤더

**이동평균선 설정** (`configs/config.yaml`):
```yaml
data:
  moving_averages: [5, 10, 20]  # MA5, MA10, MA20 차트에 표시
  # 비활성화: moving_averages: [] 또는 null
```
- MA5: 단기 추세 (20시간)
- MA10: 중기 추세 (40시간)
- MA20: 장기 추세 (80시간 ≈ 3.3일)
- MA가 유효한 구간만 생성 (초기 max(MA) 캔들 제외)

**차트 재생성** (설정 변경 시):
```bash
python src/image_generator.py --config configs/config.yaml --clean-output
```
`--clean-output` 플래그: 기존 이미지 삭제 후 재생성

---

### 5️⃣ 모델 학습

```bash
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 80 \
  --batch_size 32 \
  --pretrained
```

**모델 아키텍처**:
- Backbone: ResNet18 (ImageNet pretrained)
- Head: 3-layer FC (512→256→128→1)
- Loss: Focal Loss (class imbalance 대응)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR

**데이터 분할** (시간 기반 - 기본값 ⭐):
- Train: 가장 오래된 70% (학습용)
- Validation: 중간 15% (조기 종료 기준)
- Test: 가장 최근 15% (최종 평가)
- **효과**: 미래 데이터 누수 방지, 실전 환경 시뮬레이션

**예상 소요 시간**:
- GPU (CUDA): 1-2시간
- CPU: 4-8시간

**학습 중 출력 예시**:
```
Time-based split:
  Train: 2024-01-01 ~ 2024-07-31 (7000 samples)
  Val:   2024-08-01 ~ 2024-09-15 (1500 samples)
  Test:  2024-09-16 ~ 2024-10-27 (1500 samples)

pos: 450, neg: 9550, focal_loss_alpha: 0.955, focal_loss_gamma: 2.0

Epoch 1/80 | train_loss: 0.3214 | val_auc: 0.5821 | val_acc: 0.9100
Saved best model
...
Epoch 25/80 | train_loss: 0.1823 | val_auc: 0.6892 | val_acc: 0.9245
Saved best model
```

**생성 파일**:
- `models/best_model.pth`: 검증 AUC가 가장 높은 모델

**성공 확인**:
```bash
ls -lh models/best_model.pth
```

**랜덤 분할로 학습** (비추천):
```bash
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 80 --batch_size 32 --pretrained \
  --random-split
```

**고급 옵션**:
```bash
# Focal Loss 파라미터 조정
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 80 --batch_size 64 \
  --focal_alpha 0.7 --focal_gamma 2.5 \
  --pretrained

# 학습률 조정
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --lr 5e-5 --weight_decay 1e-4 \
  --pretrained
```

---

### 6️⃣ 모델 평가 및 검증

```bash
python src/model_pipeline.py --mode eval \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --checkpoint models/best_model.pth
```

**출력 내용**:
1. **전체 성능 지표**:
   - AUC (Area Under ROC Curve)
   - Accuracy, Precision, Recall
   - Confusion Matrix

2. **심볼별 성능 분석** (⭐ 신규):
   - 각 코인(BTC, ETH, XRP 등)별 성능
   - 모델 편향 감지
   - 평균 AUC 및 표준편차

**출력 예시**:
```
================================================================================
OVERALL TEST METRICS
================================================================================
{'auc': 0.6892, 'accuracy': 0.9245, 'precision': 0.4521, 'recall': 0.3156,
 'confusion_matrix': array([[1350,   45], [ 87,   40]]),
 'n_pos': 127, 'n_total': 1522}

================================================================================
PER-SYMBOL PERFORMANCE ANALYSIS
================================================================================

KRW-BTC:
  Samples: 215 (Positive: 18, 8.4%)
  AUC: 0.7234
  Accuracy: 0.9302
  Precision: 0.5000
  Recall: 0.3889

KRW-ETH:
  Samples: 198 (Positive: 15, 7.6%)
  AUC: 0.6721
  ...

================================================================================
SUMMARY
================================================================================
Mean AUC across symbols: 0.6892 (±0.0523)
Best symbol: KRW-BTC (AUC: 0.7234)
Worst symbol: KRW-DOGE (AUC: 0.6103)
```

**단일 이미지 예측**:
```bash
python src/model_pipeline.py --mode predict \
  --checkpoint models/best_model.pth \
  --image data/images/KRW_BTC_202501150800.png
```

**출력**: `probability: 0.6234` (0.55 이상이면 매수 신호)

---

### 7️⃣ 백테스팅 (실전 검증 ⭐)

```bash
python src/backtester.py \
  --model models/best_model.pth \
  --data data/processed/labels.csv \
  --config configs/config.yaml
```

**처리 내용**:
- Test Set에서 모델 예측 (확률 ≥ 0.55 → 매수 신호)
- 각 매수 시점부터 24시간 동안 실제 가격 추적
- 거래 시뮬레이션:
  - 5% 도달 → 익절 (target)
  - -3% 도달 → 손절 (stop_loss)
  - 24시간 경과 → 시간 종료 (timeout)
- 수수료 0.2% (매수 0.1% + 매도 0.1%) 반영
- 슬리피지 0.05% 반영

**예상 소요 시간**: 1-5분

**출력 예시**:
```
================================================================================
BACKTEST RESULTS
================================================================================
Initial Capital: $1,000,000.00
Final Capital:   $1,045,230.00
Total Return:    $45,230.00 (4.52%)

Trades:          127
Wins:            54 (42.52%)
Losses:          73
Avg Win:         8.32%
Avg Loss:        -2.87%

Max Drawdown:    -12.34%
Sharpe Ratio:    1.234
================================================================================

Sample Trades (first 10):
--------------------------------------------------------------------------------
KRW-BTC | Entry: $45123.00 @ 2024-10-15T08:00:00
  Exit: $47379.15 @ 2024-10-15T20:00:00 (target)
  PnL: $2,203.45 (+4.88%)

KRW-ETH | Entry: $2345.67 @ 2024-10-16T12:00:00
  Exit: $2275.10 @ 2024-10-16T16:00:00 (stop_loss)
  PnL: -$702.34 (-2.99%)
...
```

**성능 지표 설명**:
- **Total Return**: 전체 수익률
- **Win Rate**: 승률 (익절 비율)
- **Avg Win/Loss**: 평균 수익/손실
- **Max Drawdown**: 최대 낙폭 (리스크 지표)
- **Sharpe Ratio**: 위험 대비 수익률 (1.0 이상이 양호)

**예측 임계값 조정**:
```bash
# 보수적 전략 (높은 확률만, 적은 거래)
python src/backtester.py \
  --model models/best_model.pth \
  --data data/processed/labels.csv \
  --config configs/config.yaml \
  --threshold 0.7

# 공격적 전략 (낮은 확률도 포함, 많은 거래)
python src/backtester.py \
  --model models/best_model.pth \
  --data data/processed/labels.csv \
  --config configs/config.yaml \
  --threshold 0.5
```

**백테스트 설정** (`configs/config.yaml`):
```yaml
backtest:
  initial_capital: 1000000  # 초기 자본 (원)
  position_size: 0.1        # 포지션 크기 (자본의 10%)
  stop_loss: -0.03          # 손절매 (-3%)
  take_profit: 0.05         # 목표 수익률 (5%)
```

**성공 기준**:
- Win Rate > 40%
- Sharpe Ratio > 1.0
- Max Drawdown < 20%
- Total Return > 0%

---

### 8️⃣ FastAPI 서빙 (선택)

모델 학습이 완료되면 FastAPI를 통해 예측 API를 제공합니다.

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

**주의**: `src/api_server.py`는 아직 구현되지 않았습니다.

---

## 📁 디렉토리 구조

```text
CandlEye/
├── configs/
│   └── config.yaml              # 전체 설정 (데이터, 학습, 백테스팅)
├── data/
│   ├── candles.db               # SQLite DB (캔들 데이터)
│   ├── images/                  # 차트 이미지 (PNG)
│   │   ├── KRW_BTC_202501150800.png
│   │   └── ...
│   ├── processed/
│   │   └── labels.csv           # 메타데이터 (market, label, entry_price 등)
│   └── samples/                 # 샘플 차트 (참고용)
├── models/
│   ├── best_model.pth           # 학습된 모델 (검증 AUC 최고)
│   └── checkpoints/             # 중간 체크포인트 (선택)
├── src/
│   ├── __init__.py
│   ├── config.py                # 설정 로더
│   ├── data_collector.py        # Upbit 데이터 수집
│   ├── storage.py               # SQLite 인터페이스
│   ├── upbit_client.py          # ccxt 래퍼
│   ├── image_generator.py       # 차트 이미지 생성
│   ├── model_pipeline.py        # 학습/평가/예측
│   └── backtester.py            # 백테스팅 엔진
├── .env.example                 # 환경 변수 템플릿
├── requirements.txt             # Python 의존성
├── README.md                    # 이 문서
├── MVP.md                       # MVP 설계 문서
└── AGENTS.md                    # 에이전트 가이드
```

---

## ⚠️ 트러블슈팅

### 1. 데이터 수집 실패

**증상**: `ccxt` 관련 에러, rate limit 초과
```bash
ccxt.base.errors.RateLimitExceeded: upbit {"error":{"name":"too_many_requests"}}
```

**해결**:
```bash
# config.yaml에서 배치 크기 줄이기
data:
  fetch_batch_size: 100  # 200 → 100

# 또는 잠시 후 재시도 (1-2분 대기)
```

### 2. 이미지 생성 실패

**증상**: `Not enough data points to create a window`

**원인**: MA20 사용 시 최소 44개(20+24) 캔들 필요

**해결**:
```bash
# 데이터 더 수집하거나 MA 비활성화
data:
  moving_averages: []  # 또는 [5, 10]으로 줄이기
```

### 3. labels.csv 없음

**증상**: `FileNotFoundError: data/processed/labels.csv`

**해결**:
```bash
# 이미지 생성 먼저 실행
python src/image_generator.py --config configs/config.yaml
```

### 4. GPU 메모리 부족

**증상**: `CUDA out of memory`

**해결**:
```bash
# 배치 크기 줄이기
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --batch_size 16 \  # 32 → 16
  --pretrained
```

### 5. 학습이 너무 느림

**증상**: Epoch 1개에 30분 이상 소요

**해결**:
- GPU 사용 확인: `nvidia-smi` (CUDA 버전 체크)
- CPU로 학습 시: epochs 줄이기 (80 → 20)
- DataLoader workers 조정: `num_workers=2` (코드 수정 필요)

### 6. 모델 성능이 너무 낮음

**증상**: AUC < 0.55, Accuracy > 0.95 (모두 0 예측)

**원인**: 극심한 클래스 불균형 (label=1 비율 < 2%)

**해결**:
```bash
# Focal Loss alpha 조정
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --focal_alpha 0.9 \  # 자동 계산 대신 수동 설정
  --focal_gamma 3.0 \  # gamma 높이기
  --pretrained
```

### 7. 백테스팅 거래 수가 0

**증상**: `Generated 0 buy signals from 1500 test samples`

**원인**: 예측 확률이 모두 threshold 미만

**해결**:
```bash
# threshold 낮추기
python src/backtester.py \
  --model models/best_model.pth \
  --data data/processed/labels.csv \
  --config configs/config.yaml \
  --threshold 0.3  # 0.55 → 0.3
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

## 💡 참고 사항

### 시스템 요구사항
- **Python**: 3.10 이상
- **GPU**: CUDA 지원 GPU 권장 (학습 속도 5-10배 향상)
- **메모리**: 최소 8GB RAM (GPU 메모리 4GB 이상 권장)
- **디스크**: 최소 5GB 여유 공간 (데이터 + 모델)

### 주요 특징
- **자동 GPU 감지**: CUDA 사용 가능 시 자동으로 GPU 사용
- **클래스 불균형 처리**: Focal Loss + WeightedRandomSampler
- **시간 기반 데이터 분할**: 기본값 (데이터 누수 방지)
- **조기 종료**: 검증 AUC 20 epoch 동안 개선 없으면 중단
- **체크포인트**: 최고 검증 AUC 모델만 저장

### 평가 지표
- **AUC**: 0.70 이상 목표 (threshold 독립적)
- **Accuracy**: 클래스 불균형으로 인해 높게 나올 수 있음 (주의)
- **Precision**: 긍정 예측의 정확도
- **Recall**: 실제 긍정의 포착률
- **Win Rate (백테스팅)**: 40% 이상 목표

### 데이터 설정
- **Window**: 24개 4시간봉 (4일 차트)
- **Lookahead**: 6개 4시간봉 (24시간 예측)
- **Stride**: 3 (12시간 간격)
- **Target**: 5% 수익 + 0.1% 수수료
- **Moving Averages**: MA5, MA10, MA20 (기본값)

### 랜덤 분할 사용 시
```bash
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 80 --batch_size 32 --pretrained \
  --random-split
```
⚠️ **비추천**: 미래 데이터 누수 가능성, 과적합 위험

---

## 📚 추가 문서
- `MVP.md`: 프로젝트 설계 및 MVP 명세
- `AGENTS.md`: AI 에이전트 활용 가이드
- `NOTES.md`: 개발 노트 및 메모