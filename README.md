# CandlEye

> 업비트 차트 이미지 기반 암호화폐 수익 예측 시스템

## 프로젝트 개요

CandlEye는 업비트 1시간봉 차트를 CNN으로 분석하여 2일 후 4% 수익 달성 가능성을 예측하는 딥러닝 시스템입니다.

**핵심 기능**
- 📊 업비트 API 연동 (ccxt) → SQLite 저장
- 📈 캔들스틱 차트 이미지 생성 (mplfinance + 이동평균선)
- 🤖 ResNet18 기반 이진 분류 (Focal Loss)
- 💰 백테스팅 엔진 (수수료/슬리피지/손절매 반영)
- 📉 심볼별 성능 분석 및 실전 지향 모델 선택

**현재 버전**: v2.1 (코인별 시간 기반 분할, 투자 모델 기반 선택)

---

## 🚀 빠른 시작

```bash
# 환경 설정
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 데이터 수집 → 이미지 생성 → 학습 → 백테스팅
python src/data_collector.py --config configs/config.yaml
python src/image_generator.py --config configs/config.yaml --clean-output
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 80 --batch_size 32 --pretrained
python src/backtester.py \
  --model models/best_model.pth \
  --data data/processed/labels.csv \
  --config configs/config.yaml
```

---

## 주요 설정

**현재 설정** (`configs/config.yaml`):
- **시간 프레임**: 1시간봉 (168시간 = 7일 차트)
- **예측 목표**: 2일(48시간) 후 4% 수익
- **코인**: BTC, ETH, SOL
- **이동평균선**: 12시간, 1일, 3일
- **샘플링**: 12시간 간격 슬라이딩 윈도우

**모델 선택 기준** (실전 투자 최적화):
- Precision ≥ 65% (False Positive 제어)
- Recall ≥ 20% (기회 포착)
- AUC 최대화

**데이터 분할** (기본값):
- 코인별 독립 시간 분할: Train 70% / Val 15% / Test 15%
- 미래 데이터 누수 방지 및 실전 환경 재현

---

## 📁 디렉토리 구조

```
CandlEye/
├── src/                    # 소스 코드
│   ├── data_collector.py   # Upbit 데이터 수집
│   ├── image_generator.py  # 차트 이미지 생성
│   ├── model_pipeline.py   # 학습/평가/예측
│   └── backtester.py       # 백테스팅
├── configs/
│   └── config.yaml         # 전체 설정
├── data/
│   ├── candles.db          # SQLite DB
│   ├── images/             # 차트 PNG
│   └── processed/
│       └── labels.csv      # 메타데이터
└── models/
    └── best_model.pth      # 학습된 모델
```

---

## 주요 개선사항 (v2.1)

**1. 코인별 시간 기반 데이터 분할**
- 각 코인을 독립적으로 시간순 분할 (Train 70% / Val 15% / Test 15%)
- 미래 데이터 누수 차단 및 실전 환경 재현

**2. 투자 모델 기반 Best Model Selection**
- Precision ≥ 65% + Recall ≥ 20% + AUC 최대화
- False Positive 제어 및 실전 수익성 최적화

**3. 심볼별 성능 분석**
- 코인별 AUC/Precision/Recall 계산
- 모델 편향 감지

**4. 백테스팅 모듈**
- 수수료/슬리피지/손절매 반영
- Sharpe Ratio, Max Drawdown 제공

**5. 이동평균선 추가**
- MA 12h, 1일, 3일
- 추세 분석 및 과적합 방지

---

## 기술 스택

- **데이터**: ccxt, SQLite, pandas
- **차트**: mplfinance
- **모델**: PyTorch, torchvision (ResNet18)
- **평가**: scikit-learn

---

## 문서

- `CLAUDE.md`: Claude Code용 상세 가이드
- `MVP.md`: 프로젝트 설계
- `AGENTS.md`: AI 에이전트 활용