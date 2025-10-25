# CLAUDE.md - 차트 트레이딩 개발 가이드

## 프로젝트 개요

암호화폐 4시간봉 차트 이미지를 CNN(ResNet18)으로 학습시켜 24시간 내 5% 이상 상승 가능성을 예측하는 시스템

---

## 시스템 아키텍처

```
┌─────────────────┐
│  Upbit API      │
│  (데이터 수집)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  데이터 저장     │
│  (SQLite/CSV)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  이미지 생성기   │
│  (mplfinance)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  학습 파이프라인 │
│  (PyTorch)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  모델 저장소     │
│  (best_model)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  추론 서버       │
│  (FastAPI)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  대시보드        │
│  (Streamlit)    │
└─────────────────┘
```

---

## 1. 환경 설정

### 필수 패키지

- **딥러닝**: torch, torchvision
- **데이터**: pandas, numpy, pyupbit
- **시각화**: mplfinance, matplotlib
- **평가**: scikit-learn
- **서빙**: fastapi, uvicorn, redis
- **모니터링**: streamlit, apscheduler
- **실험 트래킹**: wandb (선택)

### 디렉토리 구조

```
project/
├── data/
│   ├── raw/              # Upbit API 수집 데이터
│   ├── images/           # 생성된 차트 이미지
│   └── processed/        # 전처리 완료 데이터
├── models/
│   ├── checkpoints/      # 학습 체크포인트
│   └── best_model.pth    # 최종 모델
├── src/
│   ├── data_collector.py # 데이터 수집
│   ├── image_generator.py# 차트 이미지 생성
│   ├── dataset.py        # PyTorch Dataset
│   ├── model.py          # ResNet18 모델
│   ├── train.py          # 학습 스크립트
│   ├── backtest.py       # 백테스트
│   └── api.py            # FastAPI 서버
├── notebooks/            # 실험용 노트북
└── configs/
    └── config.yaml       # 설정 파일
```

---

## 2. 데이터 수집

### Upbit API 사용

- **대상**: KRW-BTC, KRW-ETH
- **간격**: 4시간봉 (interval='240')
- **기간**: 2년치 (약 4,400개 캔들)
- **저장**: CSV 형식

### 데이터 정규화

- 가격: 샘플의 첫 캔들 종가 기준 비율로 변환
- 거래량: 해당 샘플의 평균 대비 상대 비율
- 이동평균선: 동일한 비율 적용

---

## 3. 레이블 생성

### 목표 수익률 달성 판정

- **입력**: 현재 캔들 종가
- **목표가**: entry_price × 1.051 (5% + 수수료 0.1%)
- **판정 기간**: 향후 24시간 (6개 캔들)
- **조건**: future_6_candles['high'].max() >= target_price
- **출력**: 1 (달성) or 0 (미달성)

### 슬라이딩 윈도우 샘플 생성

- **윈도우**: 30개 캔들 (5일치)
- **이동**: 1캔들씩 슬라이드
- **결과**: 1년 데이터 → 약 2,170개 샘플

---

## 4. 차트 이미지 생성

### 이미지 구성

**포함 요소**:

- 캔들스틱 (상승 빨강, 하락 파랑)
- 거래량 바 (하단 20%)
- 이동평균선 (MA5 빨강, MA20 노랑, MA60 초록)

**제외 요소**:

- 축 라벨, 그리드, 텍스트

### 이미지 사양

- **해상도**: 224 × 224 pixels
- **포맷**: RGB 3채널 PNG
- **비율**: 캔들 75%, 거래량 20%, 여백 5%

### 생성 도구

- **라이브러리**: mplfinance
- **파일명**: `{symbol}_{timestamp}.png`
- **최적화**: 멀티프로세싱, 이미지 캐싱

---

## 5. PyTorch Dataset

### ChartDataset 클래스

- **입력**: 차트 이미지 경로, 레이블
- **전처리**: Resize(224), ToTensor, Normalize (ImageNet 평균/표준편차)
- **출력**: (image_tensor, label_tensor)

---

## 6. 모델 아키텍처

### ResNet18 Transfer Learning

**구조**:

1. ResNet18 백본 (ImageNet pretrained)
2. FC layer 제거
3. 커스텀 분류기:
    - Linear(512 → 256) + ReLU + Dropout(0.5)
    - Linear(256 → 128) + ReLU + Dropout(0.3)
    - Linear(128 → 1) + Sigmoid

**학습 전략**:

- Phase 1: Backbone Freeze, Head만 학습 (10 epochs)
- Phase 2: 전체 Fine-tuning (50 epochs)

---

## 7. 학습 설정

### 데이터 분할

- **Train**: 70% (가장 오래된 데이터)
- **Validation**: 15% (중간 기간)
- **Test**: 15% (가장 최근 데이터)
- **주의**: 시계열 순서 유지, 랜덤 셔플 금지

### 손실 함수

- **기본**: BCELoss
- **클래스 불균형 대응**: Weighted BCE 또는 Focal Loss
- **가중치**: positive_weight = negative_count / positive_count

### 최적화

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR
- **Batch Size**: 32
- **Epochs**: 50-100

### 조기 종료

- **기준**: Validation AUC
- **Patience**: 20 epochs
- **저장**: Best AUC 모델만 저장

---

## 8. 평가

### 주요 지표

- **AUC-ROC**: 목표 > 0.70
- 클래스 불균형에 강건
- 임계값 무관 전반적 성능 평가

### 임계값 설정

- **Conservative**: 0.8 (높은 확신)
- **Balanced**: 0.7 (권장)
- **Aggressive**: 0.6 (기회 포착)

---

## 9. 백테스트

### 시뮬레이션 설정

- **초기 자본**: 1,000,000 KRW
- **포지션 크기**: 자본의 10%
- **수수료**: 0.1% (매수 + 매도)
- **슬리피지**: 0.1%

### 매매 로직

**진입 조건**:

- 모델 예측 > 임계값(0.7)
- 보유 포지션 없음

**청산 조건** (우선순위):

1. 목표 달성 (5% 수익)
2. 손절선 (-3% 손실, 선택)
3. 시간 제한 (24시간)

### 성과 지표

- 총 수익률 / 연환산 수익률
- 승률 / 목표 달성률
- 평균 수익/손실
- MDD (Maximum Drawdown)
- Sharpe Ratio
- 평균 보유 시간

### 성공 기준

- ✅ 목표 달성률 > 60%
- ✅ 연환산 수익률 > 20%
- ✅ 승률 > 55%
- ✅ MDD < 20%
- ✅ Sharpe Ratio > 1.0

---

## 10. 검증 전략

### 타 종목 검증

- BTC로 학습 → ETH, SOL, XRP로 테스트
- 목표: 성능 하락 < 10%

### 시장 상황별 검증

- 상승장, 하락장, 횡보장 각각 성능 측정
- 특정 구간 편향 확인

---

## 11. API 서버

### FastAPI 엔드포인트

**POST /predict**:

- 입력: {symbol, timestamp}
- 출력: {probability, signal}

**GET /health**:

- 상태: {status, model_version}

### 최적화

- 모델 메모리 로드 (재사용)
- GPU 추론 (가능 시)
- Redis 캐싱 (중복 요청 방지)

---

## 12. 모니터링 대시보드

### Streamlit 구성

1. **실시간 신호**: 최근 예측 결과, 확률 분포
2. **성능 트래킹**: 일별 수익률, 누적 수익 그래프
3. **모델 상태**: 예측 분포, 정확도 변화
4. **시스템 헬스**: API 응답 시간, 에러 로그

### 알림 기능

- 매수 신호 발생 시
- 목표 달성 시
- 시스템 에러 발생 시

---

## 13. 실행 순서

### 개발 단계

1. **데이터 수집**: Upbit API로 OHLCV 수집
2. **이미지 생성**: 30개 캔들 단위 차트 생성
3. **학습**: ResNet18 학습 및 검증
4. **백테스트**: 과거 데이터로 성능 검증
5. **API 서버**: FastAPI 서빙 구축
6. **대시보드**: Streamlit 모니터링

### 운영 단계

1. APScheduler로 4시간마다 자동 데이터 수집
2. 실시간 예측 API 서빙
3. 대시보드로 모니터링
4. 월 1회 재학습

---

## 14. 주요 하이퍼파라미터

### 데이터

- `window_size`: 30 (캔들 개수)
- `lookforward`: 6 (24시간)
- `target_return`: 0.05 (5%)
- `fee`: 0.001 (0.1%)

### 이미지

- `image_size`: 224
- `ma_periods`: [5, 20, 60]

### 학습

- `batch_size`: 32
- `learning_rate`: 1e-4
- `weight_decay`: 1e-5
- `dropout`: [0.5, 0.3]
- `epochs`: 50-100

### 백테스트

- `initial_capital`: 1,000,000
- `position_size`: 0.1
- `threshold`: 0.7
- `stop_loss`: -0.03 (선택)

