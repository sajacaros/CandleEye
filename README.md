# CandlEye

## 프로젝트 개요

CandlEye는 업비트 KRW 마켓 4시간봉 데이터를 수집해 차트 이미지를 생성하고, 24시간 내 목표 수익률(5%+수수료) 달성 여부를 학습하는 FastAPI + PyTorch 기반 파이프라인입니다. ccxt로 캔들 데이터를 동기화하고 SQLite에 적재한 뒤, mplfinance로 캔들스틱 이미지를 만들고 레이블을 라벨링합니다. 이후 ResNet 등 CNN 모델로 학습해 신호 예측 및 API 서빙을 목표로 합니다.

## 실행 준비

1. **가상환경 생성 및 활성화 (선택)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **의존성 설치**
   ```bash
   pip install ccxt requests python-dotenv pandas mplfinance pyyaml
   ```

3. **환경 변수 설정**
   - `.env.example`을 복사해 `.env`를 만들고, 업비트 API 키를 입력합니다.
   ```bash
   cp .env.example .env
   # .env 편집하여 UPBIT_ACCESS_KEY, 필요 시 UPBIT_SECRET_KEY 설정
   ```

## 데이터 파이프라인 실행

1. **캔들 데이터 수집**
   ```bash
   python src/data_collector.py --config configs/config.yaml
   ```
   - ccxt를 통해 기본 7개 마켓(KRW-BTC, ETH, XRP, SOL, DOGE, TRX, ADA)의 4시간봉을 SQLite `data/candles.db`에 저장합니다.

2. **이미지 및 라벨 생성**
   ```bash
   python src/image_generator.py --config configs/config.yaml
   ```
   - `data/images/`에 하루 간격(stride=6)으로 캔들 차트 이미지를 렌더링하고,
   - 메타데이터와 라벨을 `data/processed/labels.csv`로 출력합니다.

## 추가 정보

- `configs/config.yaml`을 수정하면 대상 심볼, 윈도우 크기, stride 등을 조정할 수 있습니다.
- 실행 전에 `data/`, `models/checkpoints/`는 `.gitkeep`만 추적되며 산출물은 `.gitignore`로 제외됩니다.
- 상세 기여 지침은 `AGENTS.md`를 참고해주세요.
