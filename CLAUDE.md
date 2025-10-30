# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CandlEye is a cryptocurrency trading signal prediction system that uses CNN (ResNet18) to analyze 4-hour candlestick chart images and predict whether a coin will achieve 5%+ returns within 24 hours. The pipeline collects Upbit KRW market data via ccxt, stores it in SQLite, generates candlestick images with mplfinance, and trains a PyTorch model for binary classification.

## Common Commands

### Data Collection
```bash
# Collect candle data from Upbit API
python src/data_collector.py --config configs/config.yaml

# Collect specific symbols only
python src/data_collector.py --config configs/config.yaml --symbols KRW-BTC KRW-ETH

# Limit batches per symbol (default: 100)
python src/data_collector.py --config configs/config.yaml --max-batches 50
```

### Image Generation
```bash
# Generate chart images and labels
python src/image_generator.py --config configs/config.yaml

# Clean existing images before generation
python src/image_generator.py --config configs/config.yaml --clean-output

# Process specific symbols
python src/image_generator.py --config configs/config.yaml --symbols KRW-BTC
```

### Model Training & Evaluation
```bash
# Train model from scratch
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 100 \
  --batch_size 32

# Train with pretrained ResNet18 backbone
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 100 \
  --batch_size 32 \
  --pretrained

# Train with Focal Loss for class imbalance
python src/model_pipeline.py --mode train \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --epochs 80 \
  --batch_size 64 \
  --focal_alpha 0.7 \
  --pretrained

# Evaluate trained model
python src/model_pipeline.py --mode eval \
  --data_csv data/processed/labels.csv \
  --images_dir data/images \
  --checkpoint models/best_model.pth

# Predict single image
python src/model_pipeline.py --mode predict \
  --checkpoint models/best_model.pth \
  --image data/images/KRW_BTC_202505190000.png
```

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY
```

## Architecture Overview

### Data Pipeline Flow
```
Upbit API (ccxt) → SQLite (candles.db) → Image Generator (mplfinance)
→ Dataset (labels.csv + PNG images) → PyTorch Training → Best Model (.pth)
```

### Key Modules

**`src/config.py`**: Configuration loader using dataclasses and YAML. Defines `AppConfig` containing `DataSettings`, `TrainingSettings`, `BacktestSettings`, `ApiSettings`, and `StorageSettings`. All settings are loaded from `configs/config.yaml`.

**`src/upbit_client.py`**: Thin wrapper around ccxt.upbit exchange. Provides `UpbitClient.get_minute_candles()` to fetch OHLCV data with proper timeframe mapping (240 minutes = 4h). Handles timezone conversion between UTC and KST.

**`src/storage.py`**: SQLite persistence layer. `SQLiteCandlesRepository` manages the `candles` table with composite primary key `(market, candle_time_utc)`. Provides `upsert_candles()`, `fetch_candles()`, and `latest_timestamp()` for incremental data collection.

**`src/data_collector.py`**: Entry point for data ingestion. Uses `UpbitClient` to fetch historical candles in batches (200 per request), deduplicates against existing data in SQLite, and implements rate limiting (0.15s between requests). Supports incremental updates by querying the latest stored timestamp per market.

**`src/image_generator.py`**: Generates training samples using sliding windows. For each market:
1. Fetches all candles from SQLite
2. Creates overlapping windows (default: 24 candles with stride=3)
3. Renders mplfinance candlestick charts (224x224 RGB, no axes/labels)
4. Computes binary labels: 1 if `future_6_candles['High'].max() >= entry_price * (1 + target_return + fee_rate)`, else 0
5. Saves metadata to `data/processed/labels.csv`

**`src/model_pipeline.py`**: Complete training/eval/inference pipeline containing:
- `ChartDataset`: PyTorch Dataset with ImageNet normalization and data augmentation for training
- `build_model()`: ResNet18 with custom classification head (512→256→128→1 with dropout)
- `FocalLoss`: Handles class imbalance with alpha and gamma parameters
- `train()`: Training loop with weighted sampling, early stopping, and checkpoint saving
- `evaluate_model()`: Computes AUC, accuracy, precision, recall, confusion matrix

### Configuration Structure

`configs/config.yaml` contains all hyperparameters:
- **data**: symbols (list of KRW markets), interval_minutes (240), window_size (24 candles), lookahead_candles (6 for 24h), target_return (0.05), fee_rate (0.001), stride (3 for overlapping windows)
- **training**: batch_size, learning_rate, weight_decay, epochs, device (cuda/cpu), threshold (0.55 for binary classification)
- **storage**: database_path pointing to SQLite file

### Data Flow Details

**Label Generation Logic**: The system uses the "typical price" of the last candle in a window as the entry point:
```python
entry = (last_candle['High'] + last_candle['Low'] + last_candle['Close']) / 3
target = entry * (1 + target_return + fee_rate)  # 1.051 for 5% + 0.1% fee
label = 1 if future['High'].max() >= target else 0
```

**Image Path Handling**: `ChartDataset` includes sophisticated path resolution to avoid double-prefixing when image paths in CSV already contain the full path. It checks if paths are absolute, relative, or already include the `images_dir` prefix.

**Weighted Sampling**: Training uses `WeightedRandomSampler` to address class imbalance (typically label=1 is rare). Weights are computed as inverse class frequency.

## Development Guidelines

### Adding New Markets
1. Edit `configs/config.yaml` and add symbol to `data.symbols` (format: `KRW-<COIN>`)
2. Run data collector: `python src/data_collector.py --config configs/config.yaml --symbols KRW-NEWCOIN`
3. Generate images: `python src/image_generator.py --config configs/config.yaml --symbols KRW-NEWCOIN`
4. Retrain or fine-tune model with new data

### Modifying Windowing Strategy
Key parameters in `configs/config.yaml`:
- `window_size`: Number of candles per training sample (default: 24 = 4 days of 4h candles)
- `lookahead_candles`: Prediction horizon (default: 6 = 24 hours)
- `stride`: Step size for sliding window (default: 3, meaning windows overlap)

After changing these, regenerate all images with `--clean-output` flag.

### Handling Class Imbalance
Two strategies are implemented:
1. **WeightedRandomSampler** in `get_dataloaders()` (enabled by default)
2. **Focal Loss** via `--focal_alpha` parameter during training (alternative to BCE loss)

Choose based on your data distribution. Check `labels.csv` for class ratio first.

### Model Architecture Changes
The model is defined in `model_pipeline.py:build_model()`. To modify:
- Backbone: Change `models.resnet18` to resnet34/resnet50 for more capacity
- Head layers: Adjust FC dimensions or dropout rates via function parameters
- Frozen layers: Currently all layers are trainable; add backbone freezing for faster initial training

### Incremental Data Updates
The data collector supports incremental updates:
- `SQLiteCandlesRepository.latest_timestamp()` returns the most recent candle per market
- New data collection starts from `latest_timestamp + interval` and works forward
- Use `--max-batches` to limit API calls during frequent updates

### GPU/CPU Selection
Model automatically uses CUDA if available. Override via `configs/config.yaml`:
```yaml
training:
  device: cuda  # or 'cpu'
```

## Common Issues

**"Image not found" errors**: The image paths in `labels.csv` must be relative to `images_dir` parameter or absolute. If regenerating images, ensure `--clean-output` is used to remove old files.

**Class imbalance warnings**: If label=1 ratio is < 10%, enable Focal Loss with `--focal_alpha 0.7` or verify `use_weighted_sampler=True` in `get_dataloaders()`.

**CUDA out of memory**: Reduce `batch_size` in config or training command, or reduce `image_size` in config (requires regenerating images).

**Data collector stops early**: Check `--max-batches` parameter. For full historical data, set it to a high value (e.g., 1000) on first run.

## File Naming Conventions

- **Images**: `{market_with_underscores}_{YYYYMMDDHHMM}.png` (e.g., `KRW_BTC_202505190000.png`)
- **Checkpoints**: `best_model.pth` saved in `models/` directory based on validation AUC
- **Data**: SQLite schema uses ISO format timestamps for `candle_time_utc` and `candle_time_kst`

## Testing

Currently no automated tests exist. When adding tests:
- Place in `tests/` directory mirroring `src/` structure
- Use `pytest` for execution
- Mock `UpbitClient` API calls to avoid rate limits
- Test key components: window generation logic, label computation, data loading

## Notes

- The project is currently focused on Korean won (KRW) markets on Upbit
- Data collection implements simple rate limiting (0.15s delay) - adjust if hitting API limits
- Model checkpoints only save the best validation AUC model, not intermediate epochs
- Image generation is CPU-intensive; consider multiprocessing for large datasets (not currently implemented)
