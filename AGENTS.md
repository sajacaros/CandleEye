# Repository Guidelines

## Project Structure & Module Organization
Primary application code lives in `src/` (`data_collector.py`, `image_generator.py`, `dataset.py`, `train.py`, `backtest.py`, `api.py`). Data artifacts sit in `data/` (`raw/`, `images/`, `processed/`), models in `models/` (`checkpoints/`, `best_model.pth`), shared configuration in `configs/config.yaml`, and exploratory work in `notebooks/`. Clean heavy outputs in `data/images` and `models/checkpoints` before opening a pull request.

## Build, Test, and Development Commands
- `python src/data_collector.py --config configs/config.yaml` — ingest OHLCV snapshots into `data/raw`.
- `python src/image_generator.py --config configs/config.yaml` — render candlestick panels into `data/images`.
- `python src/train.py --config configs/config.yaml` — train the ResNet18 classifier and store checkpoints under `models/`.
- `python src/backtest.py --config configs/config.yaml` — replay trading decisions on historical windows.
- `uvicorn src.api:app --reload` — run the FastAPI inference service for local checks.
- `pytest` — execute the automated test suite.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, descriptive `snake_case` functions, and `CamelCase` classes. Type hint public interfaces and keep docstrings focused on trading assumptions or chart transforms. When available, run `black src tests` and `ruff check src tests` before requesting review; if an exception is unavoidable, note the reason inline.

## Testing Guidelines
Mirror the `src/` layout in `tests/` (e.g., `tests/test_dataset.py`, `tests/pipelines/test_train.py`). Use `pytest` parametrization to exercise symbols, windows, and thresholds, and mock external APIs so runs stay deterministic. Prioritize coverage for preprocessing transforms, training callbacks, and decision thresholds, then validate with `pytest --maxfail=1 --disable-warnings`.

## Commit & Pull Request Guidelines
Write commits in an imperative tone using Conventional Commit prefixes (`feat`, `fix`, `docs`, `chore`) to signal scope (e.g., `feat: add risk-adjusted thresholding`). Reference related issues or experiments in the body and list required follow-up steps such as regenerating `data/images`. Pull requests should include a concise summary, verification evidence, and call out config changes. Keep each PR focused; separate training, API, and dashboard adjustments when practical.

## Security & Configuration Tips
Keep API tokens (Upbit, Redis, W&B) in a local `.env` or secrets manager and load them via environment variables; never commit credentials or notebooks containing them. Treat `configs/config.yaml` as the source of truth and document temporary overrides in the PR description. Clear notebook outputs before sharing (`jupyter nbconvert --clear-output`) and review dependencies before publishing production artifacts.
