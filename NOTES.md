# Implementation Notes

- Established the FastAPI + PyTorch scaffold with foundational directories, `configs/config.yaml`, and `.env.example` for environment secrets.
- Implemented configuration loading (`src/config.py`) and an SQLite persistence layer (`src/storage.py`) to manage candle schemas and queries.
- Added a ccxt-powered Upbit data collector (`src/data_collector.py`, `src/upbit_client.py`) that reads dotenv credentials, fetches seven KRW markets, and writes into SQLite.
- Built the image and label generator (`src/image_generator.py`) that renders mplfinance chart panels, labels future target hits, supports stride-based sampling, and exports metadata CSV.
- Authored `AGENTS.md` to document repository guidelines for contributors.
