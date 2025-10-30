"""Command-line entrypoint for ingesting Upbit candles into SQLite."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv

from config import AppConfig, load_config
from storage import CandleRecord, SQLiteCandlesRepository
from upbit_client import UpbitClient, UpbitCredentials


logger = logging.getLogger(__name__)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CandlEye data collector")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Markets to fetch (default: use config.data.symbols).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum number of API batches to request per symbol.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing data and collect from scratch (useful when changing interval_minutes).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return load_config(config_path)


def build_upbit_client() -> UpbitClient:
    load_dotenv()
    access_key = os.getenv("UPBIT_ACCESS_KEY")
    secret_key = os.getenv("UPBIT_SECRET_KEY")
    credentials = UpbitCredentials(access_key=access_key, secret_key=secret_key)
    return UpbitClient(credentials=credentials)


def candle_records_from_payload(market: str, payload: Iterable[dict]) -> List[CandleRecord]:
    records: List[CandleRecord] = []
    for candle in payload:
        utc_dt = UpbitClient.parse_candle_timestamp(candle["candle_date_time_utc"])
        kst_dt = datetime.fromisoformat(candle["candle_date_time_kst"])
        records.append(
            CandleRecord(
                market=market,
                candle_time_utc=utc_dt,
                candle_time_kst=kst_dt,
                opening_price=float(candle["opening_price"]),
                high_price=float(candle["high_price"]),
                low_price=float(candle["low_price"]),
                trade_price=float(candle["trade_price"]),
                timestamp=int(candle["timestamp"]),
                candle_acc_trade_price=float(candle["candle_acc_trade_price"]),
                candle_acc_trade_volume=float(candle["candle_acc_trade_volume"]),
            )
        )
    return records


def synchronize_market(
    client: UpbitClient,
    repository: SQLiteCandlesRepository,
    market: str,
    interval_minutes: int,
    fetch_batch_size: int,
    max_batches: int,
    force: bool = False,
) -> int:
    """Fetch up to max_batches of candles for a single market."""
    if force:
        newest_stored = None
        logger.info("Syncing %s (force mode: ignoring existing data)", market)
    else:
        newest_stored = repository.latest_timestamp(market)
        logger.info("Syncing %s (latest stored: %s)", market, newest_stored or "none")
    total_new = 0
    cursor_to: datetime | None = None
    batches = 0

    while batches < max_batches:
        batches += 1
        candles = client.get_minute_candles(
            market=market,
            unit=interval_minutes,
            count=fetch_batch_size,
            to=cursor_to,
        )
        if not candles:
            logger.info("No candles returned for %s, stopping.", market)
            break

        candles_sorted = sorted(
            candles,
            key=lambda item: item["candle_date_time_utc"],
        )
        new_payload = []
        for candle in candles_sorted:
            utc_dt = UpbitClient.parse_candle_timestamp(candle["candle_date_time_utc"])
            if newest_stored and utc_dt <= newest_stored:
                continue
            new_payload.append(candle)

        if not new_payload:
            logger.info("Reached existing data for %s, stopping.", market)
            break

        records = candle_records_from_payload(market, new_payload)
        inserted = repository.upsert_candles(records)
        total_new += inserted
        logger.info(
            "Inserted %s new candles for %s (batch %s)",
            inserted,
            market,
            batches,
        )

        earliest = UpbitClient.parse_candle_timestamp(candles_sorted[0]["candle_date_time_utc"])
        cursor_to = earliest - timedelta(minutes=interval_minutes)

        if newest_stored and inserted == 0:
            logger.debug("No new candles beyond existing data for %s, stopping.", market)
            break

        time.sleep(0.15)  # simple rate limiting

        if inserted < fetch_batch_size:
            logger.debug("Batch smaller than fetch size for %s, likely caught up.", market)
            break

    logger.info("Finished syncing %s. Total new candles: %s", market, total_new)
    return total_new


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        config = load_app_config(args.config)
    except Exception as exc:  # pragma: no cover - configuration errors bubble up
        logger.error("Failed to load config: %s", exc)
        return 1

    client = build_upbit_client()
    repository = SQLiteCandlesRepository(config.storage.database_path)
    symbols = args.symbols or config.data.symbols

    if not symbols:
        logger.warning("No symbols specified. Nothing to synchronize.")
        return 0

    total_inserted = 0
    for market in symbols:
        inserted = synchronize_market(
            client=client,
            repository=repository,
            market=market,
            interval_minutes=config.data.interval_minutes,
            fetch_batch_size=config.data.fetch_batch_size,
            max_batches=args.max_batches,
            force=args.force,
        )
        total_inserted += inserted

    logger.info("Inserted %s candles across %s markets.", total_inserted, len(symbols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
