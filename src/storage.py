"""SQLite persistence helpers for market data."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List


@dataclass
class CandleRecord:
    market: str
    candle_time_utc: datetime
    candle_time_kst: datetime
    opening_price: float
    high_price: float
    low_price: float
    trade_price: float
    timestamp: int
    candle_acc_trade_price: float
    candle_acc_trade_volume: float


class SQLiteCandlesRepository:
    """Lightweight repository to persist candle data."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    market TEXT NOT NULL,
                    candle_time_utc TEXT NOT NULL,
                    candle_time_kst TEXT NOT NULL,
                    opening_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    trade_price REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    candle_acc_trade_price REAL NOT NULL,
                    candle_acc_trade_volume REAL NOT NULL,
                    PRIMARY KEY (market, candle_time_utc)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_market_time
                ON candles (market, candle_time_utc)
                """
            )

    def latest_timestamp(self, market: str) -> datetime | None:
        """Return the most recent candle_time_utc stored for a market."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT candle_time_utc
                FROM candles
                WHERE market = ?
                ORDER BY candle_time_utc DESC
                LIMIT 1
                """,
                (market,),
            )
            row = cursor.fetchone()
        if row:
            return datetime.fromisoformat(row[0])
        return None

    def upsert_candles(self, candles: Iterable[CandleRecord]) -> int:
        """Insert or replace candle records."""
        records = [
            (
                candle.market,
                candle.candle_time_utc.isoformat(),
                candle.candle_time_kst.isoformat(),
                candle.opening_price,
                candle.high_price,
                candle.low_price,
                candle.trade_price,
                candle.timestamp,
                candle.candle_acc_trade_price,
                candle.candle_acc_trade_volume,
            )
            for candle in candles
        ]
        if not records:
            return 0
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candles (
                    market,
                    candle_time_utc,
                    candle_time_kst,
                    opening_price,
                    high_price,
                    low_price,
                    trade_price,
                    timestamp,
                    candle_acc_trade_price,
                    candle_acc_trade_volume
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
        return len(records)

    def fetch_candles(self, market: str, limit: int | None = None) -> List[CandleRecord]:
        """Fetch candles for a market ordered oldest to newest."""
        query = """
            SELECT
                market,
                candle_time_utc,
                candle_time_kst,
                opening_price,
                high_price,
                low_price,
                trade_price,
                timestamp,
                candle_acc_trade_price,
                candle_acc_trade_volume
            FROM candles
            WHERE market = ?
            ORDER BY candle_time_utc ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params = (market, limit)
        else:
            params = (market,)

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        candles: List[CandleRecord] = []
        for row in rows:
            candles.append(
                CandleRecord(
                    market=row[0],
                    candle_time_utc=datetime.fromisoformat(row[1]),
                    candle_time_kst=datetime.fromisoformat(row[2]),
                    opening_price=row[3],
                    high_price=row[4],
                    low_price=row[5],
                    trade_price=row[6],
                    timestamp=row[7],
                    candle_acc_trade_price=row[8],
                    candle_acc_trade_volume=row[9],
                )
            )
        return candles
