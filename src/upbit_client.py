"""Thin wrapper around the Upbit exchange using ccxt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import ccxt


API_BASE_URL = "https://api.upbit.com"
KST = timezone(timedelta(hours=9))


@dataclass
class UpbitCredentials:
    access_key: Optional[str] = None
    secret_key: Optional[str] = None


class UpbitClient:
    """HTTP client backed by ccxt for interacting with the Upbit API."""

    def __init__(
        self,
        credentials: UpbitCredentials | None = None,
        base_url: str = API_BASE_URL,
        request_timeout: float = 10.0,
    ) -> None:
        self.credentials = credentials or UpbitCredentials()
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self._exchange = ccxt.upbit(
            {
                "apiKey": self.credentials.access_key,
                "secret": self.credentials.secret_key,
                "enableRateLimit": True,
                "timeout": int(self.request_timeout * 1000),
            }
        )
        self._exchange.load_markets()

    def get_minute_candles(
        self,
        market: str,
        unit: int,
        count: int = 200,
        to: datetime | None = None,
    ) -> List[Dict[str, Any]]:
        """Fetch minute candles for the given market."""
        if count < 1 or count > 200:
            raise ValueError("count must be between 1 and 200 per Upbit API limits")
        timeframe = self._minutes_to_timeframe(unit)
        symbol = self._to_ccxt_symbol(market)
        if symbol not in self._exchange.symbols:
            raise ValueError(f"Market {market} not available on Upbit (symbol {symbol})")

        since_ms: Optional[int] = None
        if to is not None:
            to_utc = to.astimezone(timezone.utc)
            window = timedelta(minutes=unit * count)
            since_candidate = to_utc - window
            since_ms = int(since_candidate.timestamp() * 1000)

        ohlcv = self._exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=count,
            params={"price": "trade"},
        )

        candles: List[Dict[str, Any]] = []
        for entry in ohlcv:
            ts_ms, open_price, high_price, low_price, close_price, volume = entry
            dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if to is not None and dt_utc > to:
                continue
            dt_kst = dt_utc.astimezone(KST).replace(tzinfo=None)
            candles.append(
                {
                    "market": market,
                    "candle_date_time_utc": dt_utc.isoformat(),
                    "candle_date_time_kst": dt_kst.isoformat(),
                    "opening_price": open_price,
                    "high_price": high_price,
                    "low_price": low_price,
                    "trade_price": close_price,
                    "timestamp": int(ts_ms),
                    "candle_acc_trade_price": close_price * volume,
                    "candle_acc_trade_volume": volume,
                }
            )
        return candles

    @staticmethod
    def parse_candle_timestamp(value: str) -> datetime:
        """Parse an Upbit UTC timestamp string."""
        # Upbit returns values such as "2021-09-01T00:00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _minutes_to_timeframe(minutes: int) -> str:
        mapping = {
            1: "1m",
            3: "3m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "1h",
            240: "4h",
            1440: "1d",
            10080: "1w",
            43200: "1M",
        }
        try:
            return mapping[minutes]
        except KeyError as exc:
            raise ValueError(f"Unsupported timeframe for {minutes} minutes") from exc

    @staticmethod
    def _to_ccxt_symbol(market: str) -> str:
        """Convert Upbit market name (e.g., KRW-BTC) to ccxt symbol (BTC/KRW)."""
        parts = market.split("-")
        if len(parts) != 2:
            raise ValueError(f"Unexpected market format: {market}")
        quote, base = parts
        return f"{base}/{quote}"
