"""Configuration loader for CandlEye."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataSettings:
    symbols: list[str]
    interval_minutes: int
    window_size: int
    lookahead_candles: int
    image_size: int
    fetch_batch_size: int
    target_return: float
    fee_rate: float
    stride: int


@dataclass
class TrainingSettings:
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    device: str
    threshold: float


@dataclass
class BacktestSettings:
    initial_capital: float
    position_size: float
    stop_loss: float
    take_profit: float


@dataclass
class ApiSettings:
    host: str
    port: int
    reload: bool


@dataclass
class StorageSettings:
    database_path: Path


@dataclass
class AppConfig:
    data: DataSettings
    training: TrainingSettings
    backtest: BacktestSettings
    api: ApiSettings
    storage: StorageSettings
    artifacts_dir: Path


def load_config(path: str | Path) -> AppConfig:
    """Load the application configuration from YAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = yaml.safe_load(handle)
    data = DataSettings(**payload["data"])
    training = TrainingSettings(**payload["training"])
    backtest = BacktestSettings(**payload["backtest"])
    api = ApiSettings(**payload["api"])
    storage = StorageSettings(database_path=Path(payload["storage"]["database_path"]))
    artifacts_dir = Path(payload["artifacts_dir"])
    return AppConfig(
        data=data,
        training=training,
        backtest=backtest,
        api=api,
        storage=storage,
        artifacts_dir=artifacts_dir,
    )
