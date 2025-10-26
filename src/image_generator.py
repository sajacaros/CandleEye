"""Generate training images and labels from stored candle data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from config import AppConfig, load_config
from storage import SQLiteCandlesRepository


logger = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CandlEye image and label generator")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Subset of markets to process (defaults to config values).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/images",
        help="Directory to store generated images.",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/processed/labels.csv",
        help="Path for the labels metadata CSV.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing images in the output directory before generation.",
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


def ensure_directories(image_dir: Path, metadata_path: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)


def clear_output_dir(image_dir: Path) -> None:
    removed = 0
    for path in image_dir.glob("*.png"):
        path.unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.info("Removed %s existing images from %s", removed, image_dir)


def build_dataframe(records) -> pd.DataFrame:
    data = {
        "datetime": [c.candle_time_utc for c in records],
        "Open": [c.opening_price for c in records],
        "High": [c.high_price for c in records],
        "Low": [c.low_price for c in records],
        "Close": [c.trade_price for c in records],
        "Volume": [c.candle_acc_trade_volume for c in records],
    }
    df = pd.DataFrame(data)
    index = pd.DatetimeIndex(df.pop("datetime"))
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    df.index = index
    return df


def render_window(window: pd.DataFrame, output_path: Path, image_size: int) -> None:
    market_colors = mpf.make_marketcolors(up="red", down="blue", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=market_colors, gridstyle=" ", facecolor="#111111")
    dpi = 100
    figsize = (image_size / dpi, image_size / dpi * 1.2)
    fig, axes = mpf.plot(
        window,
        type="candle",
        volume=True,
        style=style,
        tight_layout=True,
        figsize=figsize,
        scale_padding=dict(left=0, right=0, top=0.4, bottom=0.4),
        returnfig=True,
    )
    if isinstance(axes, dict):
        axes_iterable = [ax for ax in axes.values() if ax is not None]
    elif isinstance(axes, (list, tuple)):
        axes_iterable = list(axes)
    else:
        axes_iterable = [axes]
    for ax in axes_iterable:
        ax.tick_params(
            bottom=False,
            left=False,
            right=False,
            top=False,
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if hasattr(ax, "yaxis"):
            ax.yaxis.get_offset_text().set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compute_label(
    window: pd.DataFrame,
    future: pd.DataFrame,
    target_return: float,
    fee_rate: float,
) -> int:
    # 마지막 봉의 평균가 (Typical Price)
    last_candle = window.iloc[-1]
    entry = (last_candle["High"] + last_candle["Low"] + last_candle["Close"]) / 3
    target = entry * (1 + target_return + fee_rate)
    future_high = future["High"].max()
    return int(future_high >= target)


def generate_samples(
    df: pd.DataFrame,
    market: str,
    config: AppConfig,
    image_dir: Path,
) -> List[dict]:
    window_size = config.data.window_size
    lookahead = config.data.lookahead_candles
    stride = max(1, config.data.stride)
    metadata: List[dict] = []
    total_windows = len(df) - (window_size + lookahead) + 1
    if total_windows <= 0:
        logger.warning("Not enough data points to create a window for %s", market)
        return metadata

    logger.info(
        "Processing %s windows for %s (window=%s, lookahead=%s, stride=%s)",
        max(0, (total_windows + stride - 1) // stride),
        market,
        window_size,
        lookahead,
        stride,
    )

    for start_idx in range(0, len(df) - window_size - lookahead + 1, stride):
        end_idx = start_idx + window_size
        window = df.iloc[start_idx:end_idx]
        future = df.iloc[end_idx : end_idx + lookahead]
        if window.empty or future.empty:
            continue
        output_path = image_dir / f"{market.replace('-', '_')}_{window.index[-1].strftime('%Y%m%d%H%M')}.png"
        render_window(window, output_path, config.data.image_size)
        label = compute_label(
            window=window,
            future=future,
            target_return=config.data.target_return,
            fee_rate=config.data.fee_rate,
        )
        metadata.append(
            {
                "market": market,
                "image_path": output_path.as_posix(),
                "label": label,
                "window_start": window.index[0].isoformat(),
                "window_end": window.index[-1].isoformat(),
                "entry_price": window["Close"].iloc[-1],
                "target_price": window["Close"].iloc[-1]
                * (1 + config.data.target_return + config.data.fee_rate),
            }
        )
        if len(metadata) % 100 == 0:
            logger.info("Generated %s samples for %s...", len(metadata), market)
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        config = load_config(args.config)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load config: %s", exc)
        return 1

    repository = SQLiteCandlesRepository(config.storage.database_path)
    image_dir = Path(args.output_dir)
    metadata_path = Path(args.metadata_path)
    ensure_directories(image_dir, metadata_path)
    if args.clean_output:
        clear_output_dir(image_dir)

    symbols = args.symbols or config.data.symbols
    all_metadata: List[dict] = []
    for market in symbols:
        records = repository.fetch_candles(market)
        if not records:
            logger.warning("No data found for %s. Run data_collector first.", market)
            continue
        df = build_dataframe(records)
        samples = generate_samples(df, market, config, image_dir)
        all_metadata.extend(samples)
        logger.info("Generated %s samples for %s", len(samples), market)

    if not all_metadata:
        logger.warning("No samples generated.")
        return 0

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    logger.info("Saved metadata to %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
