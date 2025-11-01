"""Generate training images and labels from stored candle data."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from config import AppConfig, load_config
from labeling_strategies import compute_risk_based_label, compute_4class_label
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
    # 하위 디렉토리의 PNG 파일도 재귀적으로 삭제
    for path in image_dir.glob("**/*.png"):
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


def render_window(window: pd.DataFrame, output_path: Path, image_size: int, mav: tuple[int, ...] | None = None) -> None:
    market_colors = mpf.make_marketcolors(up="red", down="blue", edge="inherit", wick="inherit")
    style = mpf.make_mpf_style(marketcolors=market_colors, gridstyle=" ", facecolor="#111111")
    dpi = 100
    figsize = (image_size / dpi, image_size / dpi * 1.2)
    fig, axes = mpf.plot(
        window,
        type="candle",
        volume=True,
        mav=mav,  # 이동평균선 추가
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
        ax.set_xlabel('')  # x축 레이블 제거
        ax.set_ylabel('')  # y축 레이블 제거 (Price, Volume 텍스트)
        if hasattr(ax, "yaxis"):
            ax.yaxis.get_offset_text().set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)




def generate_samples(
    df: pd.DataFrame,
    market: str,
    config: AppConfig,
    image_dir: Path,
) -> List[dict]:
    window_size = config.data.window_size
    lookahead = config.data.lookahead_candles
    stride = max(1, config.data.stride)

    # 이동평균선 설정
    moving_averages = config.data.moving_averages
    max_ma = max(moving_averages) if moving_averages else 0
    mav_tuple = tuple(moving_averages) if moving_averages else None

    # 심볼별 하위 디렉토리 생성
    market_dir = image_dir / market.replace('-', '_')
    market_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[dict] = []

    # MA가 유효한 구간부터 시작
    # window_with_ma의 모든 봉에 MA가 표시되려면
    # window 시작 이전에 max_ma 데이터가 추가로 필요
    start_offset = max_ma * 2 if max_ma > 0 else 0
    total_windows = len(df) - start_offset - (window_size + lookahead) + 1
    if total_windows <= 0:
        logger.warning("Not enough data points to create a window for %s (need at least %s candles for MA)",
                      market, start_offset + window_size + lookahead)
        return metadata

    if moving_averages:
        logger.info(
            "Processing %s windows for %s (window=%s, lookahead=%s, stride=%s, MA=%s)",
            max(0, (total_windows + stride - 1) // stride),
            market,
            window_size,
            lookahead,
            stride,
            moving_averages,
        )
    else:
        logger.info(
            "Processing %s windows for %s (window=%s, lookahead=%s, stride=%s)",
            max(0, (total_windows + stride - 1) // stride),
            market,
            window_size,
            lookahead,
            stride,
        )

    for start_idx in range(start_offset, len(df) - window_size - lookahead + 1, stride):
        end_idx = start_idx + window_size

        # MA 계산을 위한 확장 window (MA 기간만큼 이전 데이터 포함)
        ma_start_idx = start_idx - max_ma if max_ma > 0 else start_idx
        window_with_ma = df.iloc[ma_start_idx:end_idx]

        # 메타데이터 및 라벨링용 window (실제 24개)
        window = df.iloc[start_idx:end_idx]
        future = df.iloc[end_idx : end_idx + lookahead]

        if window.empty or future.empty:
            continue

        # 연도별 하위 디렉토리에 저장
        year = window.index[-1].year
        year_dir = market_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        output_path = year_dir / f"{market.replace('-', '_')}_{window.index[-1].strftime('%Y%m%d%H%M')}.png"

        # 차트 생성: 확장된 window로 MA 계산 (차트에는 더 많은 캔들 표시됨)
        render_window(window_with_ma, output_path, config.data.image_size, mav=mav_tuple)

        # 4-class 라벨 계산 (손절/익절 고려, 수익률 구간별 분류)
        label = compute_4class_label(
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
        # CSV 파일도 초기화
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info("Removed existing metadata file: %s", metadata_path)

    symbols = args.symbols or config.data.symbols
    total_samples = 0
    first_symbol = True
    first_image_copied = False
    sample_dir = Path("data/samples")

    for market in symbols:
        records = repository.fetch_candles(market)
        if not records:
            logger.warning("No data found for %s. Run data_collector first.", market)
            continue
        df = build_dataframe(records)
        samples = generate_samples(df, market, config, image_dir)

        if samples:
            # 처음 만든 1장의 이미지를 data/samples에 복사
            if not first_image_copied:
                sample_dir.mkdir(parents=True, exist_ok=True)
                first_image_path = Path(samples[0]["image_path"])
                sample_image_path = sample_dir / first_image_path.name
                shutil.copy(first_image_path, sample_image_path)
                logger.info("Copied first sample image to %s", sample_image_path)
                first_image_copied = True

            # 각 심볼 완료 후 바로 CSV에 저장
            samples_df = pd.DataFrame(samples)
            if first_symbol:
                # 첫 번째 심볼: 새로 생성 (header 포함)
                samples_df.to_csv(metadata_path, index=False, mode='w')
                first_symbol = False
            else:
                # 이후 심볼: append 모드 (header 없이)
                samples_df.to_csv(metadata_path, index=False, mode='a', header=False)

            total_samples += len(samples)
            logger.info("Generated %s samples for %s | Total: %s samples",
                       len(samples), market, total_samples)
            logger.info("Updated metadata file: %s", metadata_path)

    if total_samples == 0:
        logger.warning("No samples generated.")
        return 0

    logger.info("All done! Total %s samples saved to %s", total_samples, metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
