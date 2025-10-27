"""
Backtesting module for CandlEye trading strategy.

Simulates trading based on model predictions with realistic constraints:
- Trading fees (buy + sell)
- Optional slippage modeling
- Stop loss and take profit levels
- Position sizing
- Performance metrics (Sharpe ratio, max drawdown, win rate)

Usage:
    python src/backtester.py --model models/best_model.pth --data data/processed/labels.csv --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import load_config
from model_pipeline import ChartDataset, build_model, get_dataloaders


logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    market: str
    position_size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'target', 'stop_loss', 'timeout'


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    trades: List[Trade]
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float

    def print_summary(self):
        """Print a formatted summary of backtest results."""
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital:   ${self.final_capital:,.2f}")
        print(f"Total Return:    ${self.total_return:,.2f} ({self.total_return_pct:.2f}%)")
        print(f"\nTrades:          {self.num_trades}")
        print(f"Wins:            {self.num_wins} ({self.win_rate:.2f}%)")
        print(f"Losses:          {self.num_losses}")
        print(f"Avg Win:         {self.avg_win:.2f}%")
        print(f"Avg Loss:        {self.avg_loss:.2f}%")
        print(f"\nMax Drawdown:    {self.max_drawdown:.2f}%")
        print(f"Sharpe Ratio:    {self.sharpe_ratio:.3f}")
        print("="*80)


class Backtester:
    """Backtesting engine for CandlEye strategy."""

    def __init__(
        self,
        model,
        device,
        initial_capital: float = 1_000_000,
        position_size: float = 0.1,
        fee_rate: float = 0.001,
        stop_loss: float = -0.03,
        take_profit: float = 0.05,
        slippage: float = 0.0,
        prediction_threshold: float = 0.55,
        max_hold_periods: int = 6,  # 6 candles = 24 hours
    ):
        self.model = model
        self.device = device
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.fee_rate = fee_rate
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.slippage = slippage
        self.prediction_threshold = prediction_threshold
        self.max_hold_periods = max_hold_periods

    def predict_batch(self, dataloader: DataLoader) -> np.ndarray:
        """Get model predictions for a batch of images."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                out = self.model(imgs).view(-1).cpu().numpy()
                preds.extend(out.tolist())
        return np.array(preds)

    def simulate_trade(
        self,
        entry_price: float,
        future_candles: pd.DataFrame,
        market: str,
        entry_time: str,
    ) -> Trade:
        """
        Simulate a single trade given entry and future price data.

        Args:
            entry_price: Entry price (typical price of last window candle)
            future_candles: DataFrame with future candle data (High, Low, Close, datetime)
            market: Symbol name
            entry_time: Entry timestamp

        Returns:
            Trade object with results
        """
        # Apply slippage to entry
        actual_entry = entry_price * (1 + self.slippage)

        # Apply entry fee
        position_value = self.initial_capital * self.position_size
        entry_fee = position_value * self.fee_rate
        shares = (position_value - entry_fee) / actual_entry

        # Simulate holding period
        for i, (idx, candle) in enumerate(future_candles.iterrows()):
            high = candle['High']
            low = candle['Low']
            close = candle['Close']

            # Check take profit (hit during candle)
            if high >= actual_entry * (1 + self.take_profit):
                exit_price = actual_entry * (1 + self.take_profit)
                exit_price *= (1 - self.slippage)  # Apply slippage
                exit_value = shares * exit_price
                exit_fee = exit_value * self.fee_rate
                pnl = exit_value - exit_fee - position_value
                return Trade(
                    entry_time=entry_time,
                    exit_time=str(idx),
                    entry_price=actual_entry,
                    exit_price=exit_price,
                    market=market,
                    position_size=position_value,
                    pnl=pnl,
                    pnl_pct=(pnl / position_value) * 100,
                    exit_reason='target'
                )

            # Check stop loss (hit during candle)
            if low <= actual_entry * (1 + self.stop_loss):
                exit_price = actual_entry * (1 + self.stop_loss)
                exit_price *= (1 - self.slippage)
                exit_value = shares * exit_price
                exit_fee = exit_value * self.fee_rate
                pnl = exit_value - exit_fee - position_value
                return Trade(
                    entry_time=entry_time,
                    exit_time=str(idx),
                    entry_price=actual_entry,
                    exit_price=exit_price,
                    market=market,
                    position_size=position_value,
                    pnl=pnl,
                    pnl_pct=(pnl / position_value) * 100,
                    exit_reason='stop_loss'
                )

            # Timeout - reached max holding period
            if i == len(future_candles) - 1 or i >= self.max_hold_periods - 1:
                exit_price = close * (1 - self.slippage)
                exit_value = shares * exit_price
                exit_fee = exit_value * self.fee_rate
                pnl = exit_value - exit_fee - position_value
                return Trade(
                    entry_time=entry_time,
                    exit_time=str(idx),
                    entry_price=actual_entry,
                    exit_price=exit_price,
                    market=market,
                    position_size=position_value,
                    pnl=pnl,
                    pnl_pct=(pnl / position_value) * 100,
                    exit_reason='timeout'
                )

        # Fallback: exit at last available price
        exit_price = future_candles['Close'].iloc[-1] * (1 - self.slippage)
        exit_value = shares * exit_price
        exit_fee = exit_value * self.fee_rate
        pnl = exit_value - exit_fee - position_value
        return Trade(
            entry_time=entry_time,
            exit_time=str(future_candles.index[-1]),
            entry_price=actual_entry,
            exit_price=exit_price,
            market=market,
            position_size=position_value,
            pnl=pnl,
            pnl_pct=(pnl / position_value) * 100,
            exit_reason='timeout'
        )

    def run(self, df_test: pd.DataFrame, candle_data: dict, images_dir: str, image_size: int = 224) -> BacktestResult:
        """
        Run backtest on test data.

        Args:
            df_test: Test dataframe with metadata
            candle_data: Dict mapping market -> DataFrame of all candle data
            images_dir: Directory with chart images
            image_size: Image resolution

        Returns:
            BacktestResult object
        """
        # Get predictions
        test_ds = ChartDataset(df_test, images_dir=images_dir, image_size=image_size, train=False)
        test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
        predictions = self.predict_batch(test_dl)

        df_test = df_test.copy()
        df_test['prediction'] = predictions

        # Filter for buy signals
        df_signals = df_test[df_test['prediction'] >= self.prediction_threshold].copy()

        logger.info(f"Generated {len(df_signals)} buy signals from {len(df_test)} test samples")

        trades = []
        for idx, row in df_signals.iterrows():
            market = row['market']
            entry_time = row['window_end']
            entry_price = row['entry_price']

            # Get future candle data for this trade
            if market not in candle_data:
                logger.warning(f"No candle data for {market}, skipping")
                continue

            market_df = candle_data[market]

            # Find the position of entry_time in the full candle data
            try:
                entry_dt = pd.to_datetime(entry_time)
                entry_pos = market_df.index.get_indexer([entry_dt], method='nearest')[0]

                # Get future candles (up to max_hold_periods)
                future_start = entry_pos + 1
                future_end = min(entry_pos + 1 + self.max_hold_periods, len(market_df))
                future_candles = market_df.iloc[future_start:future_end]

                if future_candles.empty:
                    logger.debug(f"No future data for trade at {entry_time}, skipping")
                    continue

                trade = self.simulate_trade(entry_price, future_candles, market, entry_time)
                trades.append(trade)

            except Exception as e:
                logger.warning(f"Error simulating trade: {e}")
                continue

        # Calculate metrics
        return self._compute_metrics(trades)

    def _compute_metrics(self, trades: List[Trade]) -> BacktestResult:
        """Compute aggregated metrics from trade list."""
        if not trades:
            return BacktestResult(
                trades=trades,
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0,
                total_return_pct=0,
                num_trades=0,
                num_wins=0,
                num_losses=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                max_drawdown=0,
                sharpe_ratio=0,
            )

        total_pnl = sum(t.pnl for t in trades)
        final_capital = self.initial_capital + total_pnl

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = (num_wins / len(trades)) * 100 if trades else 0

        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

        # Max drawdown
        cumulative = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / self.initial_capital * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Sharpe ratio (simplified, assuming 252 trading days)
        returns = np.array([t.pnl_pct for t in trades])
        if len(returns) > 1:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            trades=trades,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_pnl,
            total_return_pct=(total_pnl / self.initial_capital) * 100,
            num_trades=len(trades),
            num_wins=num_wins,
            num_losses=num_losses,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
        )


def load_candle_data(config, symbols: List[str]) -> dict:
    """Load full candle data for backtesting."""
    from storage import SQLiteCandlesRepository

    repository = SQLiteCandlesRepository(config.storage.database_path)
    candle_data = {}

    for market in symbols:
        records = repository.fetch_candles(market)
        if not records:
            logger.warning(f"No data for {market}")
            continue

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
        candle_data[market] = df

    return candle_data


def parse_args():
    parser = argparse.ArgumentParser(description="CandlEye Backtester")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, default='data/processed/labels.csv', help='Path to labels CSV')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--images-dir', type=str, default='data/images', help='Images directory')
    parser.add_argument('--threshold', type=float, default=0.55, help='Prediction threshold')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load config
    config = load_config(args.config)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(pretrained=False)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    logger.info(f"Loaded model from {args.model}")

    # Load test data (using time-based split)
    _, _, _, df_test = get_dataloaders(
        args.data,
        args.images_dir,
        batch_size=32,
        time_based_split=True
    )
    logger.info(f"Loaded {len(df_test)} test samples")

    # Load candle data
    symbols = df_test['market'].unique().tolist()
    candle_data = load_candle_data(config, symbols)
    logger.info(f"Loaded candle data for {len(candle_data)} symbols")

    # Initialize backtester
    backtester = Backtester(
        model=model,
        device=device,
        initial_capital=config.backtest.initial_capital,
        position_size=config.backtest.position_size,
        fee_rate=config.data.fee_rate,
        stop_loss=config.backtest.stop_loss,
        take_profit=config.backtest.take_profit,
        slippage=0.0005,  # 0.05% slippage
        prediction_threshold=args.threshold,
        max_hold_periods=config.data.lookahead_candles,
    )

    # Run backtest
    logger.info("Starting backtest...")
    result = backtester.run(df_test, candle_data, args.images_dir, config.data.image_size)

    # Print results
    result.print_summary()

    # Print trade details
    if result.trades:
        print("\nSample Trades (first 10):")
        print("-" * 80)
        for trade in result.trades[:10]:
            print(f"{trade.market} | Entry: ${trade.entry_price:.2f} @ {trade.entry_time}")
            print(f"  Exit: ${trade.exit_price:.2f} @ {trade.exit_time} ({trade.exit_reason})")
            print(f"  PnL: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)")


if __name__ == '__main__':
    main()
