"""Risk-based labeling for chart-based trading signal generation."""

from __future__ import annotations

import pandas as pd


def compute_risk_based_label(
    window: pd.DataFrame,
    future: pd.DataFrame,
    target_return: float,
    fee_rate: float,
) -> int:
    """Compute risk-based label considering both stop-loss and take-profit.

    Label = 1 if take-profit is hit before stop-loss, else 0.
    This strategy reflects real trading scenarios where risk management is crucial.

    Args:
        window: DataFrame containing OHLCV data for the current window
        future: DataFrame containing OHLCV data for the lookahead period
        target_return: Target return percentage (e.g., 0.04 for 4%)
        fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)

    Returns:
        Label as integer (0 or 1)
    """
    # 마지막 봉의 종가를 진입가로 사용 (실전에 더 가깝)
    last_candle = window.iloc[-1]
    entry = last_candle["Close"]

    # 목표가와 손절가 계산
    take_profit_price = entry * (1 + target_return + fee_rate)
    # 손절가: -3% (백테스트 설정과 동일)
    stop_loss_price = entry * (1 - 0.03)

    # 시간순으로 확인하여 어느 쪽이 먼저 도달했는지 판단
    for idx, row in future.iterrows():
        # 해당 봉에서 손절가와 익절가 중 어느 쪽이 먼저인지 확인
        # Low가 손절가 이하면 손절 발동
        if row["Low"] <= stop_loss_price:
            # 같은 봉에서 High가 익절가 이상이면 시가 기준으로 판단
            if row["High"] >= take_profit_price:
                # 시가가 손절가와 익절가 사이에 있으면
                # Low가 먼저 찍혔다고 가정 (보수적)
                return 0
            return 0

        # High가 익절가 이상이면 익절 성공
        if row["High"] >= take_profit_price:
            return 1

    # 예측 기간 내에 손절도 익절도 도달하지 않음 → 실패로 간주
    return 0
