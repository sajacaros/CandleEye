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


def compute_4class_label(
    window: pd.DataFrame,
    future: pd.DataFrame,
    target_return: float,
    fee_rate: float,
) -> int:
    """Compute 4-class label based on profit/loss ranges.

    Classes:
    - Class 0: Stop-loss hit (-3% or below) - Strong sell signal
    - Class 1: Small loss/gain (-3% ~ 1%) - Weak signal
    - Class 2: Medium profit (1% ~ target%) - Moderate buy signal
    - Class 3: Take-profit hit (target% or above) - Strong buy signal

    This strategy reflects real trading scenarios with more granular labels.

    Args:
        window: DataFrame containing OHLCV data for the current window
        future: DataFrame containing OHLCV data for the lookahead period
        target_return: Target return percentage (e.g., 0.05 for 5%)
        fee_rate: Trading fee rate (e.g., 0.001 for 0.1%)

    Returns:
        Label as integer (0, 1, 2, or 3)
    """
    # 마지막 봉의 종가를 진입가로 사용 (실전에 더 가깝)
    last_candle = window.iloc[-1]
    entry = last_candle["Close"]

    # 목표가와 손절가 계산
    take_profit_price = entry * (1 + target_return + fee_rate)
    # 손절가: -3% (백테스트 설정과 동일)
    stop_loss_price = entry * (1 - 0.03)

    # 시간순으로 확인하여 손절 또는 익절 중 어느 것이 먼저 도달했는지 판단
    for idx, row in future.iterrows():
        # Low가 손절가 이하면 손절 발동 → Class 0
        if row["Low"] <= stop_loss_price:
            # 같은 봉에서 High가 익절가 이상이면 보수적으로 손절 우선 가정
            if row["High"] >= take_profit_price:
                return 0  # 손절이 먼저 체결되었다고 가정
            return 0  # 손절 체결

        # High가 익절가 이상이면 익절 성공 → Class 3
        if row["High"] >= take_profit_price:
            return 3  # 익절 체결

    # 예측 기간 내에 손절도 익절도 도달하지 않음 → 최종 수익률로 분류
    # 마지막 봉의 종가 기준 수익률 계산
    final_candle = future.iloc[-1]
    final_price = final_candle["Close"]
    final_return = (final_price - entry) / entry

    if final_return < 0.01:
        return 1  # Class 1: -3% ~ 1% (작은 손실 또는 미미한 수익)
    else:
        return 2  # Class 2: 1% ~ 5% (중간 수익)
