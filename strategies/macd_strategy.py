import pandas as pd
import ta
from strategies.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    def __init__(self, symbol, quantity, fast=12, slow=26, signal=9):
        super().__init__(symbol, quantity)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signal(self, data: pd.DataFrame):
        data = data.copy()
        macd_indicator = ta.trend.MACD(
            data['close'],
            window_fast=self.fast,
            window_slow=self.slow,
            window_sign=self.signal
        )
        data['macd'] = macd_indicator.macd()
        data['macd_signal'] = macd_indicator.macd_signal()
        data = data.dropna()
        if len(data) < 2:
            return 'HOLD'
        last = data.iloc[-1]
        prev = data.iloc[-2]
        if prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']:
            return 'BUY'
        elif prev['macd'] > prev['macd_signal'] and last['macd'] < last['macd_signal']:
            return 'SELL'
        return 'HOLD'
