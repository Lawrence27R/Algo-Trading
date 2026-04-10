import pandas as pd
import ta
from strategies.base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, symbol, quantity, window=20, std_dev=2):
        super().__init__(symbol, quantity)
        self.window = window
        self.std_dev = std_dev

    def generate_signal(self, data: pd.DataFrame):
        data = data.copy()
        bb = ta.volatility.BollingerBands(data['close'], window=self.window, window_dev=self.std_dev)
        data['bb_high'] = bb.bollinger_hband()
        data['bb_low'] = bb.bollinger_lband()
        data['bb_mid'] = bb.bollinger_mavg()
        data.dropna(inplace=True)
        if len(data) < 1:
            return 'HOLD'
        last = data.iloc[-1]
        if last['close'] <= last['bb_low']:
            return 'BUY'
        elif last['close'] >= last['bb_high']:
            return 'SELL'
        return 'HOLD'
