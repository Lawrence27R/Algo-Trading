import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    def __init__(self, symbol: str, quantity: int):
        self.symbol = symbol
        self.quantity = quantity

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> str:
        """
        Generate a trading signal based on market data.
        Returns 'BUY', 'SELL', or 'HOLD'.
        """
        pass
