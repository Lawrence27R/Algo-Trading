"""
ML-powered trading strategy.

Extends BaseStrategy to generate BUY / SELL / HOLD signals using
a pre-trained MLPredictor model.
"""

import pandas as pd

from strategies.base_strategy import BaseStrategy
from ml.predictor import MLPredictor


class MLStrategy(BaseStrategy):
    """Trading strategy driven by an ML classifier."""

    def __init__(self, symbol: str, quantity: int,
                 model_type: str = "random_forest",
                 confidence_threshold: float = 0.55):
        """
        Parameters
        ----------
        symbol : str
            Ticker symbol (must match a saved model).
        quantity : int
            Number of shares per order.
        model_type : str
            ``"random_forest"`` or ``"xgboost"``.
        confidence_threshold : float
            Minimum probability to emit a directional signal.
        """
        super().__init__(symbol, quantity)
        self.model_type = model_type
        self.predictor = MLPredictor(confidence_threshold=confidence_threshold)

    def generate_signal(self, data: pd.DataFrame) -> str:
        """
        Generate a trading signal from recent OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            Recent market data (at least 60 rows of OHLCV).

        Returns
        -------
        str
            ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        """
        try:
            result = self.predictor.predict(data, self.symbol, self.model_type)
            return result["signal"]
        except FileNotFoundError:
            return "HOLD"
        except Exception:
            return "HOLD"
