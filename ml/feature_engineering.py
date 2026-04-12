"""
Feature engineering module for ML-based trading strategies.

Computes technical indicators from OHLCV data and creates
binary classification target labels for next-day direction prediction.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Transforms raw OHLCV market data into ML-ready features."""

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicator features and binary target label.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: open, high, low, close, volume.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame (NaNs dropped) with an additional
            ``label`` column (1 = next day up, 0 = next day down/flat).
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ── RSI (14-period) ────────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ── MACD ──────────────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # ── Bollinger Bands ────────────────────────────────────────────────────
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        df["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        # ── EMAs ──────────────────────────────────────────────────────────────
        df["ema_9"] = close.ewm(span=9, adjust=False).mean()
        df["ema_21"] = close.ewm(span=21, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()

        # ── Volume MA ratio ────────────────────────────────────────────────────
        vol_ma = volume.rolling(20).mean()
        df["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

        # ── Returns ───────────────────────────────────────────────────────────
        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)

        # ── High-low range / close ─────────────────────────────────────────────
        df["hl_range"] = (high - low) / close.replace(0, np.nan)

        # ── ATR (14-period) ────────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # ── Rolling volatility (10-day std of returns) ─────────────────────────
        df["volatility"] = df["return_1d"].rolling(10).std()

        # ── Target label: 1 if next-day close > today's close else 0 ──────────
        df["label"] = (close.shift(-1) > close).astype(int)

        # Drop rows with NaN values
        df = df.dropna()

        return df

    @property
    def feature_columns(self):
        """Return the list of feature column names (excludes OHLCV + label)."""
        return [
            "rsi",
            "macd_line", "macd_signal", "macd_hist",
            "bb_width", "bb_pct_b",
            "ema_9", "ema_21", "ema_50",
            "volume_ratio",
            "return_1d", "return_5d",
            "hl_range",
            "atr",
            "volatility",
        ]
