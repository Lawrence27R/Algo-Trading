"""
Data loading utilities for the ML module.

Provides three data sources:
  1. ``fetch_yfinance`` — downloads data via yfinance
  2. ``load_csv``       — loads a local CSV file
  3. ``get_sample_data``— generates synthetic OHLCV data for testing
"""

import numpy as np
import pandas as pd


class DataLoader:
    """Fetches or generates OHLCV market data for ML pipelines."""

    @staticmethod
    def fetch_yfinance(symbol: str, period: str = "2y",
                       interval: str = "1d") -> pd.DataFrame:
        """
        Download historical OHLCV data from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"AAPL"``, ``"RELIANCE.NS"``).
        period : str
            Look-back period accepted by yfinance (``"1y"``, ``"2y"``, …).
        interval : str
            Bar interval (``"1d"``, ``"1h"``, …).

        Returns
        -------
        pd.DataFrame
            DataFrame with lowercase columns: open, high, low, close, volume.

        Raises
        ------
        RuntimeError
            If yfinance is not installed or the download fails.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from exc

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise RuntimeError(
                    f"yfinance returned no data for {symbol!r} "
                    f"(period={period}, interval={interval})."
                )
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df = df.dropna()
            return df
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch data for {symbol!r}: {exc}"
            ) from exc

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        """
        Load OHLCV data from a local CSV file.

        The CSV must contain at minimum columns named (case-insensitive):
        open, high, low, close, volume.  A ``date`` or ``datetime`` column
        is used as the index when present.

        Parameters
        ----------
        path : str
            Absolute or relative path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV DataFrame.
        """
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]

        # Use date column as index if available
        for date_col in ("date", "datetime", "timestamp", "time"):
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                break

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}"
            )

        return df[["open", "high", "low", "close", "volume"]].dropna()

    @staticmethod
    def get_sample_data(symbol: str, n_days: int = 500) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data (random walk) for testing.

        Parameters
        ----------
        symbol : str
            Used only to seed the random number generator for
            reproducible results per symbol.
        n_days : int
            Number of trading days to generate.

        Returns
        -------
        pd.DataFrame
            Synthetic OHLCV DataFrame with a DatetimeIndex.
        """
        rng = np.random.default_rng(seed=abs(hash(symbol)) % (2 ** 31))
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
        actual_n = len(dates)

        # Random-walk close prices
        log_returns = rng.normal(0.0003, 0.015, actual_n)
        close = 1000.0 * np.exp(np.cumsum(log_returns))

        noise = rng.uniform(0.005, 0.02, actual_n)
        high = close * (1 + noise)
        low = close * (1 - noise)
        open_ = close * (1 + rng.uniform(-0.01, 0.01, actual_n))
        volume = rng.integers(100_000, 2_000_000, actual_n).astype(float)

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dates)
        return df
