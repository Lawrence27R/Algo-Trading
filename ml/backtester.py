"""
Walk-forward backtesting module using ML signals.

Simulates a simple long-only equity strategy where:
  - BUY signal (label=1)  → enter/hold position
  - SELL signal (label=0) → exit position

Metrics computed: total return, max drawdown, Sharpe ratio, win rate,
total trades, per-window breakdown.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

from ml.feature_engineering import FeatureEngineer
from ml.model_trainer import ModelTrainer


class MLBacktester:
    """Walk-forward ML backtester."""

    #: Fraction of available cash to deploy per BUY signal.
    POSITION_SIZE_FRACTION: float = 0.95

    def __init__(self, train_window: int = 252, test_window: int = 63):
        """
        Parameters
        ----------
        train_window : int
            Number of trading days used to train each rolling window.
        test_window : int
            Number of trading days in the out-of-sample test period.
        """
        self.train_window = train_window
        self.test_window = test_window
        self.fe = FeatureEngineer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, symbol: str,
            initial_cash: float = 100_000,
            model_type: str = "random_forest") -> dict:
        """
        Execute a walk-forward backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame sorted by date ascending.
        symbol : str
            Ticker (used for model naming within each window).
        initial_cash : float
            Starting portfolio cash.
        model_type : str
            ``"random_forest"`` or ``"xgboost"``.

        Returns
        -------
        dict
            Keys: ``trade_log``, ``equity_curve``, ``metrics``, ``windows``.
        """
        featured = self.fe.build_features(df)
        feat_cols = self.fe.feature_columns

        if len(featured) < self.train_window + self.test_window:
            raise ValueError(
                f"Not enough data for walk-forward backtest: {len(featured)} rows "
                f"available, need at least {self.train_window + self.test_window}."
            )

        cash = initial_cash
        position = 0          # number of shares held
        entry_price = 0.0

        trade_log: list[dict] = []
        equity_curve: list[dict] = []
        windows: list[dict] = []

        start = 0
        while start + self.train_window + self.test_window <= len(featured):
            train_end = start + self.train_window
            test_end = min(train_end + self.test_window, len(featured))

            train_df = featured.iloc[start:train_end]
            test_df = featured.iloc[train_end:test_end]

            # Train a fresh model on this window's training slice
            trainer = ModelTrainer(symbol=f"{symbol}_wf_{start}",
                                   model_type=model_type)
            try:
                window_metrics = trainer.train(
                    df.iloc[start: train_end + 50]  # provide raw OHLCV slice
                )
            except Exception:
                start += self.test_window
                continue

            bundle = trainer.load_model()
            model = bundle["model"]
            scaler = bundle["scaler"]

            # Out-of-sample simulation
            X_test = test_df[feat_cols].values
            X_test_s = scaler.transform(X_test)
            predictions = model.predict(X_test_s)

            window_trades = 0
            window_wins = 0

            for i, (idx, row) in enumerate(test_df.iterrows()):
                signal = int(predictions[i])
                price = float(row["close"])
                date_str = str(idx)[:10]

                if signal == 1 and position == 0:
                    # BUY
                    qty = max(1, int(cash * self.POSITION_SIZE_FRACTION // price))
                    if qty > 0 and cash >= qty * price:
                        position = qty
                        entry_price = price
                        cash -= qty * price
                        trade_log.append({
                            "date": date_str,
                            "signal": "BUY",
                            "price": round(price, 4),
                            "quantity": qty,
                            "pnl": 0.0,
                        })
                        window_trades += 1

                elif signal == 0 and position > 0:
                    # SELL
                    pnl = (price - entry_price) * position
                    cash += position * price
                    trade_log.append({
                        "date": date_str,
                        "signal": "SELL",
                        "price": round(price, 4),
                        "quantity": position,
                        "pnl": round(pnl, 4),
                    })
                    if pnl > 0:
                        window_wins += 1
                    window_trades += 1
                    position = 0
                    entry_price = 0.0

                portfolio_value = cash + position * price
                equity_curve.append({"date": date_str,
                                     "value": round(portfolio_value, 2)})

            windows.append({
                "start": str(test_df.index[0])[:10],
                "end": str(test_df.index[-1])[:10],
                "accuracy": window_metrics.get("accuracy"),
                "f1": window_metrics.get("f1"),
                "trades": window_trades,
                "wins": window_wins,
            })

            start += self.test_window

        # Close any open position at the last known price
        if position > 0 and len(featured) > 0:
            last_price = float(featured["close"].iloc[-1])
            pnl = (last_price - entry_price) * position
            cash += position * last_price
            trade_log.append({
                "date": str(featured.index[-1])[:10],
                "signal": "SELL",
                "price": round(last_price, 4),
                "quantity": position,
                "pnl": round(pnl, 4),
            })

        final_value = cash
        metrics = self._compute_metrics(
            equity_curve, trade_log, initial_cash, final_value
        )

        return {
            "trade_log": trade_log,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "windows": windows,
        }

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, equity_curve: list[dict],
                         trade_log: list[dict],
                         initial_cash: float,
                         final_value: float) -> dict[str, Any]:
        total_return = round((final_value - initial_cash) / initial_cash * 100, 2)

        # Max drawdown
        values = [e["value"] for e in equity_curve]
        max_drawdown = 0.0
        if values:
            peak = values[0]
            for v in values:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak * 100 if peak > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd

        # Sharpe ratio (annualised, assuming 252 trading days)
        sharpe = 0.0
        if len(values) > 1:
            returns = [(values[i] - values[i - 1]) / values[i - 1]
                       for i in range(1, len(values))
                       if values[i - 1] > 0]
            if len(returns) > 1:
                mean_r = np.mean(returns)
                std_r = np.std(returns, ddof=1)
                if std_r > 0:
                    sharpe = round(float(mean_r / std_r * math.sqrt(252)), 4)

        # Win rate
        sell_trades = [t for t in trade_log if t["signal"] == "SELL"]
        wins = sum(1 for t in sell_trades if t["pnl"] > 0)
        win_rate = round(wins / len(sell_trades) * 100, 2) if sell_trades else 0.0

        return {
            "total_return": total_return,
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "total_trades": len(trade_log),
        }
