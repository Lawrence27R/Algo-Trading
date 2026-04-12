"""
Real-time prediction module for ML-based trading signals.

Loads a previously trained model and returns a BUY / SELL / HOLD
signal with confidence and probability breakdown for the latest row.
"""

import os

import pandas as pd

from ml.feature_engineering import FeatureEngineer
from ml.model_trainer import ModelTrainer


class MLPredictor:
    """Generates live trading signals from a saved ML model."""

    def __init__(self, confidence_threshold: float = 0.55):
        """
        Parameters
        ----------
        confidence_threshold : float
            Minimum probability required to emit BUY or SELL.
            Below this threshold the signal is ``"HOLD"``.
        """
        self.confidence_threshold = confidence_threshold
        self.fe = FeatureEngineer()

    def predict(self, df: pd.DataFrame, symbol: str,
                model_type: str = "random_forest") -> dict:
        """
        Predict the trading signal for the most recent row of *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Recent OHLCV data (at least 60 rows recommended).
        symbol : str
            Ticker symbol matching the saved model filename.
        model_type : str
            ``"random_forest"`` or ``"xgboost"``.

        Returns
        -------
        dict
            Keys: ``signal``, ``confidence``, ``features_used``,
            ``probabilities``.
        """
        if len(df) < 60:
            raise ValueError(
                f"Need at least 60 rows of OHLCV data for prediction, got {len(df)}."
            )

        trainer = ModelTrainer(symbol=symbol, model_type=model_type)
        bundle = trainer.load_model()
        model = bundle["model"]
        scaler = bundle["scaler"]
        feature_cols = bundle["feature_cols"]

        featured = self.fe.build_features(df)
        if featured.empty:
            raise ValueError("Feature engineering produced no usable rows.")

        last_row = featured[feature_cols].iloc[[-1]]
        last_scaled = scaler.transform(last_row.values)

        proba = model.predict_proba(last_scaled)[0]
        buy_prob = float(proba[1])
        sell_prob = float(proba[0])
        confidence = max(buy_prob, sell_prob)

        if confidence < self.confidence_threshold:
            signal = "HOLD"
        elif buy_prob >= sell_prob:
            signal = "BUY"
        else:
            signal = "SELL"

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "features_used": feature_cols,
            "probabilities": {
                "BUY": round(buy_prob, 4),
                "SELL": round(sell_prob, 4),
            },
        }
