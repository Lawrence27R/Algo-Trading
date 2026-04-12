"""
Model training module for ML-based trading strategies.

Supports RandomForest and XGBoost classifiers with walk-forward
compatible train/test splitting, StandardScaler normalisation, and
persistence via joblib.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ml.feature_engineering import FeatureEngineer

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class ModelTrainer:
    """Trains, evaluates, and persists ML classification models."""

    def __init__(self, symbol: str, model_type: str = "random_forest",
                 hyperparams: dict | None = None):
        """
        Parameters
        ----------
        symbol : str
            Ticker symbol (used for file naming).
        model_type : str
            Either ``"random_forest"`` or ``"xgboost"``.
        hyperparams : dict, optional
            Hyperparameters forwarded to the underlying estimator.
        """
        self.symbol = symbol
        self.model_type = model_type
        self.hyperparams = hyperparams or {}
        self.fe = FeatureEngineer()
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_path(self, symbol=None, model_type=None):
        s = symbol or self.symbol
        m = model_type or self.model_type
        return os.path.join(SAVED_MODELS_DIR, f"{s}_{m}.pkl")

    def _metrics_path(self, symbol=None, model_type=None):
        s = symbol or self.symbol
        m = model_type or self.model_type
        return os.path.join(SAVED_MODELS_DIR, f"{s}_{m}_metrics.json")

    def _build_estimator(self):
        if self.model_type == "random_forest":
            params = {"n_estimators": 200, "max_depth": 6,
                      "random_state": 42, "n_jobs": -1}
            params.update(self.hyperparams)
            return RandomForestClassifier(**params)
        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "xgboost is not installed. Run: pip install xgboost"
                ) from exc
            params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                      "use_label_encoder": False, "eval_metric": "logloss",
                      "random_state": 42, "n_jobs": -1}
            params.update(self.hyperparams)
            return XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}. "
                             "Choose 'random_forest' or 'xgboost'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        """
        Prepare features, train the model, evaluate, and persist to disk.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame.

        Returns
        -------
        dict
            Metrics: accuracy, precision, recall, f1, roc_auc, trained_at,
            plus ``feature_importance`` mapping.
        """
        featured = self.fe.build_features(df)
        feat_cols = self.fe.feature_columns
        missing = [c for c in feat_cols if c not in featured.columns]
        if missing:
            raise ValueError(f"Missing feature columns after engineering: {missing}")

        X = featured[feat_cols].values
        y = featured["label"].values

        if len(X) < 50:
            raise ValueError(
                f"Not enough data to train: {len(X)} rows after feature engineering "
                "(need at least 50). Provide a longer history."
            )

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = self._build_estimator()
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "train_size": int(split),
            "test_size": int(len(X) - split),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "symbol": self.symbol,
            "model_type": self.model_type,
        }

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = {feat_cols[i]: round(float(v), 6)
                  for i, v in enumerate(model.feature_importances_)}
        else:
            fi = {}
        metrics["feature_importance"] = fi

        # Persist
        joblib.dump({"model": model, "scaler": scaler, "feature_cols": feat_cols},
                    self._model_path())
        with open(self._metrics_path(), "w") as fh:
            json.dump(metrics, fh, indent=2)

        return metrics

    def load_model(self, symbol: str | None = None,
                   model_type: str | None = None) -> dict:
        """
        Load a previously saved model bundle (model + scaler + feature cols).

        Returns
        -------
        dict
            Keys: ``model``, ``scaler``, ``feature_cols``.
        """
        path = self._model_path(symbol, model_type)
        if not os.path.exists(path):
            s = symbol or self.symbol
            m = model_type or self.model_type
            raise FileNotFoundError(
                f"No saved model found for {s}/{m} at {path}. "
                "Train a model first."
            )
        return joblib.load(path)

    def get_feature_importance(self, symbol: str | None = None,
                               model_type: str | None = None) -> dict:
        """
        Return feature importance scores for a saved model.

        Returns
        -------
        dict
            Mapping of feature name → importance score.
        """
        path = self._metrics_path(symbol, model_type)
        if not os.path.exists(path):
            s = symbol or self.symbol
            m = model_type or self.model_type
            raise FileNotFoundError(
                f"No metrics found for {s}/{m}. Train the model first."
            )
        with open(path) as fh:
            metrics = json.load(fh)
        return metrics.get("feature_importance", {})
