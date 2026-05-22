"""
Rolling walk-forward ML — no random split, scaler fit on train only.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from layers.feature_store import FeatureStore
from layers.purged_walk_forward import calibration_test_split


@dataclass
class TrainResult:
    experiment_id: str
    oos_sharpe: float
    oos_return: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    predictions: pd.Series | None = None
    diagnostics: dict = field(default_factory=dict)


class _RollingScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "_RollingScaler":
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fit")
        return (x - self.mean_) / self.std_


def _build_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    close = df["Close"]
    fwd = close.shift(-horizon) / close - 1.0
    vol = close.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)
    vol_lag = vol.shift(1)
    return (fwd / vol_lag.replace(0, np.nan)).shift(horizon)


def _make_model():
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            verbosity=-1,
        )
    except ImportError:
        pass
    try:
        import xgboost as xgb

        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            verbosity=0,
        )
    except ImportError:
        pass
    from sklearn.linear_model import Ridge

    return Ridge(alpha=1.0)


class MLTrainPipeline:
    def __init__(
        self,
        feature_names: list[str] | None = None,
        train_window: int = 120,
        test_window: int = 20,
        embargo: int = 1,
    ) -> None:
        self.store = FeatureStore()
        self.feature_names = feature_names or self.store.list_features()
        self.train_window = train_window
        self.test_window = test_window
        self.embargo = embargo

    def _prepare_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        feats = self.store.build_matrix(df, self.feature_names)
        target = _build_target(df)
        aligned = feats.join(target.rename("target")).dropna()
        return aligned[self.feature_names], aligned["target"]

    def rolling_oos_train(self, df: pd.DataFrame, experiment_id: str = "ml_run") -> TrainResult:
        x_all, y_all = self._prepare_xy(df)
        preds = []
        importance: dict[str, float] = {f: 0.0 for f in self.feature_names}
        windows = 0
        start = 0

        while start + self.train_window + self.embargo + self.test_window <= len(x_all):
            tr_end = start + self.train_window
            te_start = tr_end + self.embargo
            te_end = te_start + self.test_window
            x_tr = x_all.iloc[start:tr_end].to_numpy()
            y_tr = y_all.iloc[start:tr_end].to_numpy()
            x_te = x_all.iloc[te_start:te_end].to_numpy()
            y_te = y_all.iloc[te_start:te_end]

            scaler = _RollingScaler().fit(x_tr)
            model = _make_model()
            model.fit(scaler.transform(x_tr), y_tr)
            pred = model.predict(scaler.transform(x_te))
            preds.append(pd.Series(pred, index=y_te.index))

            if hasattr(model, "feature_importances_"):
                for i, f in enumerate(self.feature_names):
                    importance[f] += float(model.feature_importances_[i])
            windows += 1
            start += self.test_window

        if not preds:
            return TrainResult(experiment_id, 0.0, 0.0)

        pred_series = pd.concat(preds)
        err = pred_series - y_all.reindex(pred_series.index)
        oos_ret = pred_series.mean()
        sharpe = pred_series.mean() / pred_series.std() if pred_series.std() > 0 else 0.0
        imp = {k: v / max(windows, 1) for k, v in importance.items()}

        return TrainResult(
            experiment_id=experiment_id,
            oos_sharpe=float(sharpe),
            oos_return=float(oos_ret),
            feature_importance=imp,
            predictions=pred_series,
            diagnostics={"rmse": float((err**2).mean() ** 0.5), "windows": windows},
        )

    def fit_calibration(self, df: pd.DataFrame, experiment_id: str = "ml_cal") -> TrainResult:
        cal, _ = calibration_test_split(df, 0.7)
        return self.rolling_oos_train(cal, experiment_id)
