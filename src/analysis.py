"""Power-curve analysis, anomaly detection and a simple power model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import CUT_IN_MS

MODEL_FEATURES = ["wind_speed_ms", "wind_dir_sin", "wind_dir_cos", "hour"]


def empirical_power_curve(df: pd.DataFrame, bin_width: float = 0.5) -> pd.DataFrame:
    """Median measured power per wind-speed bin, excluding downtime and below cut-in."""
    running = df[~df["state"].isin(["Downtime", "Below cut-in"])]
    bins = np.floor(running["wind_speed_ms"] / bin_width) * bin_width + bin_width / 2
    curve = running.groupby(bins).agg(
        measured_median_kw=("power_kw", "median"),
        theoretical_kw=("theoretical_power_kw", "median"),
        samples=("power_kw", "size"),
    )
    curve.index.name = "wind_speed_bin_ms"
    return curve[curve["samples"] >= 5].reset_index()


def monthly_energy_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Energy lost versus the theoretical curve, split by operating state."""
    lost = df[df["state"].isin(["Downtime", "Underperforming"])]
    table = lost.pivot_table(
        index=lost["timestamp"].dt.to_period("M"),
        columns="state",
        values="energy_loss_kwh",
        aggfunc="sum",
        fill_value=0.0,
    )
    return table / 1000  # MWh


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.Series:
    """Isolation Forest on (wind speed, power) for samples above cut-in."""
    active = df[df["wind_speed_ms"] >= CUT_IN_MS]
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(active[["wind_speed_ms", "power_kw"]])
    flag = pd.Series(False, index=df.index)
    flag.loc[active.index] = labels == -1
    return flag


def evaluate_power_models(df: pd.DataFrame, test_fraction: float = 0.2) -> dict:
    """Chronological split: compare the manufacturer curve with a fitted model.

    The model is trained on samples that are not downtime so it learns how the
    turbine really behaves; both are scored on every test sample.
    """
    split = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:split], df.iloc[split:]
    train = train[train["state"] != "Downtime"]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(train[MODEL_FEATURES], train["power_kw"])
    pred = model.predict(test[MODEL_FEATURES])

    def scores(y_pred):
        return {
            "mae_kw": float(mean_absolute_error(test["power_kw"], y_pred)),
            "rmse_kw": float(np.sqrt(mean_squared_error(test["power_kw"], y_pred))),
        }

    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": str(test["timestamp"].min()),
        "theoretical_curve": scores(test["theoretical_power_kw"]),
        "gradient_boosting": scores(pred),
    }
