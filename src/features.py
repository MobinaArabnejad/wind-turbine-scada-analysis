"""Feature engineering and operating-state classification."""
import numpy as np
import pandas as pd

CUT_IN_MS = 3.0            # below this wind speed the turbine is not expected to produce
UNDERPERF_FRACTION = 0.20  # more than 20% below the theoretical curve ...
UNDERPERF_MIN_KW = 100.0   # ... and at least 100 kW below it, counts as underperformance

STATES = ["Below cut-in", "Downtime", "Underperforming", "Normal"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add residuals, energy loss, wind-direction encoding and operating state."""
    df = df.copy()
    step_hours = df["timestamp"].diff().dropna().mode().iloc[0].total_seconds() / 3600

    df["residual_kw"] = df["power_kw"] - df["theoretical_power_kw"]
    df["power_loss_kw"] = (-df["residual_kw"]).clip(lower=0)
    df["energy_loss_kwh"] = df["power_loss_kw"] * step_hours

    rad = np.deg2rad(df["wind_direction_deg"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    df["hour"] = df["timestamp"].dt.hour

    below_cut_in = df["wind_speed_ms"] < CUT_IN_MS
    downtime = ~below_cut_in & (df["power_kw"] <= 0)
    shortfall = df["theoretical_power_kw"] - df["power_kw"]
    underperf = (
        ~below_cut_in
        & ~downtime
        & (shortfall > UNDERPERF_FRACTION * df["theoretical_power_kw"])
        & (shortfall > UNDERPERF_MIN_KW)
    )

    df["state"] = np.select(
        [below_cut_in, downtime, underperf],
        ["Below cut-in", "Downtime", "Underperforming"],
        default="Normal",
    )
    return df


def state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Share of samples and lost energy per operating state."""
    summary = df.groupby("state").agg(
        samples=("state", "size"),
        energy_loss_kwh=("energy_loss_kwh", "sum"),
    )
    summary["share_pct"] = 100 * summary["samples"] / len(df)
    return summary.reindex(STATES).round(1)
