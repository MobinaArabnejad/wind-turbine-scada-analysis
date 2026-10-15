"""Figures saved to disk (no interactive windows)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STATE_COLORS = {
    "Normal": "#2a9d8f",
    "Underperforming": "#e9c46a",
    "Downtime": "#e63946",
    "Below cut-in": "#a8a8a8",
}


def plot_power_curve_states(df: pd.DataFrame, curve: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for state, color in STATE_COLORS.items():
        part = df[df["state"] == state]
        ax.scatter(part["wind_speed_ms"], part["power_kw"], s=5, alpha=0.35,
                   color=color, label=f"{state} ({len(part):,})")
    theo = df.sort_values("wind_speed_ms")
    ax.plot(theo["wind_speed_ms"], theo["theoretical_power_kw"], color="black",
            linewidth=1.8, label="Manufacturer curve")
    ax.plot(curve["wind_speed_bin_ms"], curve["measured_median_kw"], color="#1d3557",
            linewidth=2, linestyle="--", label="Empirical median curve")
    ax.set(xlabel="Wind speed (m/s)", ylabel="Active power (kW)",
           title="Power curve by operating state")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_monthly_loss(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    table.index = table.index.astype(str)
    table.plot.bar(stacked=True, ax=ax, color=[STATE_COLORS[c] for c in table.columns])
    ax.set(xlabel="Month", ylabel="Energy lost vs theoretical curve (MWh)",
           title="Monthly energy loss by cause")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_anomalies(df: pd.DataFrame, anomalies: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df.loc[~anomalies, "wind_speed_ms"], df.loc[~anomalies, "power_kw"],
               s=4, alpha=0.25, color="#457b9d", label="Regular")
    ax.scatter(df.loc[anomalies, "wind_speed_ms"], df.loc[anomalies, "power_kw"],
               s=8, alpha=0.8, color="#e63946", label=f"Anomaly ({int(anomalies.sum()):,})")
    ax.set(xlabel="Wind speed (m/s)", ylabel="Active power (kW)",
           title="Isolation Forest anomalies")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
