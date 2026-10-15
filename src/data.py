"""Loading the SCADA export into a tidy DataFrame."""
from pathlib import Path

import pandas as pd

# Original headers in T1.csv, in order:
# Date/Time, LV ActivePower (kW), Wind Speed (m/s),
# Theoretical_Power_Curve (KWh), Wind Direction (°)
COLUMNS = [
    "timestamp",
    "power_kw",
    "wind_speed_ms",
    "theoretical_power_kw",  # the file labels this "KWh"; per 10-min sample it is a power value
    "wind_direction_deg",
]


def load_scada_data(path: Path) -> pd.DataFrame:
    """Read T1.csv, rename columns, parse timestamps and sort chronologically."""
    df = pd.read_csv(path, header=0, names=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %m %Y %H:%M")
    return df.sort_values("timestamp").reset_index(drop=True)
