"""Data-quality audit for the SCADA dataset."""
import pandas as pd


def audit_scada_data(df: pd.DataFrame) -> dict:
    """Print a quality report and return the key numbers as a dict."""
    step = df["timestamp"].diff().dropna()
    expected = step.mode().iloc[0]
    n_gaps = int((step > expected).sum())
    missing_steps = int(((step[step > expected] / expected) - 1).sum())

    report = {
        "rows": len(df),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "sampling_interval": str(expected),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "timestamps_monotonic": bool(df["timestamp"].is_monotonic_increasing),
        "time_gaps": n_gaps,
        "missing_time_steps": missing_steps,
    }

    print("=" * 60)
    print("SCADA DATA AUDIT")
    print("=" * 60)
    for key, value in report.items():
        print(f"{key:22s}: {value}")

    print("\n--- Numerical ranges ---")
    print(df.select_dtypes("number").agg(["min", "max", "mean", "std"]).T.round(2))
    return report
