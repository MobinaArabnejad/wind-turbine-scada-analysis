"""Run the full SCADA analysis: audit -> features -> power curve -> anomalies -> model.

Usage:
    python run_analysis.py
    python run_analysis.py --data path/to/T1.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import (detect_anomalies, empirical_power_curve,  # noqa: E402
                      evaluate_power_models, monthly_energy_loss)
from audit import audit_scada_data  # noqa: E402
from data import load_scada_data  # noqa: E402
from features import add_features, state_summary  # noqa: E402
from plots import (plot_anomalies, plot_monthly_loss,  # noqa: E402
                   plot_power_curve_states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "T1.csv")
    args = parser.parse_args()

    figures = ROOT / "figures"
    results = ROOT / "results"
    figures.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)

    df = load_scada_data(args.data)
    audit = audit_scada_data(df)

    df = add_features(df)
    states = state_summary(df)
    print("\n--- Operating states ---")
    print(states)

    curve = empirical_power_curve(df)
    monthly = monthly_energy_loss(df)
    anomalies = detect_anomalies(df)
    models = evaluate_power_models(df)

    plot_power_curve_states(df, curve, figures / "power_curve_states.png")
    plot_monthly_loss(monthly.copy(), figures / "monthly_energy_loss.png")
    plot_anomalies(df, anomalies, figures / "anomalies.png")

    total_theoretical = df["theoretical_power_kw"].sum() * (10 / 60) / 1000
    total_measured = df["power_kw"].sum() * (10 / 60) / 1000
    summary = {
        "audit": audit,
        "operating_states": states.reset_index().to_dict(orient="records"),
        "energy_mwh": {
            "theoretical": round(total_theoretical, 1),
            "measured": round(total_measured, 1),
            "lost_downtime": round(float(monthly.get("Downtime", 0).sum()), 1),
            "lost_underperformance": round(float(monthly.get("Underperforming", 0).sum()), 1),
        },
        "anomalies_flagged": int(anomalies.sum()),
        "models": models,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- Energy (MWh) ---")
    print(json.dumps(summary["energy_mwh"], indent=2))
    print("\n--- Power model, chronological test split ---")
    print(json.dumps(models, indent=2))
    print(f"\nFigures saved to {figures}\nSummary saved to {results / 'summary.json'}")


if __name__ == "__main__":
    main()
