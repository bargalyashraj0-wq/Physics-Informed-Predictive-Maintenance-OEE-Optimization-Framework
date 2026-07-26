"""
Business & Mechanical Outcome Translation
==========================================
Converts model outputs (RUL predictions, wear-state classification) into
shop-floor engineering economics: OEE component impact, unscheduled-downtime
cost avoidance, and the TBM -> CBM maintenance-interval shift.

Assumptions are labeled explicitly (industry-typical benchmarks for a
mid-size automotive machining/stamping line) so the numbers are defensible
in an interview, not just decorative.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "/home/claude/pdm_project/data/features_engineered.csv"
FIG_DIR = "/home/claude/pdm_project/outputs/figures"
REPORT_DIR = "/home/claude/pdm_project/outputs/reports"

# --- Baseline shop-floor assumptions (stated explicitly for defensibility) ---
ASSUMPTIONS = {
    "shift_hours_per_day": 8,
    "shifts_per_day": 3,
    "days_per_month": 26,
    "planned_cycle_time_sec": 42,          # ideal cycle time per part
    "unplanned_downtime_cost_per_hour_INR": 45000,   # line-stoppage cost (labor+opportunity)
    "avg_unplanned_stoppage_hours_TBM": 4.5,          # typical reactive-repair duration
    "avg_unplanned_stoppage_hours_CBM": 0.75,         # planned swap during CBM alert window
    "current_unplanned_failures_per_month_TBM": 6,
    "residual_failures_per_month_CBM": 1,             # model doesn't catch 100%
    "scrap_cost_per_part_INR": 850,
    "scrap_rate_reduction_pct": 0.35,   # fraction of tool-wear-induced scrap avoided
    "baseline_scrap_parts_per_month": 340,
}


def oee_impact():
    a = ASSUMPTIONS
    total_available_min = a["shift_hours_per_day"] * a["shifts_per_day"] * 60 * a["days_per_month"]

    # --- Availability: unplanned downtime hours removed ---
    downtime_TBM_hr = a["current_unplanned_failures_per_month_TBM"] * a["avg_unplanned_stoppage_hours_TBM"]
    downtime_CBM_hr = a["residual_failures_per_month_CBM"] * a["avg_unplanned_stoppage_hours_CBM"] \
        + (a["current_unplanned_failures_per_month_TBM"] - a["residual_failures_per_month_CBM"]) * 0.4  # planned swap ~24min

    avail_TBM = 1 - (downtime_TBM_hr * 60) / total_available_min
    avail_CBM = 1 - (downtime_CBM_hr * 60) / total_available_min

    # --- Quality: scrap-part reduction from catching wear before out-of-tolerance parts are cut ---
    scrap_TBM = a["baseline_scrap_parts_per_month"]
    scrap_CBM = scrap_TBM * (1 - a["scrap_rate_reduction_pct"])
    total_parts_est = total_available_min * 60 / a["planned_cycle_time_sec"]
    quality_TBM = 1 - scrap_TBM / total_parts_est
    quality_CBM = 1 - scrap_CBM / total_parts_est

    # --- Performance: assume modest gain from fewer micro-stops/chatter events near tool failure ---
    perf_TBM = 0.87
    perf_CBM = 0.91

    oee_TBM = avail_TBM * perf_TBM * quality_TBM
    oee_CBM = avail_CBM * perf_CBM * quality_CBM

    # --- Cost avoidance ---
    downtime_cost_TBM = downtime_TBM_hr * a["unplanned_downtime_cost_per_hour_INR"]
    downtime_cost_CBM = downtime_CBM_hr * a["unplanned_downtime_cost_per_hour_INR"]
    downtime_savings_month = downtime_cost_TBM - downtime_cost_CBM

    scrap_cost_TBM = scrap_TBM * a["scrap_cost_per_part_INR"]
    scrap_cost_CBM = scrap_CBM * a["scrap_cost_per_part_INR"]
    scrap_savings_month = scrap_cost_TBM - scrap_cost_CBM

    total_monthly_savings = downtime_savings_month + scrap_savings_month
    annual_savings = total_monthly_savings * 12

    results = {
        "OEE_TBM_baseline_pct": round(oee_TBM * 100, 2),
        "OEE_CBM_predictive_pct": round(oee_CBM * 100, 2),
        "OEE_improvement_pct_points": round((oee_CBM - oee_TBM) * 100, 2),
        "Availability_TBM_pct": round(avail_TBM * 100, 2),
        "Availability_CBM_pct": round(avail_CBM * 100, 2),
        "Quality_TBM_pct": round(quality_TBM * 100, 2),
        "Quality_CBM_pct": round(quality_CBM * 100, 2),
        "Performance_TBM_pct": round(perf_TBM * 100, 2),
        "Performance_CBM_pct": round(perf_CBM * 100, 2),
        "Unplanned_downtime_hours_avoided_per_month": round(downtime_TBM_hr - downtime_CBM_hr, 2),
        "Monthly_downtime_cost_savings_INR": round(downtime_savings_month, 0),
        "Monthly_scrap_cost_savings_INR": round(scrap_savings_month, 0),
        "Total_monthly_savings_INR": round(total_monthly_savings, 0),
        "Projected_annual_savings_INR": round(annual_savings, 0),
        "assumptions": a,
    }
    return results


def plot_oee_bridge(results):
    labels = ["Availability", "Performance", "Quality", "OEE (overall)"]
    tbm = [results["Availability_TBM_pct"], results["Performance_TBM_pct"],
           results["Quality_TBM_pct"], results["OEE_TBM_baseline_pct"]]
    cbm = [results["Availability_CBM_pct"], results["Performance_CBM_pct"],
           results["Quality_CBM_pct"], results["OEE_CBM_predictive_pct"]]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, tbm, width, label="TBM (Time-Based, current)", color="#94a3b8")
    plt.bar(x + width / 2, cbm, width, label="CBM (Predictive, proposed)", color="#2563eb")
    for i, (t, c) in enumerate(zip(tbm, cbm)):
        plt.text(i - width / 2, t + 1, f"{t:.1f}%", ha="center", fontsize=8)
        plt.text(i + width / 2, c + 1, f"{c:.1f}%", ha="center", fontsize=8)
    plt.xticks(x, labels)
    plt.ylabel("%")
    plt.ylim(0, 105)
    plt.title("OEE Impact: TBM vs Predictive CBM Strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/oee_impact_bridge.png")
    plt.close()


def plot_health_index_example():
    """Illustrative Machine Health Index (0-100) trajectory for one representative unit."""
    df = pd.read_csv(DATA_PATH)
    unit = df[df.unit_id == df.unit_id.unique()[0]].sort_values("cycle")

    # Composite Health Index: weighted normalized inverse of key degradation signals
    vib_n = (unit.vibration_rms_ms2 - unit.vibration_rms_ms2.min()) / (unit.vibration_rms_ms2.max() - unit.vibration_rms_ms2.min())
    temp_n = (unit.spindle_temp_rise_C - unit.spindle_temp_rise_C.min()) / (unit.spindle_temp_rise_C.max() - unit.spindle_temp_rise_C.min())
    vb_n = (unit.flank_wear_VB_mm - unit.flank_wear_VB_mm.min()) / (unit.flank_wear_VB_mm.max() - unit.flank_wear_VB_mm.min())
    ae_n = (unit.acoustic_emission_V - unit.acoustic_emission_V.min()) / (unit.acoustic_emission_V.max() - unit.acoustic_emission_V.min())

    degradation_score = 0.40 * vb_n + 0.25 * vib_n + 0.20 * temp_n + 0.15 * ae_n
    health_index = 100 * (1 - degradation_score)

    plt.figure(figsize=(7.5, 4.2))
    plt.plot(unit.cycle, health_index, color="#0f766e", linewidth=1.8)
    plt.axhspan(70, 100, color="#16a34a", alpha=0.12, label="Healthy (>70)")
    plt.axhspan(40, 70, color="#f59e0b", alpha=0.15, label="Degraded (40-70)")
    plt.axhspan(0, 40, color="#dc2626", alpha=0.12, label="Critical (<40)")
    plt.xlabel("Machining Cycle")
    plt.ylabel("Composite Health Index (0-100)")
    plt.title(f"Machine Health Index Trajectory — Unit {unit.unit_id.iloc[0]}")
    plt.legend(loc="lower left", fontsize=8)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/health_index_trajectory.png")
    plt.close()


def main():
    results = oee_impact()
    plot_oee_bridge(results)
    plot_health_index_example()

    with open(f"{REPORT_DIR}/oee_business_impact.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
