"""
Physics-Based Mechanical Feature Engineering
=============================================
Transforms raw sensor telemetry into mechanically meaningful features that
give the ML model physically grounded predictors instead of raw signal
values alone. This is the core "why a mechanical engineer adds value over
a pure data scientist" step — every feature below is derived from a
machining-physics relationship, not generic statistics.

Features engineered:
  - Specific Cutting Energy (Uc)      : energy per unit volume of material
                                         removed -> J/mm^3
  - Mechanical Efficiency Ratio       : useful cutting power / total motor power
  - Material Removal Rate (MRR)       : mm^3/min
  - Vibration Signal-to-Noise Ratio   : dB, rolling window
  - Thermal Growth Rate               : dC/cycle (bearing thermal expansion proxy)
  - Wear Acceleration                 : d(VB)/d(cycle), 2nd derivative flags
                                         onset of tertiary (failure) wear stage
  - Rolling degradation trend features (5-cycle and 15-cycle windows) per unit
"""

import numpy as np
import pandas as pd

IN_PATH = "/home/claude/pdm_project/data/machining_sensor_stream.csv"
OUT_PATH = "/home/claude/pdm_project/data/features_engineered.csv"


def specific_cutting_energy(row):
    # Uc = Fc / (f * d)   [N/mm^2 == J/mm^3], Fc = cutting force, f = feed, d = depth of cut
    mrr = row["feed_rate_mm_rev"] * row["depth_of_cut_mm"] * row["rpm"]  # mm^3/min proxy
    uc = row["cutting_force_N"] / (row["feed_rate_mm_rev"] * row["depth_of_cut_mm"] + 1e-6)
    return pd.Series({"specific_cutting_energy_Jmm3": uc, "material_removal_rate_mm3min": mrr})


def rolling_snr(series, window=5):
    roll_mean = series.rolling(window, min_periods=2).mean()
    roll_std = series.rolling(window, min_periods=2).std().replace(0, np.nan)
    snr_db = 20 * np.log10((roll_mean.abs() / roll_std).bfill())
    return snr_db.replace([np.inf, -np.inf], np.nan).bfill().fillna(0)


def engineer_features(df):
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # --- Cutting mechanics ---
    energy_feats = df.apply(specific_cutting_energy, axis=1)
    df = pd.concat([df, energy_feats], axis=1)

    # --- Mechanical efficiency: theoretical vs actual power draw ---
    theoretical_power = (df["cutting_force_N"] * df["cutting_speed_m_min"]) / 60000
    df["mechanical_efficiency_ratio"] = (theoretical_power / (df["motor_power_kW"] + 1e-6)).clip(0, 1.2)

    out = []
    for uid, g in df.groupby("unit_id"):
        g = g.copy()
        # Vibration SNR (rolling, physically: signal energy vs noise floor)
        g["vibration_snr_dB"] = rolling_snr(g["vibration_rms_ms2"], window=5)

        # Thermal growth rate (bearing thermal expansion rate, deg C/cycle)
        g["thermal_growth_rate"] = g["spindle_temp_rise_C"].diff().fillna(0)

        # Wear rate & wear acceleration (1st & 2nd derivative of flank wear)
        g["wear_rate_VB_per_cycle"] = g["flank_wear_VB_mm"].diff().fillna(0)
        g["wear_acceleration"] = g["wear_rate_VB_per_cycle"].diff().fillna(0)

        # Rolling trend windows (short + medium horizon degradation signature)
        for w in (5, 15):
            g[f"vib_rms_roll_mean_{w}"] = g["vibration_rms_ms2"].rolling(w, min_periods=1).mean()
            g[f"vib_rms_roll_std_{w}"] = g["vibration_rms_ms2"].rolling(w, min_periods=1).std().fillna(0)
            g[f"kurtosis_roll_mean_{w}"] = g["spectral_kurtosis"].rolling(w, min_periods=1).mean()
            g[f"temp_roll_mean_{w}"] = g["spindle_temp_rise_C"].rolling(w, min_periods=1).mean()

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    return result


def main():
    df = pd.read_csv(IN_PATH)
    feat_df = engineer_features(df)
    feat_df.to_csv(OUT_PATH, index=False)
    print(f"Engineered feature set: {feat_df.shape[0]:,} rows x {feat_df.shape[1]} columns -> {OUT_PATH}")
    print("\nNew physics-derived columns:")
    new_cols = [c for c in feat_df.columns if c not in
                pd.read_csv(IN_PATH, nrows=1).columns]
    print(new_cols)


if __name__ == "__main__":
    main()
