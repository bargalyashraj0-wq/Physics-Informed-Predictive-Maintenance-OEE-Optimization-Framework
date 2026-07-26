"""
Physics-Informed Synthetic Data Generator
==========================================
Simulates run-to-failure degradation trajectories for a fleet of high-speed
precision machining centers (CNC milling stations machining automotive
powertrain components: engine block bore, crankshaft journals, gear teeth).

Design philosophy — modeled jointly on:
  1. NASA C-MAPSS run-to-failure structure (unit_id, cycle, RUL labeling)
  2. Taylor's Tool Life equation for physically consistent flank wear (V_B)
     growth as a function of cutting speed, feed, and depth of cut
  3. ISO 3685 tool wear stage definitions (Healthy / Degraded / Critical)

This is SYNTHETIC data generated from governing physics equations + realistic
sensor noise — used because live shop-floor telemetry / Kaggle CNC-mill /
Bosch datasets aren't accessible in this offline environment. The generation
logic mirrors the statistical structure (degradation curves, noise bands,
sensor cross-correlations) documented in those public datasets so the
downstream pipeline (feature engineering -> XGBoost -> SHAP) is a faithful
stand-in for training on the real thing.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_MACHINES = 45          # fleet size (machining centers on the shop floor)
MAX_CYCLE_NOISE = 0.12   # run-to-run variability in life (manufacturing variance)


def taylor_tool_life(v_cutting, feed, depth_of_cut, C=380, n=0.28, a=0.15, b=0.10):
    """
    Extended Taylor Tool Life Equation:
        V * T^n * f^a * d^b = C
    Solve for T (tool life in minutes) given cutting speed V (m/min),
    feed f (mm/rev), depth of cut d (mm).
    """
    T = (C / (v_cutting * (feed ** a) * (depth_of_cut ** b))) ** (1 / n)
    return T


def generate_unit(unit_id):
    # --- Operating condition envelope (randomized per machine/job) ---
    rpm = np.random.uniform(2800, 4200)                       # spindle speed
    feed_rate = np.random.uniform(0.08, 0.22)                 # mm/rev
    depth_of_cut = np.random.uniform(0.5, 1.8)                # mm
    tool_diameter = np.random.uniform(8, 16)                  # mm
    v_cutting = np.pi * tool_diameter * rpm / 1000             # m/min

    life_minutes = taylor_tool_life(v_cutting, feed_rate, depth_of_cut)
    total_cycles = int(np.clip(life_minutes / 3.2, 90, 260))   # ~cycles to failure
    total_cycles = int(total_cycles * np.random.normal(1.0, MAX_CYCLE_NOISE))
    total_cycles = max(total_cycles, 60)

    coolant_flow_nominal = np.random.uniform(18, 26)          # L/min

    rows = []
    # Flank wear VB (mm) grows non-linearly: initial break-in -> steady -> rapid failure zone
    for cycle in range(1, total_cycles + 1):
        frac = cycle / total_cycles  # 0 -> 1 life fraction

        # --- Non-linear 3-stage flank wear model (ISO 3685) ---
        VB = (0.05 * (1 - np.exp(-frac * 6))            # break-in wear
              + 0.10 * frac                              # steady-state linear wear
              + 0.18 * (frac ** 8))                       # accelerated failure wear
        VB *= np.random.normal(1.0, 0.03)

        # --- Vibration: RMS acceleration grows with wear + chatter onset near failure ---
        vib_rms = 0.8 + 4.2 * VB + 0.9 * (frac ** 10) * np.random.normal(1, 0.15)
        vib_rms += np.random.normal(0, 0.05)

        # Spectral kurtosis: near-Gaussian (~3) when healthy, spikes with impacting/chipping
        kurtosis = 3.0 + 9.0 * (VB ** 1.6) + np.random.normal(0, 0.2)

        # --- Thermal dynamics: spindle bearing temp rise (deg C above ambient) ---
        spindle_temp_rise = 8 + 55 * VB + 6 * (frac ** 6) + np.random.normal(0, 1.2)

        # --- Acoustic emission (AE RMS, volts) rises with micro-chipping ---
        acoustic_emission = 0.15 + 1.8 * VB ** 1.3 + np.random.normal(0, 0.03)

        # --- Coolant flow degrades slightly due to nozzle wear / chip buildup ---
        coolant_flow = coolant_flow_nominal * (1 - 0.05 * frac) + np.random.normal(0, 0.3)

        # --- Cutting force proxy (N) via specific cutting energy relation ---
        cutting_force = (depth_of_cut * feed_rate * 2200) * (1 + 1.4 * VB) \
            + np.random.normal(0, 8)

        # --- Spindle motor power draw (kW) ---
        motor_power = (cutting_force * v_cutting) / 60000 * (1 + 0.35 * VB) \
            + np.random.normal(0, 0.05)

        rul = total_cycles - cycle

        # Tool wear state per ISO 3685 flank wear thresholds
        if VB < 0.15:
            state = "Healthy"
        elif VB < 0.30:
            state = "Degraded"
        else:
            state = "Critical"

        rows.append(dict(
            unit_id=unit_id, cycle=cycle, rpm=rpm, feed_rate_mm_rev=feed_rate,
            depth_of_cut_mm=depth_of_cut, cutting_speed_m_min=v_cutting,
            vibration_rms_ms2=max(vib_rms, 0), spectral_kurtosis=max(kurtosis, 2.5),
            spindle_temp_rise_C=max(spindle_temp_rise, 0),
            acoustic_emission_V=max(acoustic_emission, 0),
            coolant_flow_Lmin=max(coolant_flow, 0),
            cutting_force_N=max(cutting_force, 0),
            motor_power_kW=max(motor_power, 0),
            flank_wear_VB_mm=max(VB, 0),
            RUL=rul, tool_state=state
        ))
    return pd.DataFrame(rows)


def main():
    all_units = [generate_unit(uid) for uid in range(1, N_MACHINES + 1)]
    df = pd.concat(all_units, ignore_index=True)
    out_path = "/home/claude/pdm_project/data/machining_sensor_stream.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} cycle-records across {N_MACHINES} machines -> {out_path}")
    print(df.groupby("tool_state").size())
    print(df[["vibration_rms_ms2", "spindle_temp_rise_C", "flank_wear_VB_mm", "RUL"]].describe())


if __name__ == "__main__":
    main()
