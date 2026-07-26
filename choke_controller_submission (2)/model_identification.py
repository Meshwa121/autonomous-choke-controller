"""
model_identification.py

Develops a simple, control-oriented dynamic model of the well from
experimental (simulator) data, per deliverable:
  "Develop a simple dynamic model representing the process behavior"

Modeling approach:
  1. STATIC GAIN CURVES: run a dedicated sweep of long-hold choke steps
     across 0-100% to map steady-state (Q, WHP, FLP, BHP) vs choke opening.
     These are stored as lookup tables and interpolated (piecewise-linear)
     at runtime -- this captures the process nonlinearity without needing
     a parametric physical model.
  2. FIRST-ORDER LAG DYNAMICS: fit a single effective time constant for
     flow (tau_Q) and for pressures (tau_P) by exponential curve-fitting
     the step-test transient response. The process is modeled as a
     first-order response moving toward the (nonlinear) steady-state
     target after each choke move -- i.e. a nonlinear-static +
     linear-dynamic (Hammerstein-type) model.

This model is deliberately simple (no need for a full nonlinear ODE /
first-principles model) but captures the essential closed-loop behavior
needed for predictive control: steady-state nonlinearity + first-order
lag + (implicitly) noise.
"""
import numpy as np
import pandas as pd
import json
from scipy.optimize import curve_fit
from simulator import WellSimulator


def sweep_steady_state(hold_steps=40, choke_points=None):
    """Dedicated experiment: hold choke at each test point long enough to
    reach steady state, record final (Q, WHP, FLP, BHP)."""
    if choke_points is None:
        choke_points = list(range(0, 101, 5))

    sim = WellSimulator(seed=7)
    rows = []
    for u in choke_points:
        sim.reset()
        for _ in range(hold_steps):
            rec = sim.step(u)
        rows.append(dict(u=u, Q=rec["Q"], WHP=rec["WHP"], FLP=rec["FLP"], BHP=rec["BHP"]))
    return pd.DataFrame(rows)


def _exp_approach(t, tau, x0, xss):
    return xss + (x0 - xss) * np.exp(-t / tau)


def fit_time_constants(step_df: pd.DataFrame, choke_sequence, hold_steps):
    """Fit a single effective tau for Q and for pressures (WHP/FLP/BHP
    averaged) using the transient segments of the step-test data."""
    taus_Q, taus_P = [], []

    idx = 0
    prev_u = None
    for seg_i, u_target in enumerate(choke_sequence):
        seg = step_df.iloc[idx: idx + hold_steps].reset_index(drop=True)
        idx += hold_steps
        if prev_u is None or u_target == prev_u:
            prev_u = u_target
            continue  # skip non-step (repeated) segments and first segment

        t_local = np.arange(len(seg))
        x0_Q = seg["Q"].iloc[0]
        xss_Q = seg["Q"].iloc[-1]
        if abs(xss_Q - x0_Q) > 2:  # only fit meaningful steps
            try:
                popt, _ = curve_fit(
                    lambda t, tau: _exp_approach(t, tau, x0_Q, xss_Q),
                    t_local, seg["Q"].values, p0=[2.0], bounds=(0.2, 15)
                )
                taus_Q.append(popt[0])
            except Exception:
                pass

        for col in ["WHP", "FLP", "BHP"]:
            x0_P = seg[col].iloc[0]
            xss_P = seg[col].iloc[-1]
            if abs(xss_P - x0_P) > 5:
                try:
                    popt, _ = curve_fit(
                        lambda t, tau: _exp_approach(t, tau, x0_P, xss_P),
                        t_local, seg[col].values, p0=[3.0], bounds=(0.2, 15)
                    )
                    taus_P.append(popt[0])
                except Exception:
                    pass

        prev_u = u_target

    tau_Q = float(np.median(taus_Q)) if taus_Q else 1.6
    tau_P = float(np.median(taus_P)) if taus_P else 2.4
    return tau_Q, tau_P, taus_Q, taus_P


def main():
    # 1. Static gain curves
    gain_df = sweep_steady_state()
    gain_df.to_csv("data/steady_state_gain_curve.csv", index=False)
    print("Steady-state gain curve (identified):")
    print(gain_df)

    # 2. Time constants from step-test data (reuse the step test sequence)
    step_df = pd.read_csv("data/step_test_data.csv")
    choke_sequence = [10, 10, 25, 25, 40, 40, 60, 60, 45, 45, 75, 75, 90, 90, 55, 55]
    tau_Q, tau_P, taus_Q_list, taus_P_list = fit_time_constants(step_df, choke_sequence, hold_steps=15)

    print(f"\nIdentified tau_Q = {tau_Q:.2f} control intervals")
    print(f"Identified tau_P = {tau_P:.2f} control intervals")
    print(f"(from {len(taus_Q_list)} flow steps, {len(taus_P_list)} pressure steps)")

    model_params = {
        "tau_Q": tau_Q,
        "tau_P": tau_P,
        "gain_curve_choke": gain_df["u"].tolist(),
        "gain_curve_Q": gain_df["Q"].tolist(),
        "gain_curve_WHP": gain_df["WHP"].tolist(),
        "gain_curve_FLP": gain_df["FLP"].tolist(),
        "gain_curve_BHP": gain_df["BHP"].tolist(),
    }
    with open("data/model_params.json", "w") as f:
        json.dump(model_params, f, indent=2)
    print("\nSaved data/model_params.json and data/steady_state_gain_curve.csv")


if __name__ == "__main__":
    main()
