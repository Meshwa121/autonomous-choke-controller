"""
scenarios.py
Runs the three required demonstration scenarios in closed loop
(simulator + autonomous MPC controller) and produces the required trend
plots for each: Target Oil Rate, Actual Oil Rate, WHP, FLP, BHP, Choke.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulator import WellSimulator
from model import WellModel
from controller import ChokeMPCController
from config import WHP_MIN, FLP_MIN, BHP_MIN


def run_closed_loop(target_schedule, n_steps, seed=100, start_choke=0.0):
    """
    target_schedule: function(step_index) -> target_Q for that step
    Returns a DataFrame log of the full closed-loop run.
    """
    sim = WellSimulator(seed=seed)
    model = WellModel()
    ctrl = ChokeMPCController(model=model, horizon=5)

    # Initialize simulator state at start_choke (warm/cold start)
    sim.reset()
    current_choke = start_choke
    # take one step at start_choke to get an initial measurement
    rec = sim.step(current_choke)

    logs = []
    for k in range(n_steps):
        target_Q = target_schedule(k)
        state = dict(Q=rec["Q"], WHP=rec["WHP"], FLP=rec["FLP"], BHP=rec["BHP"])

        next_choke, info = ctrl.select_choke(state, current_choke, target_Q)
        rec = sim.step(next_choke)
        current_choke = next_choke

        logs.append(dict(
            t=k, target_Q=target_Q, choke=rec["choke"],
            Q=rec["Q"], WHP=rec["WHP"], FLP=rec["FLP"], BHP=rec["BHP"],
            feasible=info["feasible"], predicted_Q=info["predicted_Q"], margin=info["margin"]
        ))

    return pd.DataFrame(logs)


def plot_scenario(df, title, filename):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1]})

    # --- Flow: target vs actual ---
    axes[0].plot(df["t"], df["target_Q"], "k--", label="Target Oil Rate", linewidth=1.6)
    axes[0].plot(df["t"], df["Q"], color="tab:blue", label="Actual Oil Rate")
    axes[0].set_ylabel("Oil Flow (bbl/hr)")
    axes[0].legend(loc="lower right")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    # --- Pressures with limits ---
    axes[1].plot(df["t"], df["WHP"], color="tab:red", label="WHP")
    axes[1].axhline(WHP_MIN, color="tab:red", linestyle=":", linewidth=1, label="WHP min")
    axes[1].plot(df["t"], df["FLP"], color="tab:orange", label="FLP")
    axes[1].axhline(FLP_MIN, color="tab:orange", linestyle=":", linewidth=1, label="FLP min")
    axes[1].plot(df["t"], df["BHP"], color="tab:green", label="BHP")
    axes[1].axhline(BHP_MIN, color="tab:green", linestyle=":", linewidth=1, label="BHP min")
    axes[1].set_ylabel("Pressure (psi)")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].grid(alpha=0.3)

    # --- Choke position ---
    axes[2].plot(df["t"], df["choke"], color="black", drawstyle="steps-post")
    axes[2].set_ylabel("Choke (%)")
    axes[2].set_xlabel("Time (control intervals, Ts = 1 hr)")
    axes[2].set_ylim(-5, 105)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)


def main():
    # --- Scenario A: Startup to Target ---
    target_A = 100.0
    dfA = run_closed_loop(lambda k: target_A, n_steps=40, seed=101, start_choke=0.0)
    dfA.to_csv("data/scenario_A_log.csv", index=False)
    plot_scenario(dfA, "Scenario A: Startup to Target (100 bbl/hr)", "plots/scenario_A.png")

    # --- Scenario B: Target Tracking (target changes mid-run) ---
    def target_B(k):
        return 100.0 if k < 30 else 150.0
    dfB = run_closed_loop(target_B, n_steps=60, seed=102, start_choke=0.0)
    dfB.to_csv("data/scenario_B_log.csv", index=False)
    plot_scenario(dfB, "Scenario B: Target Tracking (100 -> 150 bbl/hr)", "plots/scenario_B.png")

    # --- Scenario C: Infeasible Target ---
    target_C = 260.0  # exceeds max safe achievable (~189 bbl/hr, WHP-limited)
    dfC = run_closed_loop(lambda k: target_C, n_steps=40, seed=103, start_choke=0.0)
    dfC.to_csv("data/scenario_C_log.csv", index=False)
    plot_scenario(dfC, "Scenario C: Infeasible Target (260 bbl/hr requested)", "plots/scenario_C.png")

    # --- Summary printout ---
    for name, df, target in [("A", dfA, target_A), ("B", dfB, None), ("C", dfC, target_C)]:
        final = df.iloc[-1]
        print(f"\nScenario {name} final state:")
        print(f"  Choke={final['choke']:.2f}%  Q={final['Q']:.2f} bbl/hr  "
              f"WHP={final['WHP']:.1f}  FLP={final['FLP']:.1f}  BHP={final['BHP']:.1f}")
        print(f"  Any infeasible steps: {(~df['feasible']).sum()} / {len(df)}")
        print(f"  Min margin observed: {df['margin'].min():.2f} psi")


if __name__ == "__main__":
    main()
