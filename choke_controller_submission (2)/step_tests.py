"""
step_tests.py
Open-loop step-test experiments on the well simulator, per deliverable:
"Study the process behavior by applying choke step changes / Plot the
response of flow and pressures."

Produces:
  - data/step_test_data.csv   : full time series of the step test
  - plots/step_test_response.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulator import WellSimulator

def run_step_test_sequence(sim: WellSimulator, choke_sequence, hold_steps=15):
    """Apply a sequence of choke setpoints, holding each for `hold_steps`
    control intervals, and record the full response."""
    sim.reset()
    records = []
    for u_target in choke_sequence:
        for _ in range(hold_steps):
            rec = sim.step(u_target)
            records.append(rec)
    return pd.DataFrame(records)


def main():
    sim = WellSimulator(seed=1)

    # A staircase of choke steps spanning the operating range, both up and
    # down, to characterize gain and dynamics (including some down-steps to
    # check for asymmetric behavior, though this simulator is symmetric).
    choke_sequence = [10, 10, 25, 25, 40, 40, 60, 60, 45, 45, 75, 75, 90, 90, 55, 55]

    df = run_step_test_sequence(sim, choke_sequence, hold_steps=15)
    df.to_csv("data/step_test_data.csv", index=False)

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    t = df["t"].values

    axes[0].plot(t, df["choke"], color="black", drawstyle="steps-post")
    axes[0].set_ylabel("Choke (%)")
    axes[0].set_title("Open-Loop Step Test: Choke Sequence and Process Response")

    axes[1].plot(t, df["Q"], color="tab:blue")
    axes[1].set_ylabel("Oil Flow\n(bbl/hr)")

    axes[2].plot(t, df["WHP"], color="tab:red")
    axes[2].set_ylabel("WHP (psi)")

    axes[3].plot(t, df["FLP"], color="tab:orange")
    axes[3].set_ylabel("FLP (psi)")

    axes[4].plot(t, df["BHP"], color="tab:green")
    axes[4].set_ylabel("BHP (psi)")
    axes[4].set_xlabel("Time (control intervals, Ts = 1 hr)")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/step_test_response.png", dpi=130)
    print("Saved data/step_test_data.csv and plots/step_test_response.png")
    print(df.describe())


if __name__ == "__main__":
    main()
