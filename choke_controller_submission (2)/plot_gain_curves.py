"""
plot_gain_curves.py
Plots the identified steady-state gain curves (Q, WHP, FLP, BHP vs choke
opening) from data/steady_state_gain_curve.csv, with the safe operating
envelope limits overlaid. Used in the report/notebook/deck to show the
static nonlinearity captured by the model.
"""
import pandas as pd
import matplotlib.pyplot as plt
from config import WHP_MIN, FLP_MIN, BHP_MIN

df = pd.read_csv("data/steady_state_gain_curve.csv")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].plot(df["u"], df["Q"], "o-", color="tab:blue")
axes[0, 0].set_title("Oil Flow Rate vs Choke Opening")
axes[0, 0].set_xlabel("Choke (%)")
axes[0, 0].set_ylabel("Q (bbl/hr)")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(df["u"], df["WHP"], "o-", color="tab:red")
axes[0, 1].axhline(WHP_MIN, color="tab:red", linestyle=":", label="WHP min")
axes[0, 1].set_title("Wellhead Pressure vs Choke Opening")
axes[0, 1].set_xlabel("Choke (%)")
axes[0, 1].set_ylabel("WHP (psi)")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(df["u"], df["FLP"], "o-", color="tab:orange")
axes[1, 0].axhline(FLP_MIN, color="tab:orange", linestyle=":", label="FLP min")
axes[1, 0].set_title("Flowline Pressure vs Choke Opening")
axes[1, 0].set_xlabel("Choke (%)")
axes[1, 0].set_ylabel("FLP (psi)")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(df["u"], df["BHP"], "o-", color="tab:green")
axes[1, 1].axhline(BHP_MIN, color="tab:green", linestyle=":", label="BHP min")
axes[1, 1].set_title("Bottom-Hole Pressure vs Choke Opening")
axes[1, 1].set_xlabel("Choke (%)")
axes[1, 1].set_ylabel("BHP (psi)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Identified Steady-State Gain Curves (from dedicated step-hold sweep)", fontsize=13)
plt.tight_layout()
plt.savefig("plots/gain_curves.png", dpi=130)
print("Saved plots/gain_curves.png")
