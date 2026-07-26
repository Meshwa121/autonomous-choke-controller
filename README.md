# Autonomous Production Choke Controller

An autonomous, constraint-aware controller for a single naturally flowing oil well — automatically selects the optimal choke position every control interval to hit a target oil production rate, while keeping wellhead, flowline, and bottom-hole pressures within their safe operating envelope.

Built for the Honeywell Hackathon problem statement: *"Autonomous Production Choke Controller for a Single Naturally Flowing Oil Well."*

## Overview

Manual choke adjustment doesn't scale — operators managing many wells can't continuously re-optimize choke position by hand. This project implements a **brute-force Model Predictive Control (MPC)** approach that:

- Predicts the effect of candidate choke moves before applying them
- Rejects any move that would violate safety constraints (WHP / FLP / BHP limits, ±5%/interval ramp rate)
- Tracks a target oil production rate whenever it's safely achievable
- Automatically settles at the maximum *safe* production rate when the requested target is infeasible

## Key Result: Simulator Calibrated to Real Reference Data

No simulator executable was provided with the problem statement — only a reference dataset (`reference_dataset.csv`, 120 hourly rows across 5 choke-hold segments). Rather than guessing at well physics, the simulator was **calibrated directly against this dataset**:

1. Fit first-order step responses to each held segment to extrapolate true steady-state values (segments hadn't fully settled in the observed window)
2. Fit steady-state gain curves — power law for flow (forced through the origin), quadratic for pressures — through the resulting 5 calibration points
3. Time constants and measurement noise were taken as per-segment averages from the fits

The calibrated simulator reproduces the reference dataset's implied steady-state behavior within a few percent at every tested choke level (see the notebook's calibration validation section).

## Project Structure

```
├── simulator.py                  # Well simulator, calibrated to reference_dataset.csv
├── step_tests.py                 # Open-loop step test experiment
├── model_identification.py       # Fits gain curves + time constants from step-test data
├── model.py                      # Identified dynamic model used by the controller
├── controller.py                 # Brute-force MPC controller
├── scenarios.py                  # Runs closed-loop Scenarios A, B, C
├── config.py                     # Safe operating envelope + controller constants
├── plot_gain_curves.py           # Generates the gain-curve figure
├── reference_dataset.csv         # Organizer-provided reference data (calibration source)
├── data/                         # Generated CSVs (step-test data, scenario logs, identified params)
├── plots/                        # All generated figures
└── Autonomous_Choke_Controller.ipynb   # Full narrative notebook (executed, with outputs)
```

## Methodology

1. **Simulator calibration** — fit steady-state gain curves and time constants from the reference dataset
2. **Open-loop step testing** — apply a sequence of choke steps, record flow/pressure response
3. **Dynamic model identification** — fit a Hammerstein-type model (nonlinear steady-state gain + first-order lag) purely from step-test data
4. **Brute-force MPC controller** — every control interval: generate candidate choke moves within the ±5% ramp limit → predict the trajectory over a short horizon → reject any candidate that violates WHP/FLP/BHP limits → pick the candidate that gets closest to the target
5. **Closed-loop validation** — run the controller against the simulator across 3 required scenarios

## Results

| Scenario | Target | Final Q | Final Choke | Constraint Violations |
|---|---|---|---|---|
| A — Startup to Target | 100 bbl/hr | ~100.0 bbl/hr | ~33% | 0 / 40 |
| B — Target Tracking (100→150) | 150 bbl/hr | ~151.3 bbl/hr | ~62% | 0 / 60 |
| C — Infeasible Target | 260 bbl/hr (requested) | ~190.1 bbl/hr (max safe) | ~87% | 0 / 40 |

Zero constraint violations across all 140 total control intervals. In Scenario C, the controller correctly identifies that 260 bbl/hr cannot be reached safely and settles at the maximum achievable rate, holding wellhead pressure right at its safety limit rather than backing off unnecessarily or violating it.

## Running It

```bash
pip install numpy scipy pandas matplotlib

python3 step_tests.py              # open-loop step test
python3 model_identification.py    # identify dynamic model
python3 scenarios.py               # run closed-loop Scenarios A, B, C
```

Or open `Autonomous_Choke_Controller.ipynb` for the full narrative walkthrough with embedded results.

## Tech Stack

Python (NumPy, SciPy, Pandas, Matplotlib), Jupyter Notebook. No external optimization libraries required — brute-force candidate evaluation, as explicitly permitted by the problem statement.
