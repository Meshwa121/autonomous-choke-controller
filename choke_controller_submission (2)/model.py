"""
model.py
Control-oriented predictive model, built from the identified gain curves
and time constants (data/model_params.json). Used by the MPC controller
to predict future process behavior for candidate choke moves -- the
controller NEVER calls the simulator internals directly, only this
identified model (as would be the case with a real, unknown well).
"""
import json
import numpy as np


class WellModel:
    def __init__(self, params_path="data/model_params.json"):
        with open(params_path) as f:
            p = json.load(f)
        self.tau_Q = p["tau_Q"]
        self.tau_P = p["tau_P"]
        self.u_grid = np.array(p["gain_curve_choke"])
        self.Q_grid = np.array(p["gain_curve_Q"])
        self.WHP_grid = np.array(p["gain_curve_WHP"])
        self.FLP_grid = np.array(p["gain_curve_FLP"])
        self.BHP_grid = np.array(p["gain_curve_BHP"])

    def steady_state(self, u):
        """Piecewise-linear interpolation of the identified gain curve."""
        u = np.clip(u, self.u_grid.min(), self.u_grid.max())
        Q = np.interp(u, self.u_grid, self.Q_grid)
        WHP = np.interp(u, self.u_grid, self.WHP_grid)
        FLP = np.interp(u, self.u_grid, self.FLP_grid)
        BHP = np.interp(u, self.u_grid, self.BHP_grid)
        return Q, WHP, FLP, BHP

    def predict_trajectory(self, x0, u0, choke_plan, horizon):
        """
        Predict Q, WHP, FLP, BHP over `horizon` future steps given a
        choke_plan (list of future choke positions, length >= horizon).

        x0: dict with current measured Q, WHP, FLP, BHP (used as the
            starting point for the first-order lag prediction)
        u0: current choke position (unused directly here but kept for API
            clarity / future extension, e.g. rate-based models)

        Returns dict of arrays: Q, WHP, FLP, BHP  (each length = horizon)
        """
        Q, WHP, FLP, BHP = x0["Q"], x0["WHP"], x0["FLP"], x0["BHP"]
        traj = {"Q": [], "WHP": [], "FLP": [], "BHP": []}

        for k in range(horizon):
            u_k = choke_plan[min(k, len(choke_plan) - 1)]
            Q_ss, WHP_ss, FLP_ss, BHP_ss = self.steady_state(u_k)

            Q += (1.0 / self.tau_Q) * (Q_ss - Q)
            WHP += (1.0 / self.tau_P) * (WHP_ss - WHP)
            FLP += (1.0 / self.tau_P) * (FLP_ss - FLP)
            BHP += (1.0 / self.tau_P) * (BHP_ss - BHP)

            traj["Q"].append(Q)
            traj["WHP"].append(WHP)
            traj["FLP"].append(FLP)
            traj["BHP"].append(BHP)

        return {k: np.array(v) for k, v in traj.items()}
