"""
simulator.py
Well simulator CALIBRATED against the organizer-provided reference dataset
(Autonomous_Choke_Control_Simulated_Dataset.csv).

Calibration methodology (documented for the report):
The reference dataset contains 5 choke-hold segments (u = 30, 40, 55, 45, 65 %,
in that time order, including one down-step 55->45 that confirms no
hysteresis). For each segment we fit a first-order step response
    y(t) = y_ss + (y0 - y_ss) * exp(-t / tau)
by nonlinear least squares to extract, per segment: the true steady-state
value y_ss (extrapolating past the ~20-30 hr window actually observed) and
the time constant tau.

This gives 5 calibration points (u, Q_ss, WHP_ss, FLP_ss, BHP_ss). Steady-
state curves were then fit against these points:
  - Q_ss(u)   : power law Q = k * u^n forced through the origin
                (choke closed -> zero flow), fit by log-log regression.
                k=9.379, n=0.673
  - WHP_ss(u), FLP_ss(u), BHP_ss(u): quadratic in u, least-squares fit.
                Quadratics fit noticeably better than linear for BHP
                (residual std ~7-8 psi vs ~15.5 psi, close to the ~2.7 psi
                noise floor), reflecting mild curvature in the drawdown
                response. WHP/FLP quadratics are monotonic decreasing over
                the full 0-100% range. The BHP quadratic has its vertex just
                outside the calibration range (u ~ 94.7%, an extrapolation
                artifact since real data only spans 30-65%); it is clamped
                to be non-increasing beyond that point.

Time constants (hours, = control intervals since Ts = 1 hr) and measurement
noise standard deviations were taken directly as the per-segment averages
from the fits above.

NOTE ON FLP: the reference data shows FLP decreasing as choke opens (and Q
rises) -- i.e. FLP tracks WHP minus the (shrinking) choke pressure drop,
rather than rising with downstream flowline friction. This is captured
directly by fitting FLP_ss(u) empirically rather than assuming a friction
loss law; no separate flowline-friction sub-model was imposed.

Dynamics are NOT instantaneous: each variable relaxes toward its new
steady state via its own first-order lag, discretized at Ts = 1 hour,
matching the settling behavior observed in the reference step segments.
"""

import numpy as np


class WellSimulator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

        # --- Steady-state gain curves, identified from the reference dataset ---
        # Q_ss(u) = k * u^n   (u in %, forced through origin)
        self.Q_k = 9.3790
        self.Q_n = 0.6730

        # WHP_ss(u), FLP_ss(u): quadratic, monotonic decreasing over [0,100]
        self.WHP_coeffs = (-5.37990e-03, -1.07230, 308.2321)   # (a, b, c) for a*u^2+b*u+c
        self.FLP_coeffs = (-4.20067e-04, -0.957743, 217.8634)

        # BHP_ss(u): quadratic, clamped non-increasing past the vertex
        self.BHP_coeffs = (0.105365, -19.9459, 3713.264)
        self.BHP_vertex_u = -self.BHP_coeffs[1] / (2 * self.BHP_coeffs[0])  # ~94.66

        # --- First-order lag dynamics, per-variable time constants (hours) ---
        self.tau_Q = 5.52
        self.tau_WHP = 7.99
        self.tau_FLP = 8.78
        self.tau_BHP = 15.69

        # --- Measurement noise, per-variable std, identified from residuals ---
        self.noise_Q = 0.92
        self.noise_WHP = 0.72
        self.noise_FLP = 0.51
        self.noise_BHP = 2.68

        # --- State (start shut-in, i.e. u=0 steady state) ---
        self.u = 0.0
        Q0, WHP0, FLP0, BHP0 = self._steady_state(0.0)
        self.Q = Q0
        self.WHP = WHP0
        self.FLP = FLP0
        self.BHP = BHP0

        self.t = 0
        self.history = []

    # ------------------------------------------------------------------
    def _steady_state(self, u):
        """Evaluate the identified steady-state gain curves at choke position u (%)."""
        u = float(np.clip(u, 0.0, 100.0))

        Q = self.Q_k * (u ** self.Q_n) if u > 0 else 0.0

        a, b, c = self.WHP_coeffs
        WHP = a * u**2 + b * u + c

        a, b, c = self.FLP_coeffs
        FLP = a * u**2 + b * u + c

        a, b, c = self.BHP_coeffs
        u_eff = min(u, self.BHP_vertex_u)   # clamp: BHP non-increasing past vertex
        BHP = a * u_eff**2 + b * u_eff + c

        return Q, WHP, FLP, BHP

    # ------------------------------------------------------------------
    def step(self, choke_position: float):
        """
        Advance simulator by one control interval (Ts = 1 hour) given a choke
        position command. Applies first-order lag dynamics toward the new
        steady state, plus measurement noise.

        Returns dict: t, choke, Q, WHP, FLP, BHP (all post-step, measured)
        """
        u_cmd = float(np.clip(choke_position, 0.0, 100.0))
        self.u = u_cmd

        Q_ss, WHP_ss, FLP_ss, BHP_ss = self._steady_state(u_cmd)

        # First-order discrete update: x[k+1] = x[k] + (1/tau)*(x_ss - x[k])
        # (each variable uses its own identified time constant)
        self.Q += (1.0 / self.tau_Q) * (Q_ss - self.Q)
        self.WHP += (1.0 / self.tau_WHP) * (WHP_ss - self.WHP)
        self.FLP += (1.0 / self.tau_FLP) * (FLP_ss - self.FLP)
        self.BHP += (1.0 / self.tau_BHP) * (BHP_ss - self.BHP)

        # Measurement noise (per-variable std, identified from residuals)
        Q_meas = max(0.0, self.Q + self.rng.normal(0, self.noise_Q))
        WHP_meas = max(0.0, self.WHP + self.rng.normal(0, self.noise_WHP))
        FLP_meas = max(0.0, self.FLP + self.rng.normal(0, self.noise_FLP))
        BHP_meas = max(0.0, self.BHP + self.rng.normal(0, self.noise_BHP))

        self.t += 1
        record = dict(t=self.t, choke=self.u, Q=Q_meas, WHP=WHP_meas,
                      FLP=FLP_meas, BHP=BHP_meas)
        self.history.append(record)
        return record

    def reset(self):
        self.u = 0.0
        Q0, WHP0, FLP0, BHP0 = self._steady_state(0.0)
        self.Q = Q0
        self.WHP = WHP0
        self.FLP = FLP0
        self.BHP = BHP0
        self.t = 0
        self.history = []
