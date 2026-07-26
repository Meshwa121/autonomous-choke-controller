"""
controller.py
Autonomous choke controller: simplified MPC via brute-force candidate
evaluation (per FAQ: "A simplified MPC implementation based on brute
force candidate evaluation is acceptable").

Logic per control interval:
  1. Generate candidate next choke positions within the ramp-rate limit
     (u0 - 5% ... u0 + 5%, discretized on a fine grid), clipped to [0, 100].
  2. For each candidate, predict the trajectory over a short prediction
     horizon using the identified WellModel, ASSUMING the choke is held
     at that candidate for the rest of the horizon (a common, simple MPC
     simplification when only one future move needs to be decided per
     interval, since the process is re-optimized every interval anyway --
     i.e. a receding-horizon / control-horizon-of-1 MPC).
  3. Reject any candidate whose predicted trajectory violates the safety
     envelope (WHP, FLP, BHP limits) at ANY point in the horizon -- this
     is deliberately conservative (constraint must hold along the whole
     predicted path, not just at the final point).
  4. Among the FEASIBLE candidates, choose the one whose predicted
     steady-state flow is closest to the target rate.
  5. If NO candidate is feasible at the current choke position (e.g. the
     well is already outside limits, which should not happen in normal
     operation but is handled defensively), choose the candidate that
     most reduces the worst constraint violation (safety fallback).

This structure directly implements the required behavior:
  "Achieve the target oil production rate whenever feasible. If the
   target rate cannot be achieved safely, operate at the maximum
   achievable production rate without violating constraints."
because when a high target is infeasible, EVERY candidate's predicted
steady Q will be below target, and the controller will simply pick the
feasible candidate with the highest predicted Q (closest to target from
below) -- which drives the well to its maximum safe operating point and
holds it there.
"""
import numpy as np
from model import WellModel
from config import (WHP_MIN, WHP_MAX, FLP_MIN, FLP_MAX, BHP_MIN, BHP_MAX,
                     CHOKE_MIN, CHOKE_MAX, MAX_CHOKE_RAMP)


class ChokeMPCController:
    def __init__(self, model: WellModel = None, horizon=5, candidate_resolution=0.25):
        self.model = model if model is not None else WellModel()
        self.horizon = horizon
        self.candidate_resolution = candidate_resolution  # % grid resolution for candidates

    def _candidates(self, u0):
        lo = max(CHOKE_MIN, u0 - MAX_CHOKE_RAMP)
        hi = min(CHOKE_MAX, u0 + MAX_CHOKE_RAMP)
        n = int(round((hi - lo) / self.candidate_resolution)) + 1
        return np.linspace(lo, hi, n)

    @staticmethod
    def _violation_margin(traj):
        """Returns the worst (most negative = worst violation) margin to
        the constraint boundaries across the whole predicted trajectory.
        Positive margin means safely within limits."""
        margins = np.concatenate([
            traj["WHP"] - WHP_MIN, WHP_MAX - traj["WHP"],
            traj["FLP"] - FLP_MIN, FLP_MAX - traj["FLP"],
            traj["BHP"] - BHP_MIN, BHP_MAX - traj["BHP"],
        ])
        return margins.min()

    def select_choke(self, current_state, current_choke, target_Q):
        """
        current_state: dict with Q, WHP, FLP, BHP (latest measurements)
        current_choke: float, current choke position (%)
        target_Q: float, desired oil flow rate (bbl/hr)

        Returns: (next_choke, info_dict) where info_dict carries diagnostic
        info (feasible flag, predicted Q, margin, etc.) useful for logging
        and for the report's "rationale" requirement.
        """
        candidates = self._candidates(current_choke)

        best_feasible = None   # (abs_error, candidate, predicted_Q, margin)
        best_infeasible = None  # (=> fallback) (margin, candidate, predicted_Q)

        for u_cand in candidates:
            traj = self.model.predict_trajectory(
                current_state, current_choke, choke_plan=[u_cand], horizon=self.horizon
            )
            margin = self._violation_margin(traj)
            predicted_Q = traj["Q"][-1]

            if margin >= 0:  # feasible: constraints satisfied along full horizon
                err = abs(target_Q - predicted_Q)
                # Prefer minimal tracking error; on ties, prefer the one
                # closest to current choke (smoother control action)
                key = (err, abs(u_cand - current_choke))
                if best_feasible is None or key < best_feasible[0]:
                    best_feasible = (key, u_cand, predicted_Q, margin)
            else:
                if best_infeasible is None or margin > best_infeasible[0]:
                    best_infeasible = (margin, u_cand, predicted_Q)

        if best_feasible is not None:
            _, u_next, predicted_Q, margin = best_feasible
            info = dict(feasible=True, predicted_Q=predicted_Q, margin=margin,
                        num_feasible_candidates=None)
            return float(u_next), info
        else:
            # Defensive fallback: no candidate keeps us fully within limits
            # over the horizon -> pick the least-bad (max margin) option,
            # which will generally mean closing the choke to relieve
            # pressure constraints.
            margin, u_next, predicted_Q = best_infeasible
            info = dict(feasible=False, predicted_Q=predicted_Q, margin=margin)
            return float(u_next), info
