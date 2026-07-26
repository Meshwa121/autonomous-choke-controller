"""
config.py
Central definition of the safe operating envelope and controller constants.

Limits below are set relative to the CALIBRATED simulator (fit to the
organizer-provided reference dataset). WHP_MIN = 175 psi is chosen so it
becomes the binding constraint at u ~ 86.7% (Q ~ 189 bbl/hr) -- i.e. before
the choke reaches full open -- mirroring a realistic operating envelope
where wellhead pressure, not choke travel, is what ultimately limits
production. FLP_MIN and BHP_MIN are set with margin so WHP remains the
first constraint to bind; they are still actively checked every step.
"""

# --- Safe operating envelope (active constraints) ---
WHP_MIN = 175.0     # psi, minimum wellhead pressure (binding constraint, ~86.7% choke)
FLP_MIN = 115.0     # psi, minimum flowline pressure
BHP_MIN = 2650.0    # psi, minimum bottom-hole pressure (drawdown limit)

# Upper bounds included for completeness / robustness (not expected to bind
# in this well's normal operating range, but checked anyway)
WHP_MAX = 330.0
FLP_MAX = 240.0
BHP_MAX = 3800.0    # shut-in (u=0) BHP is ~3711 psi -- must stay above that

# --- Choke constraints ---
CHOKE_MIN = 0.0
CHOKE_MAX = 100.0
MAX_CHOKE_RAMP = 5.0   # max |choke move| per control interval (%)

# --- Control interval ---
TS_HOURS = 1.0
