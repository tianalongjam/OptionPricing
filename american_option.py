"""
American option pricing via finite difference with early exercise
(a free-boundary problem — the exercise boundary itself is unknown and
must be solved for as part of the solution, not specified in advance).

The trick: at each backward timestep, after the Crank-Nicolson update,
apply a projection: V(S,t) = max(V_continuation(S,t), intrinsic(S)).
This is the PSOR (projected successive over-relaxation) or simple
projection method — projection is simpler to implement, PSOR converges
faster if you want the extra credit.
"""

import numpy as np
from black_scholes_fd import build_grid, crank_nicolson_step


def intrinsic_value(S_grid, K, option_type="put"):
    """
    Immediate exercise value at each grid point.
    American puts are the standard example (American calls on
    non-dividend stocks are never optimal to exercise early, so a call
    example wouldn't show anything interesting here).

    TODO: return max(K - S, 0) for a put.
    """
    raise NotImplementedError


def price_american_option_fd(S0, K, T, r, sigma, option_type="put",
                              S_max=None, n_space=200, n_time=200):
    """
    Same backward time-stepping as the European FD solver, but after each
    Crank-Nicolson step, project onto the intrinsic value:

        V = max(V_continuation, intrinsic_value)

    TODO: implement. Reuse build_grid and crank_nicolson_step from
    black_scholes_fd.py, add the projection step after each timestep.
    """
    raise NotImplementedError


def extract_exercise_boundary(V_grid, S_grid, t_grid, K, option_type="put"):
    """
    For each time t, find the S value where early exercise first becomes
    optimal (where V equals intrinsic value). Plotting this boundary over
    time is a good visual for your writeup — it shows the region where
    holding vs. exercising is optimal.

    TODO: implement.
    """
    raise NotImplementedError


if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    price = price_american_option_fd(S0, K, T, r, sigma, option_type="put")
    print(f"American put price: {price:.4f}")
    # Compare to European put price (American should be >= European,
    # since early exercise is an added option)