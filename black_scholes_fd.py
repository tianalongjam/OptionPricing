"""
Crank-Nicolson finite difference solver for the Black-Scholes PDE.

PDE (in terms of V(S, t), option value as a function of spot S and time t):
    dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0

Boundary conditions for a European call with strike K, maturity T:
    V(S, T) = max(S - K, 0)              # terminal payoff
    V(0, t) = 0                          # worthless if spot hits zero
    V(S, t) -> S - K*exp(-r*(T-t))  as S -> infinity

Crank-Nicolson averages the explicit and implicit finite-difference schemes,
giving unconditional stability and second-order accuracy in both S and t.
"""

import numpy as np


def analytical_black_scholes_call(S0, K, T, r, sigma):
    """
    Closed-form Black-Scholes price. Use this as ground truth to validate
    your finite difference and PINN solvers.

    TODO: implement using the standard d1/d2 formula and scipy.stats.norm.cdf
    """
    raise NotImplementedError


def build_grid(S_max, T, n_space, n_time):
    """
    Build the discretized (S, t) grid.

    TODO: return S array of shape (n_space+1,) from 0 to S_max,
    and t array of shape (n_time+1,) from 0 to T.
    """
    raise NotImplementedError


def crank_nicolson_step(V, S_grid, dt, r, sigma):
    """
    Advance the option value grid V by one timestep using Crank-Nicolson.

    Hint: this reduces to solving a tridiagonal linear system at each step.
    Look at scipy.linalg.solve_banded or build the tridiagonal matrices
    yourself with the standard central-difference coefficients for
    dV/dS and d2V/dS2.

    TODO: implement one CN timestep, return updated V.
    """
    raise NotImplementedError


def price_european_call_fd(S0, K, T, r, sigma, S_max=None, n_space=200, n_time=200):
    """
    Full pipeline: build grid, set terminal/boundary conditions, step
    backward in time from t=T to t=0, interpolate to get price at S0.

    TODO: wire the above functions together.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Sanity check: FD price should match analytical price to <0.1%
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    fd_price = price_european_call_fd(S0, K, T, r, sigma)
    exact_price = analytical_black_scholes_call(S0, K, T, r, sigma)
    print(f"FD price: {fd_price:.4f}, Analytical: {exact_price:.4f}")