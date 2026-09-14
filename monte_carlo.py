"""
Crank-Nicolson finite difference solver for the Black-Scholes PDE.
"""

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded


def analytical_black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def build_grid(S_max, T, n_space, n_time):
    S_grid = np.linspace(0, S_max, n_space + 1)
    t_grid = np.linspace(0, T, n_time + 1)
    return S_grid, t_grid


def crank_nicolson_step(V, S_grid, dt, r, sigma):
    n = len(S_grid) - 1
    dS = S_grid[1] - S_grid[0]
    i = np.arange(1, n)
    S_i = S_grid[i]

    alpha = 0.25 * dt * (sigma**2 * (S_i / dS) ** 2 - r * S_i / dS)
    beta = -0.5 * dt * (sigma**2 * (S_i / dS) ** 2 + r)
    gamma = 0.25 * dt * (sigma**2 * (S_i / dS) ** 2 + r * S_i / dS)

    n_int = n - 1
    ab_A = np.zeros((3, n_int))
    ab_A[0, 1:] = -gamma[:-1]
    ab_A[1, :] = 1 - beta
    ab_A[2, :-1] = -alpha[1:]

    V_old = V[1:n]
    rhs = (1 + beta) * V_old
    rhs[:-1] += gamma[:-1] * V_old[1:]
    rhs[1:] += alpha[1:] * V_old[:-1]
    rhs[0] += alpha[0] * V[0]
    rhs[-1] += gamma[-1] * V[-1]

    V_new_interior = solve_banded((1, 1), ab_A, rhs)
    V_new = V.copy()
    V_new[1:n] = V_new_interior
    return V_new


def price_european_call_fd(S0, K, T, r, sigma, S_max=None, n_space=200, n_time=200):
    if S_max is None:
        S_max = 3 * K
    S_grid, t_grid = build_grid(S_max, T, n_space, n_time)
    dt = t_grid[1] - t_grid[0]

    V = np.maximum(S_grid - K, 0.0)

    for step in range(n_time):
        t_after_step = T - (step + 1) * dt
        V[0] = 0.0
        V[-1] = S_max - K * np.exp(-r * t_after_step)
        V = crank_nicolson_step(V, S_grid, dt, r, sigma)
        V[0] = 0.0
        V[-1] = S_max - K * np.exp(-r * t_after_step)

    return float(np.interp(S0, S_grid, V))


if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    fd_price = price_european_call_fd(S0, K, T, r, sigma)
    exact_price = analytical_black_scholes_call(S0, K, T, r, sigma)
    err_pct = abs(fd_price - exact_price) / exact_price * 100
    print(f"FD price: {fd_price:.4f}, Analytical: {exact_price:.4f}, error: {err_pct:.3f}%")