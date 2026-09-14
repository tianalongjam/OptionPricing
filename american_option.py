"""
American option pricing via finite difference with early exercise.
"""

import numpy as np
from black_scholes_fd import build_grid, crank_nicolson_step


def intrinsic_value(S_grid, K, option_type="put"):
    if option_type == "put":
        return np.maximum(K - S_grid, 0.0)
    elif option_type == "call":
        return np.maximum(S_grid - K, 0.0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")


def price_american_option_fd(S0, K, T, r, sigma, option_type="put",
                              S_max=None, n_space=200, n_time=200):
    if S_max is None:
        S_max = 3 * K
    S_grid, t_grid = build_grid(S_max, T, n_space, n_time)
    dt = t_grid[1] - t_grid[0]

    intrinsic = intrinsic_value(S_grid, K, option_type)
    V = intrinsic.copy()

    for step in range(n_time):
        if option_type == "put":
            V[0] = K
            V[-1] = 0.0
        else:
            V[0] = 0.0
            t_after_step = T - (step + 1) * dt
            V[-1] = S_max - K * np.exp(-r * t_after_step)

        V = crank_nicolson_step(V, S_grid, dt, r, sigma)
        V = np.maximum(V, intrinsic)

    return float(np.interp(S0, S_grid, V)), S_grid, V


def extract_exercise_boundary_at_maturity(S_grid, V, K, option_type="put", tol=1e-6):
    intrinsic = intrinsic_value(S_grid, K, option_type)
    exercising = (np.abs(V - intrinsic) < tol) & (intrinsic > tol)
    if not exercising.any():
        return None
    if option_type == "put":
        return S_grid[exercising].max()
    else:
        return S_grid[exercising].min()


if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    american_price, S_grid, V = price_american_option_fd(S0, K, T, r, sigma, option_type="put")

    from black_scholes_fd import price_european_call_fd
    euro_call = price_european_call_fd(S0, K, T, r, sigma)
    euro_put = euro_call - S0 + K * np.exp(-r * T)

    print(f"American put price: {american_price:.4f}")
    print(f"European put price: {euro_put:.4f}")
    print(f"Early exercise premium: {american_price - euro_put:.4f}")

    boundary = extract_exercise_boundary_at_maturity(S_grid, V, K, "put")
    print(f"Exercise boundary (today): S <= {boundary:.2f} -> exercise now")