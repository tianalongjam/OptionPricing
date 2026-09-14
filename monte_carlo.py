"""
Monte Carlo pricer via simulation of Geometric Brownian Motion paths.

dS = r*S*dt + sigma*S*dW

Under risk-neutral measure, simulate terminal S_T, average the discounted
payoff across paths.
"""

import numpy as np


def simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=None):
    """
    Simulate S_T directly (no need to simulate the full path for a
    European option — only the terminal value matters).

    Hint: under GBM, S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    where Z ~ N(0,1). This is exact, no discretization error.

    TODO: implement, return array of shape (n_paths,)
    """
    raise NotImplementedError


def price_european_call_mc(S0, K, T, r, sigma, n_paths=100_000, seed=None):
    """
    Price = exp(-r*T) * mean(max(S_T - K, 0))

    TODO: implement using simulate_gbm_terminal, also return the standard
    error of the estimate so you can plot a confidence interval.
    """
    raise NotImplementedError


def convergence_study(S0, K, T, r, sigma, path_counts):
    """
    Price the option at increasing path counts, record price + 95% CI
    at each. This is the plot that shows you understand MC error scales
    as O(1/sqrt(n)).

    TODO: loop over path_counts, call price_european_call_mc, collect results.
    """
    raise NotImplementedError


if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    path_counts = [1_000, 10_000, 100_000, 1_000_000]
    results = convergence_study(S0, K, T, r, sigma, path_counts)
    # TODO: plot price vs. path_counts with error bars, log-x axis