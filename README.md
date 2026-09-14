# Option Pricing: Finite Difference vs. PINN vs. Monte Carlo

Comparing three numerical approaches to solving the Black-Scholes PDE for
European options, then extending to American options (free-boundary problem).

## Why these three methods

| Method | Strength | Weakness |
|---|---|---|
| Finite difference (Crank-Nicolson) | Fast, exact on a grid, easy to extend to early exercise | Scales poorly beyond 1-2 underlying assets |
| PINN | Cheap to re-evaluate after training, mesh-free | Slow/unstable to train, needs care near boundaries |
| Monte Carlo | Scales to high dimensions, handles complex payoffs | Slow convergence, needs many paths for tight error bars |

## Structure

```
option-pricing-comparison/
├── black_scholes_fd.py      # Crank-Nicolson finite difference solver (Ground truth: matches analytical price to 0.037%)
├── black_scholes_pinn.py    # Physics-informed neural network solver
├── monte_carlo.py           # GBM path simulation + payoff averaging
├── american_option.py       # Free-boundary extension (early exercise), imports from black_scholes_fd.py
└── requirements.txt
```


## Milestones (weekend build)

1. **Sat AM** — `black_scholes_fd.py`: get Crank-Nicolson matching the closed-form
   Black-Scholes price to <0.1% error on a vanilla European call.
2. **Sat PM** — `monte_carlo.py`: simulate GBM paths, price the same option, plot
   price estimate + 95% CI as path count grows. Confirm it converges to the same
   price as FD.
3. **Sun AM** — `black_scholes_pinn.py`: train a small MLP to satisfy the PDE
   residual + boundary/terminal conditions. This is the hard part — expect to
   spend most of your debugging time on loss weighting between the PDE residual
   term and the boundary term.
4. **Sun PM** — `american_option.py`: add the early-exercise constraint to the FD
   solver via a projection step (price = max(continuation value, intrinsic value)
   at each timestep). Write up the comparison in this README.

## What to say about it in an interview

Don't just say "I priced options three ways." Say what you learned about the
tradeoffs — e.g., how much slower PINN training was relative to FD, where PINN
accuracy degraded (usually near the strike/boundary), and why Monte Carlo is
the only one of the three that generalizes cleanly to a basket option.

## Results

Test case throughout: S0=100, K=100, T=1 year, r=5%, σ=20%. Analytical
Black-Scholes price: **$10.4506**.

| Method | Price | Error | Cost |
|---|---|---|---|
| Analytical (closed-form) | $10.4506 | — | instant |
| Finite difference (Crank-Nicolson) | $10.4544 | 0.037% | milliseconds |
| Monte Carlo (1M paths) | $10.4532 ± $0.029 | 0.025% | seconds |
| PINN (Adam + L-BFGS, ~15k epochs) | $11.4511 | 9.6% | minutes of training |

**American put** (same parameters, S0=K=100): $6.0900, vs. $5.5774 for the
European put — a $0.51 early-exercise premium. Exercise boundary at
maturity: optimal to exercise when S ≤ $81.

### What actually differentiates the three methods

FD and Monte Carlo both nail the price to well under 0.1% error with no
real tuning. The PINN needed several rounds of fixes to get even within
10%: rescaling the loss terms (raw boundary/terminal MSE was thousands of
times larger than the PDE residual term, so the optimizer ignored the
PDE almost entirely), normalizing network inputs (tanh saturates past
|x|≈3, so feeding it raw S values up to 300 flattened the network's
ability to distinguish spot prices), and finally importance sampling
around the strike (uniform sampling across the full S range starved the
network of training signal exactly where the payoff's kink makes the
function hardest to fit — dropping error at S=100 from 55% to 9.6%).

That difficulty is itself the finding. PINNs are the most expensive to
train and, on a problem this well-behaved, the least precise of the
three. Their advantage isn't accuracy on a single vanilla option — it's
that a trained PINN can be evaluated instantly for new spot prices
without resolving anything, unlike FD (which needs the whole grid
re-swept) or Monte Carlo (which needs new paths simulated). That
expensive-upfront/cheap-repeated-inference tradeoff is the actual reason
PINNs get used in practice for pricing surfaces with many repeated
queries, not because they're more accurate on a single price.