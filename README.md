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
option-pricing-comparison/
├── black_scholes_fd.py # Crank-Nicolson finite difference solver
├── black_scholes_pinn.py # Physics-informed neural network solver
├── monte_carlo.py # GBM path simulation + payoff averaging
├── american_option.py # Free-boundary extension (early exercise)
├── compare.py # Runs all three, plots error vs. analytical BS formula
└── requirements.txt


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

*(fill in after building — put your error table and convergence plots here)*