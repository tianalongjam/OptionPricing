"""
Physics-informed neural network solver for the Black-Scholes PDE.

The network learns V(S, t) directly. Instead of labeled training data,
the loss is built from three residuals:
  1. How badly the network's output violates the PDE itself, at randomly
     sampled interior points ("collocation points")
  2. How badly it violates the boundary conditions (S=0, S=S_max)
  3. How badly it violates the terminal condition (t=T, payoff)

NOTE: this file was written by hand and checked carefully against the
standard PINN pattern, but could not be executed in this environment
(torch's CUDA dependency is broken in this sandbox). Run it yourself
before trusting the numbers -- if training is unstable, the most likely
culprits are the loss weights (w_pde, w_bc, w_term below) or the learning
rate, both called out where they're used.
"""

import numpy as np
import torch
import torch.nn as nn


class BlackScholesPINN(nn.Module):
    def __init__(self, S_scale, T_scale, hidden_dim=64, n_layers=4):
        super().__init__()
        self.S_scale = S_scale
        self.T_scale = T_scale
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, S, t):
        # Normalize inputs to roughly [0, 1] before the tanh layers -- tanh
        # saturates past |x| ~ 3, and raw S values up to S_max=300 would
        # saturate almost everywhere, destroying the network's ability to
        # tell S=250 apart from S=300. Output stays in raw dollar units;
        # only the input scale matters here, since the final layer is
        # linear (no activation) and can freely output any magnitude.
        S_norm = S / self.S_scale
        t_norm = t / self.T_scale
        x = torch.cat([S_norm, t_norm], dim=1)
        return self.net(x)


def pde_residual(model, S, t, r, sigma):
    S.requires_grad_(True)
    t.requires_grad_(True)
    V = model(S, t)

    dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                 create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS),
                                   create_graph=True)[0]
    dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                 create_graph=True)[0]

    residual = (dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2
                + r * S * dV_dS - r * V)
    return residual


def sample_near_strike(n_samples, K, S_max, frac_near=0.5, spread=15.0):
    """
    Mix of uniform samples across [0, S_max] and samples concentrated
    near the strike K, where the payoff's curvature is sharpest and
    hardest for the network to fit. frac_near controls what fraction
    of samples get concentrated near K; spread controls how tight that
    concentration is (smaller spread = tighter cluster around K).
    """
    n_near = int(n_samples * frac_near)
    n_uniform = n_samples - n_near

    S_uniform = torch.rand(n_uniform, 1) * S_max
    S_near = K + torch.randn(n_near, 1) * spread
    S_near = torch.clamp(S_near, 0.0, S_max)

    return torch.cat([S_uniform, S_near], dim=0)


def boundary_loss(model, K, T, r, S_max, n_samples=200):
    t_samples = torch.rand(n_samples, 1) * T

    S_zero = torch.zeros(n_samples, 1)
    V_at_zero = model(S_zero, t_samples)
    loss_zero = torch.mean((V_at_zero / K) ** 2)  # V(0,t) = 0, scaled by K

    S_max_t = torch.full((n_samples, 1), float(S_max))
    V_at_max = model(S_max_t, t_samples)
    target_at_max = S_max - K * torch.exp(-r * (T - t_samples))
    loss_max = torch.mean(((V_at_max - target_at_max) / K) ** 2)

    return loss_zero + loss_max


def terminal_loss(model, K, T, S_max, n_samples=1000):
    S_samples = sample_near_strike(n_samples, K, S_max)
    t_T = torch.full((n_samples, 1), float(T))
    V_pred = model(S_samples, t_T)
    payoff = torch.clamp(S_samples - K, min=0.0)
    return torch.mean(((V_pred - payoff) / K) ** 2)


def train_pinn(K, T, r, sigma, S_max, epochs=15000, lr=1e-3,
                n_collocation=2000, w_pde=1.0, w_bc=1.0, w_term=1.0,
                verbose_every=1000, S0_track=100.0, lbfgs_steps=500):
    model = BlackScholesPINN(S_scale=S_max, T_scale=T)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def compute_loss():
        S_colloc = sample_near_strike(n_collocation, K, S_max)
        t_colloc = torch.rand(n_collocation, 1) * T
        residual = pde_residual(model, S_colloc, t_colloc, r, sigma)
        loss_pde = torch.mean(residual**2)
        loss_bc = boundary_loss(model, K, T, r, S_max)
        loss_term = terminal_loss(model, K, T, S_max)
        total = w_pde * loss_pde + w_bc * loss_bc + w_term * loss_term
        return total, loss_pde, loss_bc, loss_term

    # Phase 1: Adam. No LR decay here -- we want full step size until the
    # network is actually close, not cut short on a fixed schedule.
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_pde, loss_bc, loss_term = compute_loss()
        loss.backward()
        optimizer.step()

        if epoch % verbose_every == 0:
            current_price = predict(model, S0_track, t0=0.0)
            print(f"[Adam] epoch {epoch:6d}  total={loss.item():.5f}  "
                  f"pde={loss_pde.item():.5f}  bc={loss_bc.item():.5f}  "
                  f"term={loss_term.item():.5f}  price(S0={S0_track:.0f})={current_price:.4f}")

    # Phase 2: L-BFGS. A second-order optimizer that converges much more
    # precisely than Adam once you're already in the right neighborhood --
    # standard practice for squeezing PINNs the rest of the way. L-BFGS
    # needs a "closure" that recomputes loss and gradients on demand,
    # since it may evaluate the loss multiple times per step.
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_steps,
                               line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _, _, _ = compute_loss()
        loss.backward()
        return loss

    lbfgs.step(closure)

    final_loss, final_pde, final_bc, final_term = compute_loss()
    final_price = predict(model, S0_track, t0=0.0)
    print(f"[L-BFGS] total={final_loss.item():.5f}  pde={final_pde.item():.5f}  "
          f"bc={final_bc.item():.5f}  term={final_term.item():.5f}  "
          f"price(S0={S0_track:.0f})={final_price:.4f}")

    return model


def predict(model, S0, t0=0.0):
    model.eval()
    with torch.no_grad():
        S_t = torch.tensor([[float(S0)]])
        t_t = torch.tensor([[float(t0)]])
        return model(S_t, t_t).item()


if __name__ == "__main__":
    from black_scholes_fd import analytical_black_scholes_call

    K, T, r, sigma, S_max = 100, 1.0, 0.05, 0.2, 300
    model = train_pinn(K, T, r, sigma, S_max, epochs=15000)

    S0 = 100.0
    pinn_price = predict(model, S0, t0=0.0)
    exact_price = analytical_black_scholes_call(S0, K, T, r, sigma)
    print(f"\nPINN price: {pinn_price:.4f}, Analytical: {exact_price:.4f}")

    # Check whether error is concentrated at the strike (the payoff kink)
    # or spread evenly across the domain -- this tells you whether the
    # remaining gap is a genuine PINN limitation near S=K, or something
    # still wrong with training generally.
    print("\nError across a range of spot prices (t=0):")
    print(f"{'S0':>8}  {'PINN':>10}  {'Analytical':>10}  {'Error':>10}  {'Error %':>8}")
    for S0_test in [20, 50, 75, 90, 100, 110, 125, 150, 200]:
        pinn_p = predict(model, S0_test, t0=0.0)
        exact_p = analytical_black_scholes_call(S0_test, K, T, r, sigma)
        err = pinn_p - exact_p
        err_pct = abs(err) / max(exact_p, 1e-6) * 100
        print(f"{S0_test:>8}  {pinn_p:>10.4f}  {exact_p:>10.4f}  {err:>10.4f}  {err_pct:>7.1f}%")

        