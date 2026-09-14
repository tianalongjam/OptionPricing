"""
Physics-informed neural network solver for the Black-Scholes PDE.
"""

import numpy as np
import torch
import torch.nn as nn


class BlackScholesPINN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, S, t):
        x = torch.cat([S, t], dim=1)
        return self.net(x)


def pde_residual(model, S, t, r, sigma):
    S.requires_grad_(True)
    t.requires_grad_(True)
    V = model(S, t)

    dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]
    dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    residual = dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
    return residual


def boundary_loss(model, K, T, r, S_max, n_samples=200):
    t_samples = torch.rand(n_samples, 1) * T

    S_zero = torch.zeros(n_samples, 1)
    V_at_zero = model(S_zero, t_samples)
    loss_zero = torch.mean(V_at_zero**2)

    S_max_t = torch.full((n_samples, 1), float(S_max))
    V_at_max = model(S_max_t, t_samples)
    target_at_max = S_max - K * torch.exp(-r * (T - t_samples))
    loss_max = torch.mean((V_at_max - target_at_max) ** 2)

    return loss_zero + loss_max


def terminal_loss(model, K, T, S_max, n_samples=1000):
    S_samples = torch.rand(n_samples, 1) * S_max
    t_T = torch.full((n_samples, 1), float(T))
    V_pred = model(S_samples, t_T)
    payoff = torch.clamp(S_samples - K, min=0.0)
    return torch.mean((V_pred - payoff) ** 2)


def train_pinn(K, T, r, sigma, S_max, epochs=5000, lr=1e-3,
                n_collocation=2000, w_pde=1.0, w_bc=1.0, w_term=1.0,
                verbose_every=500):
    model = BlackScholesPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        S_colloc = torch.rand(n_collocation, 1) * S_max
        t_colloc = torch.rand(n_collocation, 1) * T
        residual = pde_residual(model, S_colloc, t_colloc, r, sigma)
        loss_pde = torch.mean(residual**2)

        loss_bc = boundary_loss(model, K, T, r, S_max)
        loss_term = terminal_loss(model, K, T, S_max)

        loss = w_pde * loss_pde + w_bc * loss_bc + w_term * loss_term
        loss.backward()
        optimizer.step()

        if epoch % verbose_every == 0:
            print(f"epoch {epoch:5d}  total={loss.item():.5f}  pde={loss_pde.item():.5f}  "
                  f"bc={loss_bc.item():.5f}  term={loss_term.item():.5f}")

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
    model = train_pinn(K, T, r, sigma, S_max, epochs=5000)

    S0 = 100.0
    pinn_price = predict(model, S0, t0=0.0)
    exact_price = analytical_black_scholes_call(S0, K, T, r, sigma)
    print(f"\nPINN price: {pinn_price:.4f}, Analytical: {exact_price:.4f}")