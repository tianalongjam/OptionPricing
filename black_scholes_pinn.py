"""
Physics-informed neural network solver for the Black-Scholes PDE.

Network learns V(S, t) directly. Loss = PDE residual loss + boundary loss
+ terminal condition loss, no labeled data needed (that's the "physics-
informed" part).

You've done this pattern before for your neural PDE solver research — same
idea, different PDE and different boundary conditions.
"""

import torch
import torch.nn as nn


class BlackScholesPINN(nn.Module):
    """
    Small MLP mapping (S, t) -> V.

    TODO: pick an architecture. Something like 4 hidden layers, 64 units,
    tanh activations is a reasonable starting point for a 2D PDE like this.
    """

    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        # TODO: build the network
        raise NotImplementedError

    def forward(self, S, t):
        # TODO: concatenate S, t and pass through network
        raise NotImplementedError


def pde_residual(model, S, t, r, sigma):
    """
    Compute the Black-Scholes PDE residual at points (S, t) using autograd.

    dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V

    Use torch.autograd.grad with create_graph=True to get first and second
    derivatives of V w.r.t. S and t.

    TODO: implement. This is the part most people get wrong on the first
    pass — double check your second derivative is taken correctly (grad of
    grad, not grad squared).
    """
    raise NotImplementedError


def boundary_loss(model, K, T, r, S_max):
    """
    Loss enforcing V(0,t)=0 and V(S_max,t) ~= S_max - K*exp(-r*(T-t)).

    TODO: sample t points, evaluate model at S=0 and S=S_max, compute MSE
    against the known boundary values.
    """
    raise NotImplementedError


def terminal_loss(model, K, T, S_max, n_samples=1000):
    """
    Loss enforcing V(S,T) = max(S-K, 0).

    TODO: sample S points, evaluate model at t=T, compute MSE against payoff.
    """
    raise NotImplementedError


def train_pinn(K, T, r, sigma, S_max, epochs=5000, lr=1e-3):
    """
    Training loop. Total loss = pde_loss + w1*boundary_loss + w2*terminal_loss.

    Heads up: getting the loss weights (w1, w2) right is usually the main
    source of pain. If the PDE loss dominates, boundary conditions get
    ignored and the solution drifts. Track each loss term separately so you
    can see which one is stuck.

    TODO: implement training loop, sample collocation points each epoch
    (or use a fixed set), backprop, step optimizer.
    """
    raise NotImplementedError


if __name__ == "__main__":
    K, T, r, sigma, S_max = 100, 1.0, 0.05, 0.2, 300
    model = train_pinn(K, T, r, sigma, S_max)
    # TODO: evaluate model at S0=100, t=0 and compare to analytical price