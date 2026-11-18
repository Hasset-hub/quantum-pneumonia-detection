import torch
import torch.nn as nn


class LogisticHead(nn.Module):
    """Baseline 1: a single linear layer (logistic regression on embeddings)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def matched_hidden(n_qubits: int, n_layers: int) -> int:
    """Hidden width so the MLP after the projection has ~the same params as the quantum circuit + readout.

    Quantum after projection: 2*n*L circuit angles + (n + 1) readout.
    MLP after projection:     n*h + h (hidden) + h + 1 (output) = h*(n + 2) + 1.
    """
    return max(1, round((2 * n_qubits * n_layers + n_qubits) / (n_qubits + 2)))


class MatchedMLPHead(nn.Module):
    """Baseline 2: same projection as the quantum head, then a small MLP with matched parameter count."""

    def __init__(self, in_dim: int, n_qubits: int, n_layers: int):
        super().__init__()
        h = matched_hidden(n_qubits, n_layers)
        self.proj = nn.Linear(in_dim, n_qubits)
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(n_qubits, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.proj(x)).squeeze(-1)
