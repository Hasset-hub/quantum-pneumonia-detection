from .backbone import build_backbone
from .classical_head import LogisticHead, MatchedMLPHead
from .quantum_head import QuantumHead

__all__ = ["build_backbone", "LogisticHead", "MatchedMLPHead", "QuantumHead", "build_head"]


def build_head(name: str, in_dim: int, n_qubits: int = 4, n_layers: int = 2, **qkw):
    if name == "logreg":
        return LogisticHead(in_dim)
    if name == "mlp":
        return MatchedMLPHead(in_dim, n_qubits, n_layers)
    if name == "quantum":
        return QuantumHead(in_dim, n_qubits, n_layers, **qkw)
    raise ValueError(f"unknown head: {name}")
