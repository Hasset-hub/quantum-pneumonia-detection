import math

import pytest
import torch

from qpd.models import MatchedMLPHead, QuantumHead
from qpd.models.quantum_head import simulate_local
from qpd.utils import count_params


def test_single_qubit_ry_expectation():
    thetas = torch.linspace(0, math.pi, 7).unsqueeze(1)
    z = simulate_local(thetas, torch.zeros(0, 1, 2))
    assert torch.allclose(z[:, 0], torch.cos(thetas[:, 0]), atol=1e-5)


def test_cnot_ring():
    # RY(pi) puts qubit 0 in |1>; CNOT 0->1 flips qubit 1; CNOT 1->0 flips qubit 0 back.
    z = simulate_local(torch.tensor([[math.pi, 0.0]]), torch.zeros(1, 2, 2))
    assert torch.allclose(z, torch.tensor([[1.0, -1.0]]), atol=1e-5)


def test_parameter_shift_matches_autograd():
    torch.manual_seed(0)
    angles = torch.rand(1, 3)
    weights = (torch.rand(2, 3, 2) * 3).requires_grad_(True)
    simulate_local(angles, weights)[0, 0].backward()
    w = weights.detach()
    flat_idx = 4
    wp, wm = w.clone().view(-1), w.clone().view(-1)
    wp[flat_idx] += math.pi / 2
    wm[flat_idx] -= math.pi / 2
    shift = (simulate_local(angles, wp.view_as(w))[0, 0] - simulate_local(angles, wm.view_as(w))[0, 0]) / 2
    assert torch.isclose(weights.grad.view(-1)[flat_idx], shift, atol=1e-5)


@pytest.mark.parametrize("n,layers", [(4, 1), (4, 2), (6, 2), (8, 4)])
def test_heads_are_parameter_matched(n, layers):
    q, m = QuantumHead(512, n, layers), MatchedMLPHead(512, n, layers)
    assert abs(count_params(q) - count_params(m)) / count_params(q) < 0.05


def test_quantum_head_shapes_and_grads():
    head = QuantumHead(16, n_qubits=4, n_layers=2)
    out = head(torch.randn(5, 16))
    assert out.shape == (5,)
    out.sum().backward()
    assert head.weights.grad is not None and head.proj.weight.grad is not None
