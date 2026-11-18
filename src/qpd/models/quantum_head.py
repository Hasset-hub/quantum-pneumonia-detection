"""Variational quantum classification head.

Circuit (n qubits, L layers):
    angle encoding   RY(a_i) on qubit i, with a = pi * tanh(W x + b)
    for each layer:  RY(w[l,i,0]) RZ(w[l,i,1]) on every qubit, then a CNOT ring i -> i+1 (mod n)
    readout          <Z_i> for every qubit -> Linear(n, 1) -> logit

Two interchangeable backends run the same circuit:
    "local"        exact statevector simulation in PyTorch. Differentiable with autograd, runs on GPU,
                   no account needed. Used for fast training sweeps and for CI.
    "quantumrings" the QuantumRings SDK simulator with finite shots. Gradients use the
                   parameter-shift rule, so every gradient step costs 2 * (n + 2nL) + 1 circuit runs
                   per sample. Use it on small data fractions and for final evaluation.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------------------
# Local exact statevector simulator (batched, differentiable)
# --------------------------------------------------------------------------------------------------


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    e = torch.exp(-0.5j * theta.to(torch.complex64))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


def _apply_1q(state: torch.Tensor, gate: torch.Tensor, q: int) -> torch.Tensor:
    """state: [B, 2, ..., 2]; gate: [B, 2, 2] or [2, 2]."""
    if gate.dim() == 2:
        gate = gate.expand(state.shape[0], 2, 2)
    state = state.movedim(q + 1, -1)
    state = torch.einsum("b...j,bij->b...i", state, gate)
    return state.movedim(-1, q + 1)


def _apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    s0 = state.narrow(control + 1, 0, 1)
    s1 = state.narrow(control + 1, 1, 1).flip(target + 1)
    return torch.cat([s0, s1], dim=control + 1)


def simulate_local(angles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """angles: [B, n]; weights: [L, n, 2]. Returns <Z_i>: [B, n]."""
    b, n = angles.shape
    state = torch.zeros((b,) + (2,) * n, dtype=torch.complex64, device=angles.device)
    state.view(b, -1)[:, 0] = 1.0
    for i in range(n):
        state = _apply_1q(state, _ry(angles[:, i]), i)
    for layer in weights:
        for i in range(n):
            state = _apply_1q(state, _ry(layer[i, 0]), i)
            state = _apply_1q(state, _rz(layer[i, 1]), i)
        if n > 1:
            for i in range(n):
                state = _apply_cnot(state, i, (i + 1) % n)
    probs = state.abs() ** 2
    out = []
    for i in range(n):
        dims = [d for d in range(1, n + 1) if d != i + 1]
        p = probs.sum(dim=dims) if dims else probs
        out.append(p[:, 0] - p[:, 1])
    return torch.stack(out, dim=1).float()


# --------------------------------------------------------------------------------------------------
# QuantumRings backend
# --------------------------------------------------------------------------------------------------


class QuantumRingsRunner:
    """Builds and runs the ansatz on the QuantumRings simulator.

    Credentials come from the environment (see .env.example): QR_TOKEN, QR_ACCOUNT, QR_BACKEND.
    """

    def __init__(self, shots: int = 1024):
        import QuantumRingsLib as qrl

        self.qrl = qrl
        self.shots = shots
        self.provider = qrl.QuantumRingsProvider(
            token=os.environ["QR_TOKEN"], name=os.environ["QR_ACCOUNT"]
        )
        self.backend = self.provider.get_backend(os.environ.get("QR_BACKEND", "scyllax"))

    def expectations(self, angles, weights) -> list[float]:
        """angles: length-n sequence; weights: [L][n][2] nested sequence. Returns <Z_i> per qubit."""
        qrl, n = self.qrl, len(angles)
        q = qrl.QuantumRegister(n, "q")
        c = qrl.ClassicalRegister(n, "c")
        qc = qrl.QuantumCircuit(q, c)
        for i in range(n):
            qc.ry(float(angles[i]), i)
        for layer in weights:
            for i in range(n):
                qc.ry(float(layer[i][0]), i)
                qc.rz(float(layer[i][1]), i)
            if n > 1:
                for i in range(n):
                    qc.cx(i, (i + 1) % n)
        for i in range(n):
            qc.measure(i, i)
        counts = self.backend.run(qc, shots=self.shots).result().get_counts()
        total = sum(counts.values())
        z = [0.0] * n
        for bits, k in counts.items():
            bits = bits.replace(" ", "")
            for i in range(n):
                # Qiskit-style ordering: rightmost character is qubit 0.
                z[i] += (1 if bits[-1 - i] == "0" else -1) * k / total
        return z


class _ParameterShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, weights, runner):
        ctx.save_for_backward(angles, weights)
        ctx.runner = runner
        w = weights.detach().cpu().tolist()
        out = [runner.expectations(a, w) for a in angles.detach().cpu().tolist()]
        return torch.tensor(out, dtype=torch.float32, device=angles.device)

    @staticmethod
    def backward(ctx, grad_out):
        angles, weights = ctx.saved_tensors
        runner, shift = ctx.runner, math.pi / 2
        a_np, w_np = angles.detach().cpu(), weights.detach().cpu()
        g_out = grad_out.detach().cpu()
        grad_a = torch.zeros_like(a_np)
        grad_w = torch.zeros_like(w_np)
        for b in range(a_np.shape[0]):
            a = a_np[b]
            # d<Z>/d(theta) = [f(theta + pi/2) - f(theta - pi/2)] / 2
            for i in range(a.numel()):
                ap, am = a.clone(), a.clone()
                ap[i] += shift
                am[i] -= shift
                d = (torch.tensor(runner.expectations(ap.tolist(), w_np.tolist()))
                     - torch.tensor(runner.expectations(am.tolist(), w_np.tolist()))) / 2
                grad_a[b, i] = (d * g_out[b]).sum()
            for idx in range(w_np.numel()):
                wp, wm = w_np.clone().view(-1), w_np.clone().view(-1)
                wp[idx] += shift
                wm[idx] -= shift
                d = (torch.tensor(runner.expectations(a.tolist(), wp.view_as(w_np).tolist()))
                     - torch.tensor(runner.expectations(a.tolist(), wm.view_as(w_np).tolist()))) / 2
                grad_w.view(-1)[idx] += (d * g_out[b]).sum()
        return grad_a.to(angles.device), grad_w.to(weights.device), None


# --------------------------------------------------------------------------------------------------
# Head module
# --------------------------------------------------------------------------------------------------


class QuantumHead(nn.Module):
    def __init__(self, in_dim: int, n_qubits: int = 4, n_layers: int = 2,
                 backend: str = "local", shots: int = 1024):
        super().__init__()
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.proj = nn.Linear(in_dim, n_qubits)
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 2))
        self.readout = nn.Linear(n_qubits, 1)
        self._runner = None
        self.set_backend(backend, shots)

    def set_backend(self, backend: str, shots: int = 1024) -> None:
        if backend not in ("local", "quantumrings"):
            raise ValueError(f"unknown quantum backend: {backend}")
        self.backend, self.shots = backend, shots
        if backend == "quantumrings" and self._runner is None:
            self._runner = QuantumRingsRunner(shots)

    def circuit(self, angles: torch.Tensor) -> torch.Tensor:
        if self.backend == "local":
            return simulate_local(angles, self.weights)
        return _ParameterShift.apply(angles, self.weights, self._runner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = math.pi * torch.tanh(self.proj(x))
        return self.readout(self.circuit(angles)).squeeze(-1)
