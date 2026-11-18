"""Sanity check: the QuantumRings simulator and the local statevector simulator agree.

Usage (after filling in .env):  python scripts/check_quantumrings.py
Expect differences of a few hundredths from shot noise. A large mismatch usually means the
bitstring ordering assumption in QuantumRingsRunner.expectations needs flipping.
"""
import torch
from dotenv import load_dotenv

from qpd.models.quantum_head import QuantumRingsRunner, simulate_local

load_dotenv()
torch.manual_seed(0)
n, L = 4, 2
angles = torch.rand(1, n) * 3.14
weights = torch.rand(L, n, 2) * 3.14
local = simulate_local(angles, weights)[0].tolist()
remote = QuantumRingsRunner(shots=8192).expectations(angles[0].tolist(), weights.tolist())
for i, (a, b) in enumerate(zip(local, remote)):
    print(f"qubit {i}: local={a:+.3f}  quantumrings={b:+.3f}  diff={abs(a - b):.3f}")
