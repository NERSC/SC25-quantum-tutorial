# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]

# GQE is an optional component of the CUDA-QX Solvers Library. To install its
# dependencies, run:
# pip install cudaq-solvers[gqe]
#
# Run this script with
# python3 gqe_simple.py

import cudaq

cudaq.set_target('nvidia', option='fp64')
#cudaq.set_target('qpp-cpu')

import cudaq_solvers as solvers
from cudaq import spin

from lightning.fabric.loggers import CSVLogger
from cudaq_solvers.gqe_algorithm.gqe import get_default_config

# Set deterministic seed and environment variables for deterministic behavior
# Disable this section for non-deterministic behavior
import os, torch

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create the hamiltonian
spin_ham = spin.z(0)*spin.z(3)
n_qubits = 4

# Generate the operator pool consisting of e^{i*P_j*t_j}

# These are the coefs t_j's
params = [0.05, -0.05, 0.1, -0.1]

def pool(params):
    ops = []
    i = 0

    ops.append(cudaq.SpinOperator(spin.y(i) * spin.z(i + 1) * spin.x(i + 2) * spin.i(i + 3)))
    ops.append(cudaq.SpinOperator(spin.x(i) * spin.z(i + 1) * spin.y(i + 2) * spin.i(i + 3)))
    ops.append(cudaq.SpinOperator(spin.y(i) * spin.y(i + 1) * spin.y(i + 2) * spin.x(i + 3)))

    pool = []
    for c in params:
        for op in ops:
            pool.append(c * op)

    return pool


op_pool = pool(params)

def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    return [term.get_pauli_word(n_qubits) for term in op]


# Kernel that applies the selected operators
@cudaq.kernel
def kernel(n_qubits: int, coeffs: list[float],
           words: list[cudaq.pauli_word]):
    q = cudaq.qvector(n_qubits)

    for i in range(n_qubits//2):
        x(q[i])

    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):

    full_coeffs = []
    full_words = []

    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

        return cudaq.observe(kernel, spin_ham, n_qubits, full_coeffs,
                            full_words).expectation()


# Configure GQE
cfg = get_default_config()
cfg.use_fabric_logging = False
logger = CSVLogger("gqe_simple_logs/gqe.csv")
cfg.fabric_logger = logger
cfg.save_trajectory = False
cfg.verbose = True
cfg.max_epochs = 10
print(dir(cfg))

# Run GQE
minE, best_ops = solvers.gqe(cost, op_pool, max_iters=3, ngates=5, config=cfg)

# Only print results from rank 0 when using MPI
print(f'Ground Energy = {minE}')
print('Final optimized operators for the ansatz are')
for idx in best_ops:
    # Get the first (and only) term since these are simple operators
    term = next(iter(op_pool[idx]))
    print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))
