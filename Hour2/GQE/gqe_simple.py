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
import os, torch, numpy as np

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create the hamiltonian
spin_ham = spin.z(0)*spin.z(3)
n_qubits = 4

thetas = [np.pi/2, np.pi/4, np.pi/8, 3*np.pi/8, 3*np.pi/4, 3*np.pi/2]

# Assuming we only want to apply a sequence of rx gates
# Storing relevnat parameters for the rx gates as the op pool (qubits and thetas)
def pool(thetas):
    pool = []

    for i in range(n_qubits):
        for theta in thetas:
            pool.append({'qubit': i, 'theta': theta})

    return pool

op_pool = pool(thetas)

# Kernel that applies the selected operators
@cudaq.kernel
def kernel(n_qubits: int, thetas: list[float], qubits: list[int]):
    q = cudaq.qvector(n_qubits)

    for i in range(n_qubits//2):
        x(q[i])

    for i in range(len(thetas)):
        rx(thetas[i], q[qubits[i]])

# The transformer returned a list of our paramter dicts
# Since the kernel doesn't accept a list of dicts, we extract them here to pass in
def cost(sampled_ops: list[dict], **kwargs):
    thetas = [op['theta'] for op in sampled_ops]
    qubits = [op['qubit'] for op in sampled_ops]

    return cudaq.observe(kernel, spin_ham, n_qubits, thetas, qubits).expectation()

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
