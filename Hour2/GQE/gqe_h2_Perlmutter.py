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
# python3 gqe_h2.py
#
# In order to leverage CUDA-Q MQPU and distribute the work across
# multiple QPUs (thereby observing a speed-up), run with:
#
# mpiexec -np N and vary N to see the speedup...
# e.g. PMIX_MCA_gds=hash mpiexec -np 2 python3 gqe_h2.py --mpi

import cudaq, cudaq_solvers as solvers
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo, mcscf
import numpy as np
import openfermion
from openfermion import FermionOperator, PolynomialTensor
from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner

def create_molecule(geometry,
                    basis='sto-3g',
                    charge=0,
                    spin=0,
                    n_active_orbitals=None,
                    n_active_electrons=None):

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    if n_active_orbitals is None:
        n_active_orbitals = mol.nao
    if n_active_electrons is None:
        n_active_electrons = mol.nelectron

    mc = mcscf.CASCI(mf, n_active_orbitals, n_active_electrons)
    mc.kernel()

    # Get correct CAS orbitals
    ncore = mc.ncore
    ncas  = mc.ncas
    mo = mc.mo_coeff[:, ncore:ncore+ncas]

    # One-electron integrals
    h1 = mo.T @ mf.get_hcore() @ mo

    # Two-electron integrals
    eri_ao = mol.intor("int2e")
    eri_mo = ao2mo.incore.full(eri_ao, mo)
    eri_mo = ao2mo.restore(1, eri_mo, ncas)

    # Build OpenFermion InteractionOperator
    h2 = np.transpose(eri_mo, (0,2,1,3))
    H_f = InteractionOperator(0.0, h1, h2)

    # Convert → FermionOperator
    H_ferm_op = get_fermion_operator(H_f)

    # JW → QubitOperator
    H_qubit_of = jordan_wigner(H_ferm_op)
    H_qubit_real=openfermion.ops.operators.qubit_operator.QubitOperator()
    for term, coeff in H_qubit_of.terms.items():
        H_qubit_real += openfermion.ops.operators.qubit_operator.QubitOperator(term, coeff.real)
    Spin_operator = cudaq.SpinOperator(H_qubit_real)

    # Build CUDA-Q SpinOperator
    #Spin_operator = cudaq.SpinOperator(H_qubit_of)

    class MolObj: pass
    m = MolObj()
    m.hamiltonian = Spin_operator
    m.n_orbitals = ncas
    m.n_electrons = n_active_electrons
    return m

import argparse, cudaq
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument('--mpi', action='store_true')
args = parser.parse_args()

if args.mpi:
    try:
        cudaq.set_target('nvidia', option='mqpu')
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #cudaq.mpi.initialize()
    except RuntimeError:
        print(
            'Warning: NVIDIA GPUs or MPI not available, unable to use CUDA-Q MQPU. Skipping...'
        )
        exit(0)
else:
    try:
        cudaq.set_target('nvidia', option='fp64')
    except RuntimeError:
        cudaq.set_target('qpp-cpu')

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

# Create the molecular hamiltonian
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule = create_molecule(geometry, 'sto-3g', 0, 0, n_active_orbitals=2, n_active_electrons=2)

spin_ham = molecule.hamiltonian
n_qubits = molecule.n_orbitals * 2
n_electrons = molecule.n_electrons

# Generate the operator pool

params = [
    0.003125, -0.003125, 0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025,
    0.05, -0.05, 0.1, -0.1
]


#def pool(params):
#    ops = []
#    i = 0
#
#    ops.append(
#        cudaq.SpinOperator(
#            spin.y(i) * spin.z(i + 1) * spin.x(i + 2) * spin.i(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.x(i) * spin.z(i + 1) * spin.y(i + 2) * spin.i(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.i(i) * spin.y(i + 1) * spin.z(i + 2) * spin.x(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.i(i) * spin.x(i + 1) * spin.z(i + 2) * spin.y(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.x(i) * spin.x(i + 1) * spin.x(i + 2) * spin.y(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.x(i) * spin.x(i + 1) * spin.y(i + 2) * spin.x(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.x(i) * spin.y(i + 1) * spin.y(i + 2) * spin.y(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.y(i) * spin.x(i + 1) * spin.y(i + 2) * spin.y(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.x(i) * spin.y(i + 1) * spin.x(i + 2) * spin.x(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.y(i) * spin.x(i + 1) * spin.x(i + 2) * spin.x(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.y(i) * spin.y(i + 1) * spin.x(i + 2) * spin.y(i + 3)))
#    ops.append(
#        cudaq.SpinOperator(
#            spin.y(i) * spin.y(i + 1) * spin.y(i + 2) * spin.x(i + 3)))
#
#    pool = []
#    for c in params:
#        for op in ops:
#            pool.append(c * op)
#
#    return pool
#
#
#op_pool = pool(params)
#
#
#def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
#    return [term.evaluate_coefficient() for term in op]
#
#
#def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
#    return [term.get_pauli_word(n_qubits) for term in op]
#
#
## Kernel that applies the selected operators
#@cudaq.kernel
#def kernel(n_qubits: int, n_electrons: int, coeffs: list[float],
#           words: list[cudaq.pauli_word]):
#    q = cudaq.qvector(n_qubits)
#
#    for i in range(n_electrons):
#        x(q[i])
#
#    for i in range(len(coeffs)):
#        exp_pauli(coeffs[i], q, words[i])
#
#
#def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
#
#    full_coeffs = []
#    full_words = []
#
#    for op in sampled_ops:
#        full_coeffs += [c.real for c in term_coefficients(op)]
#        full_words += term_words(op)
#
#    if args.mpi:
#        handle = cudaq.observe_async(kernel,
#                                     spin_ham,
#                                     n_qubits,
#                                     n_electrons,
#                                     full_coeffs,
#                                     full_words,
#                                     qpu_id=kwargs['qpu_id'])
#        return handle, lambda res: res.get().expectation()
#    else:
#        return cudaq.observe(kernel, spin_ham, n_qubits, n_electrons,
#                             full_coeffs, full_words).expectation()
#
#
## Configure GQE
#cfg = get_default_config()
#cfg.use_fabric_logging = False
#logger = CSVLogger("gqe_h2_logs/gqe.csv")
#cfg.fabric_logger = logger
#cfg.save_trajectory = False
#cfg.verbose = True
#
## Run GQE
#minE, best_ops = solvers.gqe(cost, op_pool, max_iters=25, ngates=10, config=cfg)
#
## Only print results from rank 0 when using MPI
#if not args.mpi or rank == 0:
#    print(f'Ground Energy = {minE}')
#    print('Ansatz Ops')
#    for idx in best_ops:
#        # Get the first (and only) term since these are simple operators
#        term = next(iter(op_pool[idx]))
#        print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))
#
##if args.mpi:
##    cudaq.mpi.finalize()
