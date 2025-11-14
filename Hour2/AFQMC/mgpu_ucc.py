from src.MoleleculeBuilder import BuildMoleculeProblem, build_h_chain_atom_string, jordan_wigner_fermion
from src.mpi_info import MPIInfo
from src.vqe_ucc import * 
from  pathlib import Path
import cudaq
import time
import os, time, socket
import argparse
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

option = "mgpu,fp64"
cudaq.set_target("nvidia", option = option)

print(f"[rank {rank}/{size}] host={socket.gethostname()} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
      f"SLURM_LOCALID:{os.environ.get("SLURM_LOCALID")}, SLURM_PROCID: {os.environ.get("SLURM_PROCID", "0")}")

DEFAULTS = dict(
    total_h_atoms=6,      # 6 | 10
    h_spacing=1.0,        # 1.0 | 1.5 | 2.0
    ansatz="UpCCD",       # "UCCSD" | "UpCCD"
    init="ccsd",          # "ccsd" | "mp2" | "zeros"
)

parser = argparse.ArgumentParser(description="Run CUDA-Q VQE on H-chain.")
parser.add_argument(
    "--total-h-atoms", type=int, default=DEFAULTS["total_h_atoms"],
    help=f"Total number of H atoms (default: {DEFAULTS['total_h_atoms']})."
)
parser.add_argument(
    "--h-spacing", type=float, default=DEFAULTS["h_spacing"],
    help=f"Inter-atomic spacing (default: {DEFAULTS['h_spacing']})."
)
parser.add_argument(
    "--ansatz", type=str, choices=["UCCSD", "UpCCD"], default=DEFAULTS["ansatz"],
    help=f"Ansatz type (default: {DEFAULTS['ansatz']})."
)
parser.add_argument(
    "--init", type=str, choices=["ccsd", "mp2", "zeros"], default=DEFAULTS["init"],
    help=f"Initialization method (default: {DEFAULTS['init']})."
)
parser.add_argument("--cudaq-full-stack-trace", action="store_true",
               help="Enable CUDA-Q full stack trace (optional)")

args = parser.parse_args()

total_h_atoms = args.total_h_atoms
h_spacing     = args.h_spacing
ansatz_type   = args.ansatz
init_method   = args.init


h_chain = build_h_chain_atom_string(total_h_atoms,h_spacing)
mol_id = f"H{total_h_atoms}_{h_spacing}_chain"
mol_problem  = BuildMoleculeProblem(atom=h_chain,basis="sto-6g",spin=0,mol_identifier=mol_id)
obi,tbi, econst, electron_count,norbitals, fer_ham = mol_problem.get_mol_hamiltonian_from_fcidump()
num_qubit = 2 * norbitals
ham_spin_op = jordan_wigner_fermion(obi,tbi,ecore=0.0,tolerance=1e-15)

mol_problem = BuildMoleculeProblem(atom=h_chain,basis="sto-6g", spin=0,mol_identifier=f"H{total_h_atoms}_{h_spacing}_chain")

path = f"VQE_Ansatzes/{mol_id}/{ansatz_type}"
npz_path = f"{path}.npz"
meta_path = f"{path}.json"

with np.load(npz_path, allow_pickle=True) as data:
    pauli_words = data["pauli_words"].tolist()          
    coeffs = data["coeffs"].astype(float).tolist()  
    block_ids = data["block_ids"].astype(int).tolist() 
    init_point = data[f"init_{init_method}"].astype(float).tolist()
    excitation_list = data["excitation_list"].tolist() 

# print("excitation_list", excitation_list[:10])
# print(len(excitation_list))
# print("param_vector: ",init_point[:10])
# print(len(init_point))

num_ansatz_params = len(excitation_list)

mol_data = mol_problem.get_mol_hamiltonian_from_fcidump()


ucc_spec = (num_qubit,init_point,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])

t0 = time.perf_counter()
energy = cudaq.observe(ucc_circuit,ham_spin_op,*ucc_spec).expectation() + econst
t1 = time.perf_counter()
if rank == 0:
  print(f"H{total_h_atoms} Qubits: ", num_qubit)
  print("Energy: ", energy)
  print("runtime: ", t1-t0)