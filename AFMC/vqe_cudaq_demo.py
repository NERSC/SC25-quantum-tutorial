from src.MoleleculeBuilder import BuildMoleculeProblem, build_h_chain_atom_string, jordan_wigner_fermion
from src.mpi_info import MPIInfo
from mpi4py import MPI
from datetime import datetime
import socket
import numpy as np, json
from src.vqe_ucc import * 
import time
from datetime import datetime
from pathlib import Path
import os
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mpi_info = MPIInfo(comm=comm,rank=rank,size=size)
option = "mqpu,fp64"
cudaq.set_target("nvidia", option = option)
verbose = (rank == 0)
now = datetime.now().astimezone()

if verbose:
    print(f"Started: {now:%Y-%m-%d %H:%M:%S %Z} | JobID: {os.getenv('SLURM_JOB_ID','N/A')}")

print(f"[rank {rank}/{size}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
      f"num_qpus={cudaq.get_target().num_qpus()}, host={socket.gethostname()}", flush=True)


SAVING = True
OUTPUT_DIR = "VQE_Trials"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

DEFAULTS = dict(
    total_h_atoms=6,      # 6 | 10
    h_spacing=1.0,        # 1.0 | 1.5 | 2.0
    ansatz="UCCSD",       # "UCCSD" | "UpCCD"
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




path = f"VQE_Ansatzes/{mol_id}/{ansatz_type}"
npz_path = f"{path}.npz"
meta_path = f"{path}.json"

with np.load(npz_path, allow_pickle=True) as data:
    pauli_words = data["pauli_words"].tolist()          
    coeffs = data["coeffs"].astype(float).tolist()  
    block_ids = data["block_ids"].astype(int).tolist() 
    init_point = data[f"init_{init_method}"].astype(float).tolist()
    excitation_list = data["excitation_list"].tolist() 


num_ansatz_params = len(excitation_list)

mol_data = mol_problem.get_mol_hamiltonian_from_fcidump()


ucc_spec = (num_qubit,THETA,pauli_words,coeffs,block_ids,mol_problem.active_orbitals,mol_problem.active_mol_nelec[0],mol_problem.active_mol_nelec[1])

max_walltime  = None
epsilon = 1e-3
opt_tol = 1e-7
optimizer = "L-BFGS-B"
amp_threshold = 1e-10
opt = MQPUGradients(
            kernel=ucc_circuit,
            hamiltonian=ham_spin_op,
            kernel_arg_spec=ucc_spec,
            init_theta=init_point,
            econst=econst,
            max_walltime=max_walltime,
            epsilon=epsilon,
            optimizer_method=optimizer,
            opt_tol=opt_tol,
            mpi_info=mpi_info,
            verbose = verbose
        )

if verbose:
  print(f"System: , H{total_h_atoms} (R = {h_spacing})")
  print(f"Ansatz, Initialization Method: {ansatz_type,init_method}")
  print("HF Energy: ", mol_problem.get_hf_energy())
  print(f"QUBITS: {num_qubit}")
  print("Num params: ", num_ansatz_params)
  print("Runtime Budget: ", max_walltime)



t1 = time.time()
results, energies = opt.optimize()
t2 = time.time()
converged = results.success
total_energy = results.fun + econst
opt_params = results.x
# print(f"optimized parameters: {opt_params}")
runtimes = opt.timing_summary()

eval_counts = opt.gather_eval_counts()
runtimes["eval_counts"] = eval_counts
runtimes["full_optim"] = t2 - t1



energy_list = energies
if verbose:
  state = convert_state_big_endian(
      np.array(cudaq.get_state(
          ucc_circuit,
          num_qubit,
          opt_params,
          pauli_words,
          coeffs,
          block_ids,
          mol_problem.active_orbitals,
          mol_problem.active_mol_nelec[0],
          mol_problem.active_mol_nelec[1]
      ), dtype=complex)
  )

  Statevector = BlockStatevector(statevector=state, mol_problem=mol_problem,ampl_eps=amp_threshold)

  t3 = time.time()
  runtimes["post_process"] = t3 - t2



  print("FCI Energy: ", {float(mol_problem.get_FCI_energy())})
  print(f"Converged: {converged}")



  print("FULL OPTIMIZATION TIME: ", runtimes["full_optim"])
  print("POST-PROCESS Runtime: ", runtimes["post_process"])
  for k,v in runtimes.items():
    print(f"{k}: {v}")


  print("num_energy_evals (global):", eval_counts["total"])
  print("num_energy_evals_per_rank (global):", eval_counts["per_rank"])
  if SAVING:
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%y-%m-%d_%H-%M-%S")
    
    output_directory = Path(OUTPUT_DIR)
    file_name = f'nodes{size}_{mol_id}_{ansatz_type}_{init_method}'
    # append storage time
    file_name += f'_{formatted_datetime}.npz'
    file_name = Path(file_name)
    dest = output_directory / file_name   

    atom = mol_problem.atom

    active_orbitals =  None
    active_mol_nelec = None
    wf = Statevector.getIPIEWavefunction()
   
    np.savez_compressed(
        dest,
        atom=atom,
        basis=mol_problem.basis,
        spin=mol_problem.spin,
        active_orbitals=active_orbitals,
        active_mol_nelec=active_mol_nelec,
        mol_identifier=mol_problem.mol_identifier,
        coeffs=wf[0],
        occ_a=wf[1],
        occ_b=wf[2],
        runtimes=runtimes,
        converged=converged,
        energy_list=energy_list,
        num_params=num_ansatz_params,
        amp_threshold = amp_threshold
    )

