from pathlib import Path
from ipie.config import config
from src.afqmc_runner  import *
from mpi4py import MPI
import os
import re
from datetime import datetime
import numpy as np
from mpi_info import MPIInfo


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mpi_info = MPIInfo(comm=comm,rank=rank,size=size)
config.update_option("use_gpu", False)

comm_size = comm.size if comm is not None else 1
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%y-%m-%d_%H-%M-%S")

VQE_DIR = 'VQE_Trials'
VQE_trial_filenames  = []   # e.g. 
max_det = None

verbose = rank==0


# 1024 wakers, 10 steps per block, 500 blcoks is default
AFQMC_PARAMS = AFQMCParams(num_total_walkers=1024,
    num_steps_per_block=10,
    num_blocks=200,
    timestep=0.01,
    stabilize_freq=5,
    pop_control_freq=5,
    comm_size=comm_size,
    verbose=verbose)


if VQE_trial_filenames == []:
   VQE_trial_filenames = os.listdir(VQE_DIR)


for entry_name in VQE_trial_filenames:
    full_path = os.path.join(VQE_DIR, entry_name)

    mol_problem, (coeffs,occa,occb), vqe_run_data = unpack_vqe_data(full_path,mpi_info=mpi_info)
    num_dets = len(coeffs)
    if verbose:
        print("# Num Full Determinents: ", num_dets)


    Trial = TrialWfn(coeffs=coeffs,
        occa=occa,
        occb=occb,
        mol_problem=mol_problem,
        max_det=max_det)


    this_afqmc_run = AFQMC_Event(mol_problem=mol_problem, afqmc_params=AFQMC_PARAMS)
    if verbose:
        print(f'# Reading from: {full_path}')
        print(f"# Num Truncated Determinents: {Trial.num_dets}")
        print("# HF Energy: ", mol_problem.get_hf_energy())
        print(f"# Final Energy from VQE: {vqe_run_data['energy_list'][-1]}")
        print(f"# Trial Variational energy:", Trial.build_PH_trial().energy)

    # run afqmc, 
    this_afqmc_run.run_afqmc(Trial)
    if verbose:
        print(f"""*************************************************************************** 
            Final AFQMC Energy with Auto-Correlation Corrected Error Bars 
            {this_afqmc_run.mean_ac} +- {this_afqmc_run.err_ac}) 
***************************************************************************""")




