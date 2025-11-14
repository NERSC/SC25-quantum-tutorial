#!/bin/bash
#SBATCH -A m4916
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1              # Specify number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH -o vqe_output-%j_%t.out

module load cray-mpich
module load conda
conda activate SC_Tutorial_env


# parameter values
TOTAL_H_ATOMS=6      # 6 | 10
H_SPACING=1.0        # 1.0 | 1.5 | 2.0
ANSATZ="UCCSD"       # "UCCSD" | "UpCCD"
INIT="ccsd"          # ccsd | mp2 | zeros

# Launch (one MPI rank per node; each rank sees 4 GPUs)
srun python vqe_cudaq_demo.py \
  --total-h-atoms ${TOTAL_H_ATOMS} \
  --h-spacing ${H_SPACING} \
  --ansatz ${ANSATZ} \
  --init ${INIT} \
  --cudaq-full-stack-trace
