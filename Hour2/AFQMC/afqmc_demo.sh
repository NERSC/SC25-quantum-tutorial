#!/bin/bash
#SBATCH -A m4916
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1          
#SBATCH -n 32               # specify number of tasks
#SBATCH -o afqmc_output-%j_%t.out

module load cray-mpich
module load conda
conda activate SC_Tutorial_env




# Launch (one MPI rank per node; each rank sees 4 GPUs)
srun python afqmc_demo.py