#!/bin/bash
#SBATCH -N 1               # select number of nodes (powers of 2)
#SBATCH -n 1               # select number of tasks (powers of 2)  
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=None
#SBATCH -t 01:00:00
#SBATCH -q regular
#SBATCH -A m4916           # add account num
#SBATCH -C gpu          
#SBATCH -o mgpu_ucc-%j_%t.out
#SBATCH --image=nersc/cudaq:0.12.1

TOTAL_H_ATOMS=6      # 14 | 16
H_SPACING=1.0        # 1.0 | 1.5 | 2.0
ANSATZ="UpCCD"       # "UCCSD" | "UpCCD"
INIT="ccsd"          # ccsd | mp2 | zeros


srun -n 1 --cpu-bind=cores --gpu-bind=none --mpi=pmix --module cuda-mpich shifter  python mgpu_ucc.py \
  --total-h-atoms ${TOTAL_H_ATOMS} \
  --h-spacing ${H_SPACING} \
  --ansatz ${ANSATZ} \
  --init ${INIT} \
  --cudaq-full-stack-trace


