#!/bin/bash
#SBATCH -N 1               # select number of nodes (powers of 2)
#SBATCH -n 4               # select number of tasks (powers of 2)  
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=None
#SBATCH -t 00:30:00
#SBATCH -q debug
#SBATCH -A m2651           # add account num
#SBATCH -C gpu          
#SBATCH -o mgpu_test-%j_%t.out
#SBATCH --image=nersc/cudaq:0.12.1

srun -n 4 --cpu-bind=cores --gpu-bind=none --mpi=pmix --module cuda-mpich shifte python mgpu_test.py --cudaq-full-stack-trace


