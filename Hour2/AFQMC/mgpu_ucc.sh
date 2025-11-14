#!/bin/bash
#SBATCH -N 1               # select number of nodes (powers of 2)
#SBATCH -n 4               # select number of tasks (powers of 2)  
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=None
#SBATCH -t 01:00:00
#SBATCH -q regular
#SBATCH -A m4916           # add account num
#SBATCH -C gpu          
#SBATCH -o mgpu_ucc-%j_%t.out

module load cray-mpich

# create new conda env, pip install cudaq
# make sure mpi4py is build against cray-mpich
# to install this mpi4py in your conda env, run: MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
module load conda
conda activate SC_Tutorial_env


export MPI_PATH="$CRAY_MPICH_DIR"


# copy directory in $Home via: cp -r /opt/nvidia/cudaq/distributed_interfaces/ $HOME/
# based on https://github.com/zohimchandani/cudaq-perlmutter/tree/main
source $HOME/distributed_interfaces/activate_custom_mpi.sh  
export CUDAQ_MGPU_LIB_MPI="$CUDAQ_MPI_COMM_LIB"
# export LD_LIBRARY_PATH=$HOME:$LD_LIBRARY_PATH
# parameter values
TOTAL_H_ATOMS=6      # 14 | 16
H_SPACING=1.0        # 1.0 | 1.5 | 2.0
ANSATZ="UpCCD"       # "UCCSD" | "UpCCD"
INIT="ccsd"          # ccsd | mp2 | zeros


srun python mgpu_ucc.py \
  --total-h-atoms ${TOTAL_H_ATOMS} \
  --h-spacing ${H_SPACING} \
  --ansatz ${ANSATZ} \
  --init ${INIT} \
  --cudaq-full-stack-trace


