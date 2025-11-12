#!/usr/bin/env bash

# to be run in /SC_Tutorial
set -e  # stop if anything fails
module load conda
conda create -y -n SC_Tutorial_env python=3.12.11
conda activate SC_Tutorial_env
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
cd ipie
pip install -e .
cd ..
pip install pyscf
pip install cudaq
mv ./distributed_interfaces $HOME/
