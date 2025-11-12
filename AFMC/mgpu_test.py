import os, time, subprocess, socket

from mpi4py import MPI
import cudaq
import argparse

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# import pynvml as nv
# nv.nvmlInit()
# h = nv.nvmlDeviceGetHandleByIndex(0)  
# def mem_gb():
#     return nv.nvmlDeviceGetMemoryInfo(h).used / (1024**3)


option = "mgpu,fp64"
cudaq.set_target("nvidia", option=option)

print(f"[rank {rank}/{size}] host={socket.gethostname()} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
      f"SLURM_LOCALID:{os.environ.get("SLURM_LOCALID")}, SLURM_PROCID: {os.environ.get("SLURM_PROCID", "0")}")

# print(f"[rank {rank}] host={socket.gethostname()} CVD={os.getenv('CUDA_VISIBLE_DEVICES')} mem_before={mem_gb():.2f} GiB")



n = 28      # SET num qubits

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(1, n):
        cx(q[0], q[i])

#### cudaq.sample()
t0 = time.time()
res = cudaq.sample(ghz, n, shots_count=1000)
dt = time.time() - t0

if rank == 0:
    print(f"{n}-qubit GHZ cudaq.sample(), option={option}, time={dt:.2f}s")
    res.dump()


#### cudaq.observe(), set shots to zero for analytic estimtaor, non-zero for sampling estimator
term_count = 10
hamiltonian = cudaq.SpinOperator.random(n, term_count, seed = 44)
shots = 0
result = cudaq.observe(ghz, hamiltonian, n,shots_count=shots).expectation()
if rank == 0:
    print("Observe_shots: ", shots)
    print("Energy: ",result)

# time.sleep(0.5)
# print(f"[rank {rank}] mem_after={mem_gb():.2f} GiB  delta={mem_gb():.2f}")