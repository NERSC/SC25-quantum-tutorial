import cudaq
from cudaq import spin
import cudaq_solvers as solvers

# Define quantum kernel (ansatz)
@cudaq.kernel
def ansatz(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    x.ctrl(q[1], q[0])


# Define Hamiltonian
H = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - \
    2.1433 * spin.y(0) * spin.y(1) + \
    0.21829 * spin.z(0) - 6.125 * spin.z(1)

print(cudaq.draw(ansatz, 0.0))
print(f"Initial state energy: {cudaq.observe(ansatz, H, 0.0).expectation()}")

# Run VQE with defaults (cobyla optimizer)
energy, parameters, data = solvers.vqe(
    lambda thetas: ansatz(thetas[0]),
    H,
    initial_parameters=[0.0],
    verbose=True
)

print(cudaq.draw(ansatz, parameters[0]))
print(f"Ground state energy: {energy}")
