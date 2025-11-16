import cudaq, cudaq_solvers as solvers
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo, mcscf
import numpy as np
import openfermion
from openfermion import FermionOperator, PolynomialTensor
from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner
from utils import create_molecule

# Create the molecular hamiltonian
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
#geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.59))]
#geometry = [('N', (0., 0., 0.)), ('N', (0., 0., 1.10))]
molecule = create_molecule(geometry, basis='sto-3g', charge=0, spin=0, n_active_orbitals=2, n_active_electrons=2)

# Get the number of qubits and electrons
numQubits = molecule.n_orbitals * 2
numElectrons = molecule.n_electrons

spin = 0
initialX = [-.2] * solvers.stateprep.get_num_uccsd_parameters(
    numElectrons, numQubits)


# Define the UCCSD ansatz
@cudaq.kernel
def ansatz(thetas: list[float]):
    q = cudaq.qvector(numQubits)
    for i in range(numElectrons):
        x(q[i])
    solvers.stateprep.uccsd(q, thetas, numElectrons, spin)


def cost(theta):

    exp_val = cudaq.observe(ansatz, molecule.hamiltonian, theta).expectation()

    return exp_val

exp_vals = []


def callback(xk):
    exp_vals.append(cost(xk))

option = "mgpu,fp64"
cudaq.set_target("nvidia", option = option)
result = minimize(cost,
                  initialX,
                  method='COBYLA',
                  callback=callback,
                  options={'maxiter': 300})

print('UCCSD-VQE energy =  ', result.fun)
print('Number of Variational Parameters =', len(initialX))

plt.plot(exp_vals)
plt.xlabel('Epochs')
plt.ylabel('Energy')
plt.title('VQE')
plt.show()
