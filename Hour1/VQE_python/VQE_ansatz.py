import cudaq, cudaq_solvers as solvers
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo, mcscf
import numpy as np
import openfermion
from openfermion import FermionOperator, PolynomialTensor
from openfermion import InteractionOperator, get_fermion_operator
from openfermion.transforms import jordan_wigner

def create_molecule(geometry,
                    basis='sto-3g',
                    charge=0,
                    spin=0,
                    n_active_orbitals=None,
                    n_active_electrons=None):

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    if n_active_orbitals is None:
        n_active_orbitals = mol.nao
    if n_active_electrons is None:
        n_active_electrons = mol.nelectron

    mc = mcscf.CASCI(mf, n_active_orbitals, n_active_electrons)
    mc.kernel()

    # Get correct CAS orbitals
    ncore = mc.ncore
    ncas  = mc.ncas
    mo = mc.mo_coeff[:, ncore:ncore+ncas]

    # One-electron integrals
    h1 = mo.T @ mf.get_hcore() @ mo

    # Two-electron integrals
    eri_ao = mol.intor("int2e")
    eri_mo = ao2mo.incore.full(eri_ao, mo)
    eri_mo = ao2mo.restore(1, eri_mo, ncas)

    # Build OpenFermion InteractionOperator
    h2 = np.transpose(eri_mo, (0,2,1,3))
    H_f = InteractionOperator(0.0, h1, h2)

    # Convert → FermionOperator
    H_ferm_op = get_fermion_operator(H_f)

    # JW → QubitOperator
    H_qubit_of = jordan_wigner(H_ferm_op)
    H_qubit_real=openfermion.ops.operators.qubit_operator.QubitOperator()
    for term, coeff in H_qubit_of.terms.items():
        H_qubit_real += openfermion.ops.operators.qubit_operator.QubitOperator(term, coeff.real)
    Spin_operator = cudaq.SpinOperator(H_qubit_real)

    # Build CUDA-Q SpinOperator
    #Spin_operator = cudaq.SpinOperator(H_qubit_of)

    class MolObj: pass
    m = MolObj()
    m.hamiltonian = Spin_operator
    m.n_orbitals = ncas
    m.n_electrons = n_active_electrons
    return m

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
