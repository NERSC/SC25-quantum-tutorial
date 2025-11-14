from __future__ import annotations
from pyscf import ao2mo
from pyscf import gto, scf,fci, ao2mo,mcscf,lib,tools
from typing import Tuple, Optional
import hashlib
import numpy as np
from pyscf.tools.fcidump import read
from pyscf.tools import fcidump as fcd
from ipie.hamiltonians.generic  import Generic as HamGeneric
from  pathlib import Path
import os
import pyscf.cc as cc


class BuildMoleculeProblem:

  def __init__(self,
               atom : str,
               basis : str ,
               spin: int,
               active_orbitals: Optional[int] = None,
               active_mol_nelec: Optional[Tuple[int,int]] = None,
               charge: int = 0,
               unit : Optional[str] = 'angstrom',
               mol_identifier: Optional[str] = None,
               mpi_info = None,
               fci_dump_path = None):
    self.atom = atom
    self.basis = basis
    self.spin = spin
    self.mol_identifier = mol_identifier
    self.mpi_info = mpi_info
    self.fci_dump_path = fci_dump_path if fci_dump_path is None else Path(fci_dump_path)

    if self.mol_identifier is None:
       self.mol_identifier = f"{self.atom}"
    if self.mpi_info is None:
        self.rank = 0
        self.size = 1
    else:
        self.rank = self.mpi_info.rank
        self.size  = self.mpi_info.size
    self.chk_file_path = Path(f"chk_files/{self.mol_identifier}_{self.basis}_{self.spin}.chk")
    self.unit = unit
    self.charge = charge

    self.mol = self.PySCF_mol()


    self.active_orbitals = self.mol.nao_nr()
    self.active_mol_nelec = self.mol.nelec
    self.active_space = False

    if self.fci_dump_path is None:
        if self.active_space:
            self.fci_dump_path = Path(f"fci_dump_files/{self.mol_identifier}_{self.basis}_{self.spin}_AO{self.active_orbitals}_AE{self.active_mol_nelec[0]}-{self.active_mol_nelec[1]}.FCIDUMP")
        else:
            self.fci_dump_path = Path(f"fci_dump_files/{self.mol_identifier}_{self.basis}_{self.spin}.FCIDUMP")

    self.mf = self.build_scf()
    self.active_electrons = self.active_mol_nelec[0] + self.active_mol_nelec[1]
    self.active_spin_orbitals = self.active_orbitals * 2
    if self.rank == 0:
        if not self.fci_dump_path.exists():
            print(f"Creating FCI_Dump full path: {self.fci_dump_path.resolve()}")

            self.fci_dump_path.parent.mkdir(parents=True, exist_ok=True)

            self.build_fci_dump()
        else:
           print(f"Using FCI_Dump full path: {self.fci_dump_path.resolve()}")

    self.FCI = None
    self.ipie_ham = None
    self.ccsd = None
  def PySCF_mol(self):
    self.mol = gto.M(
          atom=self.atom,
          basis=self.basis,
          unit=self.unit,
          verbose=0,
          spin=self.spin,
          charge = self.charge
      )
    return self.mol
  @property
  def full_spatial_orbitals(self):
    return self.mol.nao_nr()
  @property
  def mol_nelec(self):
    return self.mol.nelec
  @property
  def full_electrons(self):
    return sum(self.mol.nelec)
  
  def frozen_spatial_orbitals(self):
    nspatial = self.mf.mo_coeff.shape[1]
    return [i for i in range(nspatial) if i not in self.active_spatial]

  def build_ccsd(self):
      if self.ccsd is None:
        frz = self.frozen_spatial_orbitals()
        c = cc.CCSD(self.mf, frozen=frz)
        c.kernel()
        self.ccsd = c
      return self.ccsd

  def sparse_pauli_h(self):
      # already have: self.get_of_jw_hamiltonian()
      problemJW = self.get_of_jw_hamiltonian()
      num_qubits = max(i for term in problemJW.terms for (i, _) in term) + 1
      plist, coeffs = [], []
      for key, val in problemJW.terms.items():
          pterm = ['I'] * num_qubits
          for (q, P) in key: pterm[q] = P
          plist.append(''.join(pterm))
          coeffs.append(val)
      from qiskit.quantum_info import SparsePauliOp
      return SparsePauliOp(plist, coeffs)
  
  def build_scf(self):
    # Pick RHF/ROHF based on spin
    chk = Path(self.chk_file_path)

    scf_cls = scf.ROHF if self.mol.spin else scf.RHF

    if chk.exists():
        if self.rank == 0:
            print(f"Checkpoint full path: {chk.resolve()}")

        data = lib.chkfile.load(str(chk), 'scf')  # dict of arrays/scalars
        mf = scf_cls(self.mol)
        # choose fields you need
        for k in ('mo_coeff','mo_occ','mo_energy','e_tot','converged', "hcore", "X","mol"):
            if k in data:
                # print(f"{k}:{data[k]}")
                v = data[k]
                setattr(mf, k, np.asarray(v) if hasattr(v, 'shape') else v)
        return mf
    self.chk_file_path.parent.mkdir(parents=True, exist_ok=True)
    mf = scf_cls(self.mol)
    if self.rank == 0:
        print(f"chk doesn't exist, writing in {os.getcwd()}", flush=True)
        self.chk_file_path.parent.mkdir(parents=True, exist_ok=True)
        mf.chkfile = str(self.chk_file_path)
    mf.kernel()
    return mf

  def build_fci_dump(self):
    fci_path = self.fci_dump_path
    if self.active_space:
        ncas = self.active_orbitals
        nelecas_total = self.active_electrons  # int is fine
        mc = mcscf.CASSCF(self.mf, ncas, nelecas_total).run()
        tools.fcidump.from_mcscf(mc, str(fci_path))
    else:
        tools.fcidump.from_scf(self.mf, str(fci_path))
       
       


  
  def get_hf_energy(self):
    return self.mf.energy_tot()
  # need .build() to be called

  def build_ipie_ham_from_fcidump(self, eig_thresh: float = 1e-12, verbose: bool = False) -> HamGeneric:
    fcidump_path = self.fci_dump_path
    res = read(str(fcidump_path), verbose=verbose)  # your read() from above
    norb  = int(res['NORB'])
    h1e   = np.asarray(res['H1'], dtype=float)
    h2pk  = np.asarray(res['H2'], dtype=float)      # packed pair-space
    ecore = float(res.get('ECORE', 0.0))

    assert h1e.shape == (norb, norb)

    # Unpack pair-space supermatrix
    npair = norb * (norb + 1) // 2
    V_pair = np.zeros((npair, npair), dtype=float)
    k = 0
    for a in range(npair):
        for b in range(a + 1):
            val = h2pk[k]
            V_pair[a, b] = val
            V_pair[b, a] = val
            k += 1
    V_pair = 0.5 * (V_pair + V_pair.T)

    # Map unordered pair index to ordered (p,q)
    def pair_index(i: int, j: int) -> int:
        if i < j:
            i, j = j, i
        return i * (i + 1) // 2 + j

    pair_idx = np.empty((norb, norb), dtype=int)
    for p in range(norb):
        for q in range(norb):
            pair_idx[p, q] = pair_index(p, q)

    flat_map = pair_idx.reshape(norb * norb)
    V_full = V_pair[np.ix_(flat_map, flat_map)]
    V_full = 0.5 * (V_full + V_full.T)

    # Eigendecomp → Cholesky vectors
    w, U = np.linalg.eigh(V_full)          # U: (norb^2, norb^2)
    keep = w > max(eig_thresh, 0.0)
    if not np.any(keep):
        raise RuntimeError("No positive eigenvalues in ERI supermatrix; check FCIDUMP.")
    w_keep = w[keep]                        # (k,)
    U_keep = U[:, keep]                     # (norb^2, k)

    # FIXED ORIENTATION: (k,1) * (k, n^2) via U_keep.T
    L_flat = (np.sqrt(w_keep)[:, None] * U_keep.T)  # (k, norb^2)
    chol_flat = L_flat.T                              # (norb^2, k)

    h1e_spin = np.stack([h1e, h1e], axis=0)         # RHF
    ham = HamGeneric(h1e_spin, chol_flat, ecore)

    if verbose:
        print(f"[ham_from_fcidump] norb={norb}, npair={npair}, "
              f"nchol={chol_flat.shape[1]}, ecore={ecore:.12f}")
    self.ipie_ham = ham
    return self.ipie_ham


  


  def get_mol_hamiltonian_from_fcidump(
    self,
    return_fermion_string: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, int, int, Optional[str]]:
    """
    Load 1e/2e integrals (+ core energy) from an FCIDUMP and build the
    spin-orbital, blocked-ordering Hamiltonian expected by
    `generate_molecular_spin_ham_restricted_blocked`.
    """
    fcidump_path = str(self.fci_dump_path)

    # Robust parse with canonicalization fallback
    res = read(fcidump_path)

    h1e = np.asarray(res['H1'])            # (norb, norb), spatial
    h2e_packed = np.asarray(res['H2'])     # packed ERIs (ij|kl)
    ecore = float(res.get('ECORE', 0.0))
    norb = int(res['NORB'])
    nelec = int(res['NELEC'])

    # Full chemist (pq|rs)
    h2e_chem = ao2mo.restore(1, h2e_packed, norb)  # -> (norb, norb, norb, norb)

    # If your downstream builder wants (p, r, s, q), keep this transpose; otherwise adjust.
    h2e_pqrs = np.asarray(h2e_chem.transpose(0, 2, 3, 1), order='C')

    obi, tbi, core_energy, ferm_ham = self.generate_molecular_spin_ham_restricted_blocked(
        h1e, h2e_pqrs, ecore
    )

    if not return_fermion_string:
        ferm_ham = None

    return (obi, tbi, core_energy, nelec, norb, ferm_ham)

  
  # From Nvidia Qchem
  def generate_molecular_spin_ham_restricted_blocked(self, h1e, h2e, ecore):
      """
      Generate the molecular spin Hamiltonian with **blocked spin-orbital ordering**:
          [α0, α1, ..., α_{N-1}, β0, β1, ..., β_{N-1}]
      """

      n_spatial = h1e.shape[0]
      nqubits = 2 * n_spatial

      one_body_coeff = np.zeros((nqubits, nqubits))
      two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))
      ferm_ham = []

      # α(i) = i, β(i) = n_spatial + i
      def a(i): return i
      def b(i): return n_spatial + i

      for p in range(n_spatial):
          for q in range(n_spatial):

              # Same-spin one-body terms
              one_body_coeff[a(p), a(q)] = h1e[p, q]
              ferm_ham.append(f"{h1e[p, q]} a_{p}^dagger a_{q}")
              one_body_coeff[b(p), b(q)] = h1e[p, q]
              ferm_ham.append(f"{h1e[p, q]} b_{p}^dagger b_{q}")

              for r in range(n_spatial):
                  for s in range(n_spatial):
                      val = 0.5 * h2e[p, q, r, s]

                      # Same-spin αααα and ββββ
                      two_body_coeff[a(p), a(q), a(r), a(s)] = val
                      ferm_ham.append(f"{val} a_{p}^dagger a_{q}^dagger a_{r} a_{s}")

                      two_body_coeff[b(p), b(q), b(r), b(s)] = val
                      ferm_ham.append(f"{val} b_{p}^dagger b_{q}^dagger b_{r} b_{s}")

                      # Mixed-spin αββα and βααβ
                      two_body_coeff[a(p), b(q), b(r), a(s)] = val
                      ferm_ham.append(f"{val} a_{p}^dagger a_{q}^dagger b_{r} b_{s}")

                      two_body_coeff[b(p), a(q), a(r), b(s)] = val
                      ferm_ham.append(f"{val} b_{p}^dagger b_{q}^dagger a_{r} a_{s}")

      full_hamiltonian = " + ".join(ferm_ham)
      return one_body_coeff, two_body_coeff, ecore, full_hamiltonian






  
  def get_FCI_energy(self):
     return self.get_FCI()[0]
  def get_FCI(self):
      if getattr(self, "FCI", None) is not None:
          return self.FCI

      mf  = self.mf
      mol = self.mol
      chk = Path(self.chk_file_path)

      # Current run's metadata
      mo_coeff = np.asarray(mf.mo_coeff)
      norb     = mo_coeff.shape[1]
      nelec    = mol.nelectron if isinstance(mol.nelectron, tuple) else (mol.nelectron//2, mol.nelectron - mol.nelectron//2)
      def _sha1(a: np.ndarray) -> str:
        a = np.ascontiguousarray(a)
        return hashlib.sha1(a.view(np.uint8)).hexdigest()
      meta_now = {
          "norb": int(norb),
          "nelec": tuple(nelec),
          "mo_sha1": _sha1(mo_coeff),
      }

      # Try to load from chk if compatible
      if chk.exists():
          try:
              meta_saved = lib.chkfile.load(str(chk), "fci/meta")
              if (int(meta_saved.get("norb", -1)) == meta_now["norb"] and
                  tuple(meta_saved.get("nelec", ())) == meta_now["nelec"] and
                  meta_saved.get("mo_sha1", "") == meta_now["mo_sha1"]):
                  efci   = float(lib.chkfile.load(str(chk), "fci/e_tot"))
                  fcivec = np.asarray(lib.chkfile.load(str(chk), "fci/ci"))
                  self.FCI = (efci, fcivec)
                  return self.FCI
          except Exception:
              pass  # group/keys not present → fall through to compute

      # Compute FCI in the current MO basis
      hcore_ao = mf.get_hcore()
      h1e = mo_coeff.T @ hcore_ao @ mo_coeff                     # (ij) in MO basis
      eri = ao2mo.kernel(mol, mo_coeff)                          # (pq|rs) in MO basis (chemist's)
      cisolver = fci.FCI(mol, mo_coeff)
      efci, fcivec = cisolver.kernel(h1e, eri, norb, nelec)

      # Save to chk for reuse
      chk.parent.mkdir(parents=True, exist_ok=True)
      lib.chkfile.dump(str(chk), "fci/e_tot", float(efci))
      lib.chkfile.dump(str(chk), "fci/ci",   np.asarray(fcivec))
      lib.chkfile.dump(str(chk), "fci/meta", meta_now)

      self.FCI = (efci, fcivec)
      return self.FCI



  def generate_molecular_spin_ham_restricted(self,h1e, h2e, ecore):

      # This function generates the molecular spin Hamiltonian
      # H = E_core+sum_{`pq`}  h_{`pq`} a_p^dagger a_q +
      #                          0.5 * h_{`pqrs`} a_p^dagger a_q^dagger a_r a_s
      # h1e: one body integrals h_{`pq`}
      # h2e: two body integrals h_{`pqrs`}
      # `ecore`: constant (nuclear repulsion or core energy in the active space Hamiltonian)

      # Total number of qubits equals the number of spin molecular orbitals
      nqubits = 2 * h1e.shape[0]

      # Initialization
      one_body_coeff = np.zeros((nqubits, nqubits))
      two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

      ferm_ham = []

      for p in range(nqubits // 2):
          for q in range(nqubits // 2):

              # p & q have the same spin <a|a>= <b|b>=1
              # <a|b>=<b|a>=0 (orthogonal)
              one_body_coeff[2 * p, 2 * q] = h1e[p, q]
              temp = str(h1e[p, q]) + ' a_' + str(p) + '^dagger ' + 'a_' + str(q)
              ferm_ham.append(temp)
              one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[p, q]
              temp = str(h1e[p, q]) + ' b_' + str(p) + '^dagger ' + 'b_' + str(q)
              ferm_ham.append(temp)

              for r in range(nqubits // 2):
                  for s in range(nqubits // 2):

                      # Same spin (`aaaa`, `bbbbb`) <a|a><a|a>, <b|b><b|b>
                      two_body_coeff[2 * p, 2 * q, 2 * r,
                                    2 * s] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' a_' + str(
                          p) + '^dagger ' + 'a_' + str(
                              q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                      ferm_ham.append(temp)
                      two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                    2 * s + 1] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' b_' + str(
                          p) + '^dagger ' + 'b_' + str(
                              q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                      ferm_ham.append(temp)

                      # Mixed spin(`abab`, `baba`) <a|a><b|b>, <b|b><a|a>
                      #<a|b>= 0 (orthogonal)
                      two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                    2 * s] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' a_' + str(
                          p) + '^dagger ' + 'a_' + str(
                              q) + '^dagger ' + 'b_' + str(r) + ' b_' + str(s)
                      ferm_ham.append(temp)
                      two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                    2 * s + 1] = 0.5 * h2e[p, q, r, s]
                      temp = str(0.5 * h2e[p, q, r, s]) + ' b_' + str(
                          p) + '^dagger ' + 'b_' + str(
                              q) + '^dagger ' + 'a_' + str(r) + ' a_' + str(s)
                      ferm_ham.append(temp)

      full_hamiltonian = " + ".join(ferm_ham)

      return one_body_coeff, two_body_coeff, ecore, full_hamiltonian


def build_h_chain_atom_string(total_atoms: int, spacing: float = 0.74, axis: str = "z") -> str:
    """
    Create a linear hydrogen chain "H x y z; ..." with given spacing (Å)
    along the chosen axis ('x', 'y', 'z'). Origin at index 0.
    """
    if total_atoms < 1:
        raise ValueError("total_atoms must be >= 1.")
    # Unit axis vector
    ax = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis.lower()]
    atoms = []
    for i in range(total_atoms):
        x = i * spacing * ax[0]
        y = i * spacing * ax[1]
        z = i * spacing * ax[2]
        atoms.append(f"H {x:.6f} {y:.6f} {z:.6f}")
    return "; ".join(atoms)



import numpy as np
import itertools

from cudaq import spin


############################################################
def generate_molecular_spin_ham_restricted(h1e, h2e, ecore):

    # This function generates the molecular spin Hamiltonian
    # H= E_core+sum_{`pq`}  h_{`pq`} a_p^dagger a_q +
    #                          0.5 * h_{`pqrs`} a_p^dagger a_q^dagger a_r a_s
    # h1e: one body integrals h_{`pq`}
    # h2e: two body integrals h_{`pqrs`}
    # `ecore`: constant (nuclear repulsion or core energy in the active space Hamiltonian)

    # Total number of qubits equals the number of spin molecular orbitals
    nqubits = 2 * h1e.shape[0]

    # Initialization
    one_body_coeff = np.zeros((nqubits, nqubits))
    two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

    for p in range(nqubits // 2):
        for q in range(nqubits // 2):

            # p & q have the same spin <a|a>= <b|b>=1
            # <a|b>=<b|a>=0 (orthogonal)
            one_body_coeff[2 * p, 2 * q] = h1e[p, q]
            one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[p, q]

            for r in range(nqubits // 2):
                for s in range(nqubits // 2):

                    # Same spin (`aaaa`, `bbbbb`) <a|a><a|a>, <b|b><b|b>
                    two_body_coeff[2 * p, 2 * q, 2 * r,
                                   2 * s] = 0.5 * h2e[p, q, r, s]
                    two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                   2 * s + 1] = 0.5 * h2e[p, q, r, s]

                    # Mixed spin(`abab`, `baba`) <a|a><b|b>, <b|b><a|a>
                    #<a|b>= 0 (orthogonal)
                    two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                   2 * s] = 0.5 * h2e[p, q, r, s]
                    two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                   2 * s + 1] = 0.5 * h2e[p, q, r, s]

    return one_body_coeff, two_body_coeff, ecore


#######################################################################


def jordan_wigner_one_body(p, q, coef):

    # Diagonal term: 0.5 h_{pp} (I_p - Z_p)
    if p == q:
        spin_hamiltonian = 0.5 * coef * spin.i(p)
        spin_hamiltonian -= 0.5 * coef * spin.z(p)

    # h_`pq`(a_p^dagger a_q + a_q^dagger a_p) = R(h_`pq`) (a_p^dagger a_q + a_q^dagger a_p) +
    #                                          `imag` (h_`pq`) (a_p^dagger a_q - a_q^dagger a_p)
    # Off-diagonal real part: 0.5 * real(h_{`pq`}) [ X_p (Z_{p+1}^{q-1}) X_q + Y_p (Z_{p+1}^{q-1}) Y_q ]
    # Off-diagonal imaginary part: 0.5* `im`(h_`pq`) [y_p (Z_{p+1}^{q-1}) x_q - x_p (Z_{p+1}^{q-1}) y_q]

    else:
        if p > q:
            p, q = q, p
            coef = np.conj(coef)

        # Compute the parity string (Z_{p+1}^{q-1})
        z_indices = [i for i in range(p + 1, q)]
        parity_string = 1.0
        for i in z_indices:
            parity_string *= spin.z(i)

        spin_hamiltonian = 0.5 * coef.real * spin.x(p) * parity_string * spin.x(
            q)
        spin_hamiltonian += 0.5 * coef.real * spin.y(
            p) * parity_string * spin.y(q)
        spin_hamiltonian += 0.5 * coef.imag * spin.y(
            p) * parity_string * spin.x(q)
        spin_hamiltonian -= 0.5 * coef.imag * spin.x(
            p) * parity_string * spin.y(q)

    return spin_hamiltonian


#############


def jordan_wigner_two_body(p, q, r, s, coef):

    # Diagonal term: p=r, q=s or p=s,q=r --> (+ h_`pqpq` = + h_`qpqp` = - h_`qppq` = - h_`pqqp`)
    #
    # exchange operator:  h_`pqpq` (a_p^dagger a_q^dagger a_p a_q) + h_`qpqp` (a_q^dagger a_p^dagger a_q a_p)
    # p<q: -1/4 (I_p I_q - I_p Z_q - Z_p I_q+Z_p Z_q)
    #
    # coulomb operator: h_`qppq` (a_q^dagger a_p^dagger a_p a_q) + h_`pqqp` (a_p^dagger a_q^dagger a_q a_p)
    # p<q: 1/4 (I_p I_q - I_p Z_q - Z_p I_q + Z_p Z_q)

    if len(set([p, q, r, s])) == 2:

        if p == r:
            spin_hamiltonian = -0.25 * coef * spin.i(p) * spin.i(q)
            spin_hamiltonian += 0.25 * coef * spin.i(p) * spin.z(q)
            spin_hamiltonian += 0.25 * coef * spin.z(p) * spin.i(q)
            spin_hamiltonian -= 0.25 * coef * spin.z(p) * spin.z(q)

        elif q == r:
            spin_hamiltonian = 0.25 * coef * spin.i(p) * spin.i(q)
            spin_hamiltonian -= 0.25 * coef * spin.i(p) * spin.z(q)
            spin_hamiltonian -= 0.25 * coef * spin.z(p) * spin.i(q)
            spin_hamiltonian += 0.25 * coef * spin.z(p) * spin.z(q)

    # Off-diagonal term with three different sets of non-equal indices
    # Number with excitation operator
    # + h_`pqqs` = + h_`qpsq` = - h_`qpqs` = - h_`pqsq` and their hermitian conjugate
    # Real (h_`pqqs`) (a_p^dagger a_q^dagger a_q a_s + a_s^dagger a_q^dagger a_q a_p) +
    # `imag` (h_`pqqs`) (a_p^dagger a_q^dagger a_q a_s - a_s^dagger a_q^dagger a_q a_p)
    # p <q <s: (1/4)(Z_{p+1}^{s-1}) [ I_q {real (h_`pqqs`/4) (x_p x_s + y_p y_s) + {`imag` (h_`pqqs`/4) (y_p x_s - x_p y_s)}
    #                           - Z_q {real (h_`pqqs`/4) (x_p x_s + y_p y_s) + `imag`(h_`pqqs`) (y_p x_s -x_p y_s)}]

    if len(set([p, q, r, s])) == 3:

        if q == r:
            if p > r:
                a, b = s, p
                coef = np.conj(coef)
            else:
                a, b = p, s
            c = q

        elif q == s:
            if p > r:
                a, b = r, p
                coef = -1.0 * np.conj(coef)
            else:
                a, b = p, r
                coef *= -1.0
            c = q

        elif p == r:
            if q > s:
                a, b = s, q
                coef = -1.0 * np.conj(coef)
            else:
                a, b = q, s
                coef = -1.0 * coef
            c = p

        elif p == s:
            if q > r:
                a, b = r, q
                coef = np.conj(coef)
            else:
                a, b = q, r
            c = p

        parity_string = 1.0
        z_qubit = [i for i in range(a + 1, b)]
        for i in z_qubit:
            parity_string *= spin.z(i)

        spin_hamiltonian = 0.25 * coef.real * spin.x(
            a) * parity_string * spin.x(b) * spin.i(c)
        spin_hamiltonian += 0.25 * coef.real * spin.y(
            a) * parity_string * spin.y(b) * spin.i(c)
        spin_hamiltonian += 0.25 * coef.imag * spin.y(
            a) * parity_string * spin.x(b) * spin.i(c)
        spin_hamiltonian -= 0.25 * coef.imag * spin.x(
            a) * parity_string * spin.y(b) * spin.i(c)

        spin_hamiltonian -= 0.25 * coef.real * spin.x(
            a) * parity_string * spin.x(b) * spin.z(c)
        spin_hamiltonian -= 0.25 * coef.real * spin.y(
            a) * parity_string * spin.y(b) * spin.z(c)
        spin_hamiltonian -= 0.25 * coef.imag * spin.y(
            a) * parity_string * spin.x(b) * spin.z(c)
        spin_hamiltonian += 0.25 * coef.imag * spin.x(
            a) * parity_string * spin.y(b) * spin.z(c)

    # Off-diagonal term with four different sets of non-equal indices
    # h_`pqrs` = h_`qpsr` = - h_`qprs` = - h_`pqsr`
    # real {h_`pqrs`} (a_p^dagger a_q^dagger a_r a_s + a_s^dagger a_r^dagger a_q a_p) +
    # `imag` (h_`pqrs`) (a_p^dagger a_q^dagger a_r a_s - a_s^dagger a_r^dagger a_q a_p)
    # p<q<r<s real part: -1/8 (Z_{p+1}^{q-1}) (Z_{r+1}^{s-1}) (x_p x_q x_r x_s - x_p x_q y_r y_s + x_p y_q x_r y_s + x_p y_q y_r x_s
    #                         + y_p x_q x_r y_s + y_p x_q y_r x_s - y_p y_q x_r x_s + y_p y_q y_r y_s)
    # p<q<r<s `imag` part: -1/8 (x_p x_q x_r y_s + x_p x_q y_r x_s - x_p y_q x_r x_s + x_p y_q y_r y_s
    #                         - y_p x_q x_r x_s + y_p x_q y_r y_s - y_p y_q x_r y_s - y_p y_q y_r x_s)
    # also we need to compute p<r<q<s and p<r<s<q

    elif len(set([p, q, r, s])) == 4:

        if (p > q) ^ (r > s):
            coef *= -1.0

        if p < q < r < s:
            a, b, c, d = p, q, r, s

            parity_string_a = 1.0
            z_qubit_a = [i for i in range(a + 1, b)]
            for i in z_qubit_a:
                parity_string_a *= spin.z(i)

            parity_string_b = 1.0
            z_qubit_b = [i for i in range(c + 1, d)]
            for i in z_qubit_b:
                parity_string_b *= spin.z(i)

            spin_hamiltonian = -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian -= -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian -= -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)

            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)

        elif p < r < q < s:
            a, b, c, d = p, r, q, s

            parity_string_a = 1.0
            z_qubit_a = [i for i in range(a + 1, b)]
            for i in z_qubit_a:
                parity_string_a *= spin.z(i)

            parity_string_b = 1.0
            z_qubit_b = [i for i in range(c + 1, d)]
            for i in z_qubit_b:
                parity_string_b *= spin.z(i)

            spin_hamiltonian = -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)

            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)

        elif p < r < s < q:
            a, b, c, d = p, r, s, q

            parity_string_a = 1.0
            z_qubit_a = [i for i in range(a + 1, b)]
            for i in z_qubit_a:
                parity_string_a *= spin.z(i)

            parity_string_b = 1.0
            z_qubit_b = [i for i in range(c + 1, d)]
            for i in z_qubit_b:
                parity_string_b *= spin.z(i)

            spin_hamiltonian = -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= -0.125 * coef.real * spin.x(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian -= -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += -0.125 * coef.real * spin.y(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)

            spin_hamiltonian -= 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.x(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.x(b) * spin.x(
                    c) * parity_string_b * spin.x(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.x(b) * spin.y(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian -= 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.y(b) * spin.x(
                    c) * parity_string_b * spin.y(d)
            spin_hamiltonian += 0.125 * coef.imag * spin.y(
                a) * parity_string_a * spin.y(b) * spin.y(
                    c) * parity_string_b * spin.x(d)

    return spin_hamiltonian


##########################################################
def jordan_wigner_fermion(h_pq, h_pqrs, ecore, tolerance=1e-12):

    # Compute the qubit `hamiltonian` using `jordan` `wigner`
    # one-body and two-body integrals could be real or complex.
    #
    # For one-body integrals: there are two terms (diagonal term (number operator) h_pp and
    #                                    off-diagonal term (excitation operator) h_`pq`)
    # H_1= sum_`pq` h_{`pq`} a_p^dagger a_q + h.c.
    #
    # For two-body integrals: There are three different terms
    #                 (diagonal term (coulomb and exchange operator):h_`pqqp`, h_`pqpq`,
    #                   off diagonal terms: (number with excitation operator)h_`pqqr`, h_`pqrp`,
    #                   and (double excitation operator) h_`pqrs`
    # H_2 = sum_{`pqrs`} h_`pqrs` a_p^dagger a_q^dagger a_r a_s + h.c.
    #
    # Jordan Wigner transformation
    # a_j^dagger = (Z_{k=1} ^{j-1}) [0.5(X_j - i Y_j)]
    # a_j = (Z_{k=1} ^{j-1}) [0.5(X_j + i Y_j)]
    #
    # Some combination of indices are not allowed because of the `pauli` principles and
    # the anti-commutation relations of the creation and annihilation operators.

    spin_hamiltonian = ecore

    nqubit = h_pq.shape[0]

    for p in range(nqubit):

        # Diagonal one-body term (number operator): sum_p h_{pp} a_p^dagger a_p
        coef = h_pq[p, p]
        if np.abs(coef) > tolerance:
            spin_hamiltonian += jordan_wigner_one_body(p, p, coef)

    for p, q in itertools.combinations(range(nqubit), 2):

        # Off-diagonal one-body term (excitation operator): sum_p<q (h_{`pq`} a_p^dagger a_q + h_{`qp`} a_q^dagger a_p)
        coef = 0.5 * (h_pq[p, q] + np.conj(h_pq[q, p]))
        if np.abs(coef) > tolerance:
            spin_hamiltonian += jordan_wigner_one_body(p, q, coef)

        # Diagonal term two-body (coulomb and exchange operators)
        # Diagonal term: p=r, q=s or p=s,q=r --> (+ h_`pqpq` = + h_`qpqp` = - h_`qppq` = - h_`pqqp`)

        # exchange operator
        coef = h_pqrs[p, q, p, q] + h_pqrs[q, p, q, p]
        if np.abs(coef) > tolerance:
            spin_hamiltonian += jordan_wigner_two_body(p, q, p, q, coef)

        # coulomb operator
        coef = h_pqrs[p, q, q, p] + h_pqrs[q, p, p, q]
        if np.abs(coef) > tolerance:
            spin_hamiltonian += jordan_wigner_two_body(p, q, q, p, coef)

    for (p, q), (r, s) in itertools.combinations(
            itertools.combinations(range(nqubit), 2), 2):

        # h_`pqrs` = - h_`qprs` = -h_`pqsr` = h_`qpsr`
        # Four point symmetry if integrals are complex: `pqrs` = `srqp` = `qpsr` = `rspq`
        # Eight point symmetry if integrals are real: `pqrs` = `rqps` = `psrq` = `srqp` = `qpsr` = `rspq` = `spqr` = `qrsp`


        coef=0.5*(h_pqrs[p,q,r,s] + np.conj(h_pqrs[s,r,q,p]) - h_pqrs[q,p,r,s] - np.conj(h_pqrs[s,r,p,q]) \
             - h_pqrs[p,q,s,r] - np.conj(h_pqrs[r,s,q,p]) + h_pqrs[q,p,s,r] + np.conj(h_pqrs[r,s,p,q]))

        # Compute number with excitation operator and double excitation operator
        if np.abs(coef) > tolerance:
            spin_hamiltonian += jordan_wigner_two_body(p, q, r, s, coef)

    # Remove term with zero coefficient.
    spin_hamiltonian = spin_hamiltonian.canonicalize().trim(tolerance)

    return spin_hamiltonian


##############


def jordan_wigner_pe(v_pq, tolerance=1e-12):

    nqubit = v_pq.shape[0]

    spin_pe_ham = 0.0

    for p in range(nqubit):

        # Diagonal one-body term
        coef = v_pq[p, p]
        if np.abs(coef) > tolerance:
            spin_pe_ham += jordan_wigner_one_body(p, p, coef)

    for p, q in itertools.combinations(range(nqubit), 2):

        coef = 0.5 * (v_pq[p, q] + np.conj(v_pq[q, p]))
        if np.abs(coef) > tolerance:
            spin_pe_ham += jordan_wigner_one_body(p, q, coef)

    # Remove term with zero coefficient.
    op_pe = spin_pe_ham.canonicalize().trim(tolerance)

    return op_pe
