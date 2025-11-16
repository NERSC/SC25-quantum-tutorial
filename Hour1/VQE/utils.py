import cudaq, cudaq_solvers as solvers
from cudaq import spin
from pyscf import gto, scf, ao2mo, mcscf
import numpy as np
import itertools



def generate_molecular_spin_ham_restricted_blocked(h1e, h2e, ecore):
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
    h2 = np.asarray(eri_mo.transpose(0, 2, 3, 1), order='C')
    
    obi, tbi, core_energy, ferm_ham = generate_molecular_spin_ham_restricted_blocked(h1, h2, ecore=0.0)
    
    Spin_operator = jordan_wigner_fermion(obi,tbi,ecore=0.0,tolerance=1e-15)

    class MolObj: pass
    m = MolObj()
    m.hamiltonian = Spin_operator
    m.n_orbitals = ncas
    m.n_electrons = n_active_electrons
    return m
