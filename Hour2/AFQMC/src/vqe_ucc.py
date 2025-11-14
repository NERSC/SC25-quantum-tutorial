import cudaq

@cudaq.kernel
def ucc_circuit(qubit_num: int, 
                theta: list[float],
                pauli_words: list[cudaq.pauli_word],
                coeffs: list[float],
                block_ids: list[int],
                n_spat: int,
                n_alpha: int,
                n_beta: int):

  q = cudaq.qvector(qubit_num)  # Expect qubit_num == 2 * n_spat

  # --- Hartree–Fock prep (RHF), consistent with q = N-1-j mapping ---
  # alpha occupied spin-orbitals: j = 0..n_alpha-1  -> qubits q = N-1 - j
  for i in range(n_alpha):
      x(q[i])
  # beta occupied spin-orbitals:  j = n_spat..n_spat+n_beta-1 -> q = N-1 - j = n_spat - 1 - (j - n_spat)
  for j in range(n_beta):
      x(q[n_spat +j])

  # --- UCC exponentials ---
  for i in range(len(pauli_words)):
    pidx  = block_ids[i]                 # which parameter controls this term
    exp_pauli(-theta[pidx] * coeffs[i], q, pauli_words[i])    




import numpy as np
from scipy.optimize import minimize
import numpy as np
import time
from mpi4py import MPI  
from scipy.optimize import OptimizeResult
THETA = object()

def _sanitize_theta(theta) -> list[float]:
    return np.asarray(theta, dtype=float).ravel().tolist()

def _build_args_from_spec(theta, spec):
    args = []
    for item in spec:
        if item is THETA:
            args.append(_sanitize_theta(theta))
        else:
            args.append(item)
    return tuple(args)




class MQPUGradients:
    def __init__(
        self,
        *,
        kernel,
        hamiltonian,
        kernel_arg_spec,
        init_theta,
        econst,
        max_walltime = None,
        epsilon=1e-3,
        optimizer_method="L-BFGS-B",
        opt_tol=1e-10,
        max_iters = None,
        mpi_info=None,              # <- optional: {'comm': ..., 'rank': int, 'size': int}
        qpus_per_rank=4,
        verbose = True       # <- max concurrent QPUs this rank will use
    ):
        self.kernel = kernel
        self.hamiltonian = hamiltonian
        self.kernel_arg_spec = tuple(kernel_arg_spec)
        self.init_theta = np.asarray(init_theta, dtype=float).ravel()
        self.econst = float(econst)
        self.epsilon = float(epsilon)
        self.optimizer_method = optimizer_method
        self.opt_tol = float(opt_tol)
        self.max_iters = max_iters
        self.verbose = verbose
        # Optional MPI context
        self.mpi = mpi_info
        self.max_walltime = max_walltime
        if self.mpi is None:
            raise RuntimeError("Need mpi4py")

        # Local QPU discovery (per-process / per-node)
        self.num_qpus = cudaq.get_target().num_qpus()
        self.qpus_avail = max(1, min(int(qpus_per_rank), int(self.num_qpus)))

        if self.mpi is None:
            # print("IN GRADIENT CALCULTAION - NUM-QPU: ", self.num_qpus)
            self.energy_evals = {r:0 for r in range(1)}
            self.rank = 0
            self.szie = 1
        else:
            self.energy_evals = {r:0 for r in range(self.mpi.size)}
            self.rank = self.mpi.rank
            self.size = self.mpi.size
            # print("IN GRADIENT Calculation")
            # if self.mpi.rank == 0:
                
            #     print(f"[MPI size={self.mpi.size}] Rank 0 sees {self.num_qpus} QPUs; using up to {self.qpus_avail} per rank.")
            # else:
            #     print(f"[rank {self.mpi.rank}] local QPUs: {self.num_qpus}; using up to {self.qpus_avail}")

        # ---- timing & counters ----
        self.exp_vals = []
        self.time_cost_total = 0.0
        self.time_grad_total = 0.0
        self.num_cost_calls = 0
        self.num_grad_calls = 0
        self.total_energy_evals = 0
        self.iter = 1
        # ---------------------------
    def gather_eval_counts(self):
        if self.mpi is None:
            return {"total": self.total_energy_evals, "per_rank": {0: self.total_energy_evals}}

        comm, rank, size = self.mpi.comm, self.mpi.rank, self.mpi.size
        # send just the scalar count from each rank
        local_count = self.energy_evals[rank]
        counts = comm.gather(local_count, root=0)
        total_local = self.total_energy_evals
        totals = comm.gather(total_local, root=0)

        if rank == 0:
            per_rank = {r: counts[r] for r in range(size)}
            return {"total": sum(totals), "per_rank": per_rank}
        else:
            return None
    # ------------ helpers ------------
    def _my_indices(self, n):
        """Shard indices across ranks; in serial, take all."""
        if self.mpi is None:
            return range(n)
        r, p = self.mpi.rank, self.mpi.size
        return range(r, n, p)

    # ------------ core evals ------------
    def _observe_energy(self, theta) -> float:
        args = _build_args_from_spec(theta, self.kernel_arg_spec)
        self.total_energy_evals += 1
        self.energy_evals[self.rank] += 1 
        t0 = time.time()
        e = cudaq.observe(self.kernel, self.hamiltonian, *args).expectation()
        # print("optim energy_eval time: ", time.time()-t0)
        return e

    def _batched_gradient(self, x: np.ndarray, indices=None) -> np.ndarray:
        """
        Central-difference gradient. If 'indices' is provided, only compute those
        components and return a full-size vector with zeros elsewhere.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        eps = self.epsilon
        if indices is None:
            indices = range(n)

        # Build perturbed points only for selected indices
        futures_p, futures_m = [], []
        idx_list = []
        qid = 0
        t0 = time.time()
        # print(f"Rank/Node: {self.mpi.rank}, parameter indicies: {indices} ")
        # print("optim len indices", len(indices))

        for i in indices:
            ei = np.zeros(n)
            ei[i] = 1.0
            xp = x + eps * ei
            xm = x - eps * ei

            args_plus = _build_args_from_spec(xp, self.kernel_arg_spec)
            f_plus = cudaq.observe_async(
                self.kernel, self.hamiltonian, *args_plus,
                qpu_id=(qid % self.qpus_avail)
            )
            self.energy_evals[self.rank] += 1
            qid += 1

            args_minus = _build_args_from_spec(xm, self.kernel_arg_spec)
            f_minus = cudaq.observe_async(
                self.kernel, self.hamiltonian, *args_minus,
                qpu_id=(qid % self.qpus_avail)
            )
            self.energy_evals[self.rank] += 1
            qid += 1

            futures_p.append(f_plus)
            futures_m.append(f_minus)
            idx_list.append(i)

        # Gather and assemble full gradient vector (zeros outside my indices)
        g_full = np.zeros(n, dtype=float)
        for i, f_p, f_m in zip(idx_list, futures_p, futures_m):
            ep = f_p.get().expectation()
            em = f_m.get().expectation()
            self.total_energy_evals += 2
            g_full[i] = (ep - em) / (2.0 * eps)
        # print("optim B: ", time.time() - t0)

        return g_full

    # ------------ public API used by SciPy ------------
    def cost(self, x: np.ndarray) -> float:
        if self.mpi is None:
            return float(self._observe_energy(x))
        # MPI: only root needs the value; workers never call cost().
        if self.mpi.rank == 0:
            return float(self._observe_energy(x))
        else:
            return 0.0

    def jac(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        if self.mpi is None:
            g = self._batched_gradient(x)
            self.time_grad_total += (time.perf_counter() - t0)
            self.num_grad_calls += 1
            return g

        # MPI mode: root orchestrates; workers are in a loop inside optimize()
        comm = self.mpi.comm; rank = self.mpi.rank

        # Notify workers we are about to do a JAC step, then broadcast x
        comm.bcast("JAC", root=0)
        x = np.asarray(x, float).ravel()
        comm.bcast(x, root=0)

        # Root also computes its shard; workers do theirs in their loop, not here.
        # To keep a single code path, root computes its local shard and participates
        # in a Reduce(SUM) to assemble the full gradient.
        n = x.size
        g_local = self._batched_gradient(x, indices=self._my_indices(n))

        g_full = np.zeros(n, dtype=float) if rank == 0 else None
        comm.Reduce([g_local, MPI.DOUBLE], [g_full, MPI.DOUBLE] if rank == 0 else None,
                    op=MPI.SUM, root=0)

        self.time_grad_total += (time.perf_counter() - t0)
        if rank == 0:
            self.num_grad_calls += 1
            return g_full
        else:
            # Return value is ignored by SciPy on non-root; still return something
            return g_local
    def _make_callback_intermediate(self, *, max_walltime=None):
        self.start = time.perf_counter()

        # store last iterate so we can build a partial OptimizeResult if we stop
        self._last_intermediate = None
        self._stopped_reason = None

        def callback(*, intermediate_result: OptimizeResult):
            # record & (optionally) keep your own trace
            self._last_intermediate = intermediate_result
            xk = intermediate_result.x
            self.callback(xk)
            if self.verbose:  # your existing bookkeeping (appends exp_vals, etc.)
                print(f"Iteration  {self.iter}, Energy = {self.exp_vals[self.iter]},  Time  = {time.perf_counter()-self.start}")
            self.iter += 1
            # budgets
            over_wall = (max_walltime is not None and
                         time.perf_counter() - self.start >= max_walltime)
            if over_wall:
                self._stopped_reason = "walltime"
                raise StopIteration  # <- mandated by docs to terminate

        return callback
    def callback(self, xk, *_, **__):
        # Only rank 0 keeps a trace in MPI mode
        if (self.mpi is None) or (self.mpi.rank == 0):
            E = self.cost(xk) + self.econst
            self.exp_vals.append(E)

    def timing_summary(self) -> dict:
        return {
            "time_cost_total": self.time_cost_total,
            "time_grad_total": self.time_grad_total,
            "time_quantum_total": self.time_cost_total + self.time_grad_total,
            "num_cost_calls": self.num_cost_calls,
            "num_grad_calls": self.num_grad_calls,
            "avg_cost_time": (self.time_cost_total / self.num_cost_calls) if self.num_cost_calls else 0.0,
            "avg_grad_time": (self.time_grad_total / self.num_grad_calls) if self.num_grad_calls else 0.0,
            "num_energy_evals": self.total_energy_evals,
        }

    def optimize(self):
        theta0 = self.init_theta.copy()
        if self.max_iters is not None:
            options = {'maxiter':self.max_iters}
        else:
            options = None
        # record initial value (not included in avg timing below)
        if (self.mpi is None) or (self.mpi.rank == 0):
            self.exp_vals.append(self.cost(theta0) + self.econst)
            if self.verbose:
                print(f"Iteration  0, Energy = {self.exp_vals[0]},  Time  = 0")


        eval_times = []  # backward-compat

        def timed_cost(x):
            t0 = time.perf_counter()
            val = self.cost(x)
            dt = time.perf_counter() - t0
            eval_times.append(dt)
            self.time_cost_total += dt
            self.num_cost_calls += 1
            return val


        comm = self.mpi.comm; rank = self.rank; size = self.size

        if rank == 0:
            cb = self._make_callback_intermediate(max_walltime=self.max_walltime)
            # start worker loops by broadcasting a no-op so they enter bcast
            # (not strictly necessary, workers will block waiting for first cmd)
            # Run SciPy on root only
            try:
                result = minimize(
                    fun=timed_cost,   # no broadcasts for cost
                    x0=theta0,
                    method=self.optimizer_method,
                    jac=self.jac,     # jac will handle "JAC" broadcasts to workers
                    callback=cb,
                    tol=self.opt_tol,
                    options=options
                )
            except StopIteration:
    
                    ir = getattr(self, "_last_intermediate", None)
                    if ir is not None:
                        result = OptimizeResult(x=ir.x, fun=ir.fun, success=False,
                                                message=f"Stopped early ({self._stopped_reason}).")
                    else:
                        # fall back to current theta0 if somehow nothing was recorded
                        result = OptimizeResult(x=theta0, fun=self.cost(theta0), success=False,
                                                message=f"Stopped early ({self._stopped_reason}).")

            
            finally:
                # Tell workers to stop
                comm.bcast("STOP", root=0)
            self.exp_vals.append(self.cost(result.x) + self.econst)
            return result, np.array(self.exp_vals, dtype=float)

        else:
            # Worker loop: respond to root's commands
            while True:
                cmd = comm.bcast(None, root=0)  # "JAC" or "STOP"
                if cmd == "STOP":
                    break
                elif cmd == "JAC":
                    x = comm.bcast(None, root=0)  # receive x
                    n = x.size
                    g_local = self._batched_gradient(x, indices=self._my_indices(n))
                    comm.Reduce([g_local, MPI.DOUBLE], None, op=MPI.SUM, root=0)
                else:
                    pass

            # Return a placeholder result to keep API consistent on workers
            class _Res: pass
            result = _Res()
            result.x = theta0
            result.fun = self.cost(theta0)
            result.success = True
            return result, np.array(self.exp_vals, dtype=float)

def convert_state_big_endian(state_little_endian):

        state_big_endian = 0. * state_little_endian

        n_qubits = int(np.log2(state_big_endian.size))
        for j, val in enumerate(state_little_endian):
                little_endian_pos = np.binary_repr(j, n_qubits)
                big_endian_pos = little_endian_pos[::-1]
                int_big_endian_pos = int(big_endian_pos, 2)
                state_big_endian[int_big_endian_pos] = state_little_endian[j]

        return state_big_endian


class BlockStatevector:
    """
    Build an ipie-style multi-determinant trial from a BIG-ENDIAN statevector.

    Assumptions
    ----------
    - statevector : 1D complex numpy array of length 2**n_qubits (BIG-ENDIAN).
      Big-endian here means index j corresponds to the bitstring with the MSB on
      the left; to map to qubit indices (q=0 is LSB), we flip the bits first.
    - Jordan–Wigner (block spin ordering): qubits 0..(norb-1) = alpha,
      qubits norb..(2*norb-1) = beta, where norb = n_spin_orbitals // 2.
    """

    def __init__(self, statevector: np.ndarray, mol_problem, ampl_eps: float | None = None,vectorized=True):
        self.mol_problem = mol_problem
        self.statevector = np.asarray(statevector, dtype=np.complex128)

        self.n_spin_orbitals = self.mol_problem.active_spin_orbitals     # = 2 * norb
        self.n_alpha = self.mol_problem.active_mol_nelec[0]
        self.n_beta  = self.mol_problem.active_mol_nelec[1]

        self.coeffs: list[np.complex128] = []
        self.occ_a:  list[np.ndarray]    = []
        self.occ_b:  list[np.ndarray]    = []
        self.vectorized = vectorized

        self.__compute_wavefunction(ampl_eps)
    def getIPIEWavefunction(self):
        """Return (coeffs, occ_alpha_list, occ_beta_list)."""
        return self.wavefunction

    # -------- internals --------

    @staticmethod
    def _flip_bits_to_little_endian(j: int, n_qubits: int) -> int:
        """Convert big-endian index j into a little-endian integer mask."""
        return int(f"{j:0{n_qubits}b}"[::-1], 2)

    def _decode_occ_block_spin(self, j_le: int, n_qubits: int):
        """Decode occupations from a LITTLE-ENDIAN integer j_le (block spin JW)."""
        norb = n_qubits // 2
        a_occ, b_occ = [], []
        # qubit q is set iff (j_le >> q) & 1 == 1
        for q in range(n_qubits):
            if (j_le >> q) & 1:
                if q < norb:
                    a_occ.append(q)           # alpha orbital index
                else:
                    b_occ.append(q - norb)    # beta orbital index
        return a_occ, b_occ

    def __compute_wavefunction(self, eps: float | None):
        psi = self.statevector
        n_dim = psi.size
        n_qubits = int(round(np.log2(n_dim)))
        assert (1 << n_qubits) == n_dim, "Statevector length must be a power of 2."
        assert n_qubits == self.n_spin_orbitals, \
            f"Qubit count ({n_qubits}) != spin-orbital count ({self.n_spin_orbitals})."

        norb = n_qubits // 2
        n_alpha, n_beta = self.n_alpha, self.n_beta

        # Optional amplitude mask (pre-filter)
        if eps is not None:
            amp_mask = np.abs(psi) >= eps
            if not np.any(amp_mask):
                raise ValueError("No amplitudes above threshold eps.")
            idx_all = np.nonzero(amp_mask)[0]
        else:
            idx_all = np.arange(n_dim, dtype=np.int64)

        # We will build results only for selected indices to avoid 2^n memory.
        sel_coeffs = []
        sel_occ_a  = []
        sel_occ_b  = []

        # Vectorized bit extraction helper (little-endian bit order!)
        # bits_le[i, q] = ((idx[i] >> q) & 1)
        def bits_le_from_indices(idx):
            # idx are BIG-endian indices; produce LITTLE-endian bit matrix
            # bits_le[:, q] = ((idx >> (n_qubits - 1 - q)) & 1)
            shifts = (n_qubits - 1 - np.arange(n_qubits, dtype=idx.dtype))
            return (idx[:, None] >> shifts) & 1

        # Process in chunks to avoid huge (len(idx) x n_qubits) allocations
        CHUNK = 1_000_000 // max(n_qubits, 1)  # rough heuristic; tune as needed
        for start in range(0, idx_all.size, CHUNK):
            sl = slice(start, min(start + CHUNK, idx_all.size))
            idx = idx_all[sl]
            if idx.size == 0: break

            bits = bits_le_from_indices(idx)              # (m, n_qubits), little-endian
            bits_a = bits[:, :norb]                       # alpha block (q=0..norb-1)
            bits_b = bits[:, norb:]                       # beta  block (q=norb..2norb-1)

            # Electron-count filter (vectorized)
            mask_elec = (bits_a.sum(axis=1) == n_alpha) & (bits_b.sum(axis=1) == n_beta)
            if not np.any(mask_elec):
                continue

            idx_kept = idx[mask_elec]
            # Pull amplitudes
            amps_kept = psi[idx_kept]

            # Gather occupations (loop only over kept determinants, typically sparse)
            a_lists = [np.flatnonzero(row).astype(np.int32, copy=False) for row in bits_a[mask_elec]]
            b_lists = [np.flatnonzero(row).astype(np.int32, copy=False) for row in bits_b[mask_elec]]

            sel_coeffs.append(amps_kept)
            sel_occ_a.extend(a_lists)
            sel_occ_b.extend(b_lists)

        if not sel_coeffs:
            raise ValueError(
                "No determinants selected (check electron counts, threshold, or spin-orbital mapping)."
            )

        coeffs = np.concatenate(sel_coeffs).astype(np.complex128, copy=False)

        # Sort by |coeff| descending
        order = np.argsort(-np.abs(coeffs))
        coeffs = coeffs[order]
        occ_a  = [sel_occ_a[i] for i in order]
        occ_b  = [sel_occ_b[i] for i in order]

        # Normalize the selected subspace coefficients
        norm = np.linalg.norm(coeffs)
        if norm == 0.0:
            raise ValueError("Selected amplitudes have zero norm after filtering.")
        coeffs = coeffs / norm

        self.coeffs = coeffs
        self.occ_a  = occ_a
        self.occ_b  = occ_b
        self.wavefunction = (self.coeffs, self.occ_a, self.occ_b)
    


    # -------- utilities --------

    def order_truncate(self, max_det: int | None = None):
        """Optionally truncate to top-|coeff| determinants and renormalize."""
        if max_det is None or max_det >= len(self.coeffs):
            return self.wavefunction
        coeffs = self.coeffs[:max_det]
        occ_a  = self.occ_a[:max_det]
        occ_b  = self.occ_b[:max_det]
        coeffs = coeffs / np.linalg.norm(coeffs)
        return (coeffs, occ_a, occ_b)


