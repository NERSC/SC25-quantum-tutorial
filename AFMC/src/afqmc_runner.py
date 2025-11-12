import numpy as np
from numpy import load
import os
from src.MoleleculeBuilder import BuildMoleculeProblem
import sys; sys.path.append('../')
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.utils.from_pyscf import generate_integrals
from ipie.systems.generic  import Generic
from ipie.walkers.uhf_walkers   import UHFWalkersParticleHole
from ipie.utils.mpi             import MPIHandler
from ipie.qmc.afqmc             import AFQMC
from ipie.analysis.extraction  import extract_observable
from ipie.analysis.autocorr import reblock_by_autocorr  
from pyscf.fci import cistring
from types import SimpleNamespace
from mpi4py import MPI
import time
from mpi4py import MPI  #
import pandas as pd
from  pathlib import Path

class TrialWfn:
  def __init__(self,coeffs,occa,occb,mol_problem,max_det = None,verbose=False):
    self.coeffs = coeffs
    self.occa  = occa
    self.occb = occb
    self.full_num_dets = len(self.coeffs)
    self.mol_problem = mol_problem
    self.verbose = verbose
    self.order_truncate(max_det)
    self.num_dets = len(self.t_coeffs)

  def get_PH_trial(self):
    trial = ParticleHole(self.trial_wfn, 
                      self.mol_problem.mol_nelec,
                      self.mol_problem.full_spatial_orbitals, 
                      num_dets_for_props=self.num_dets,
                      use_active_space= self.mol_problem.active_space,
                      verbose = self.verbose)

    return trial
  def build_PH_trial(self):
    trial = self.get_PH_trial()
    ipie_ham = self.mol_problem.build_ipie_ham_from_fcidump()
    system = Generic(self.mol_problem.mol_nelec)
    trial.compute_trial_energy = True
    trial.build()
    trial.half_rotate(ipie_ham)
    self.var_energy, _, _ = trial.calculate_energy(system, ipie_ham)
    return trial
  def order_truncate(self, max_det=None):
      c = self.coeffs
      a = self.occa
      b = self.occb

      # sort by |coeff| descending
      order = np.argsort(-np.abs(c))
      c = c[order]
      a = [a[i] for i in order]
      b = [b[i] for i in order]

      if max_det is not None:
        if max_det < len(c):
            c = c[:max_det]
            a = a[:max_det]
            b = b[:max_det]
        else: 
          print(f"num dets {len(c)} >= max dets {max_det}, returning full trial")
      n = np.linalg.norm(c)
      c = c / n
      self.t_coeffs,self.t_occa,self.t_occb = c,a,b
      self.trial_wfn = (self.t_coeffs,self.t_occa,self.t_occb)
  def get_fci_overlap(self):
    """Return ⟨Ψ_FCI | Ψ_trial⟩ using ParticleHole expansions (complex scalar)."""

    # --- get FCI as a ParticleHole trial in the SAME MO basis/space ---
    fci_TrialWfn = self.mol_problem.get_FCI_trial(max_det = 100)
    fci_PH = fci_TrialWfn.get_PH_trial()

    # --- get your AFQMC trial as ParticleHole over the full space ---
    trial_PH = self.get_PH_trial()

    # --- sizes must match the determinant basis used by both PH objects ---
    norb   = self.mol_problem.full_spatial_orbitals
    nalpha,nbeta = self.mol_problem.mol_nelec
    

    def det_address(alpha_occ, beta_occ, norb, nalpha, nbeta):
        """Alpha-major linear index (Na * Nb grid) matching PySCF ordering."""
        bit_a = sum(1 << i for i in alpha_occ)  # little-endian: orb 0 is LSB
        bit_b = sum(1 << i for i in beta_occ)
        idx_a = cistring.str2addr(norb, nalpha, bit_a)
        idx_b = cistring.str2addr(norb, nbeta,  bit_b)
        nbeta_dim = cistring.num_strings(norb, nbeta)
        return idx_a * nbeta_dim + idx_b

    # --- map FCI determinants to coefficients for O(1) lookup ---
    fci_map = {}
    for cF, a_occF, b_occF in zip(fci_PH.coeffs, fci_PH.occa, fci_PH.occb):
        addrF = det_address(a_occF, b_occF, norb, nalpha, nbeta)
        # If duplicates appear (shouldn't), accumulate
        fci_map[addrF] = fci_map.get(addrF, 0.0 + 0.0j) + cF

    # --- accumulate overlap over the trial determinants ---
    ovlp = 0.0 + 0.0j
    for cT, a_occT, b_occT in zip(trial_PH.coeffs, trial_PH.occa, trial_PH.occb):
        addrT = det_address(a_occT, b_occT, norb, nalpha, nbeta)
        cF = fci_map.get(addrT, 0.0 + 0.0j)
        if cF != 0.0:
            ovlp += np.conjugate(cF) * cT
    norm_fci = np.sqrt(np.sum(np.abs(fci_PH.coeffs)**2))
    norm_trial = np.sqrt(np.sum(np.abs(trial_PH.coeffs)**2))
    normalized_ovlp = ovlp / (norm_fci * norm_trial)
    return abs(normalized_ovlp)






class AFQMCParams(SimpleNamespace):
    def __init__(self, num_total_walkers,
        num_steps_per_block,
        num_blocks,
        timestep,
        stabilize_freq,
        pop_control_freq,
        comm_size,
        verbose
    ):
        self.num_total_walkers = num_total_walkers
        self.num_steps_per_block = num_steps_per_block
        self.num_blocks = num_blocks
        self.timestep = timestep
        self.stabilize_freq = stabilize_freq
        self.pop_control_freq = pop_control_freq
        self.comm_size = comm_size
        self.verbose = verbose
        self.num_walkers = num_total_walkers // comm_size

    def copy_and_update(self, update_dict : dict):
        new_params =  AFQMCParams(
        self.num_total_walkers,
        self.num_steps_per_block,
        self.num_blocks,
        self.timestep,
        self.stabilize_freq,
        self.pop_control_freq,
        self.comm_size,
        self.verbose
       )
        
        for key, value in update_dict.items():
           if not hasattr(new_params, key):
              raise KeyError(f'unknown param field: {key}')
           setattr(new_params, key, value)

        new_params.num_walkers = (
           new_params.num_total_walkers // new_params.comm_size
        )
    
        return new_params



class AFQMC_Event:
    def __init__(self,
                 mol_problem : BuildMoleculeProblem,
                 afqmc_params : AFQMCParams,
                 saving_to_file : bool = False):
        self.mol_problem = mol_problem
        self.afqmc_params = afqmc_params
        self.saving_to_file = saving_to_file

    def run_afqmc(self, trial_wfn:TrialWfn, global_offset=0,comm=MPI.COMM_WORLD,group_id = 0):
        self.trial_wfn = trial_wfn.trial_wfn
        self.coeffs = trial_wfn.t_coeffs
        self.occa = trial_wfn.t_occa
        self.occb = trial_wfn.t_occb
        self.num_dets = len(self.coeffs)
        afqmc_run = AFQMCRunner(params=self.afqmc_params, mol_problem=self.mol_problem, Trial=trial_wfn, global_offset=global_offset,comm=comm, group_id=group_id)
        ETotals,trial_energy,afqmc_runtime = afqmc_run.run()
        # print("Trial Energy",trial_energy)
        self.ETotals = ETotals
        self.trial_energy = trial_energy
        self.afqmc_runtime = afqmc_runtime
        self.afqmc_dict = self.afqmc_params.__dict__
        self.mean_ac, self.err_ac,_,_ = self.reblocking_analysis()
    def reblocking_analysis(self, burnin=0.25):
        energy_list = self.ETotals
        start = int(burnin * len(energy_list))
        y = np.asarray(energy_list[start:], dtype=float)

        mean_ac = err_ac = None
        block_size = None
        block_df = None

        if len(y) > 10:
            df = reblock_by_autocorr(y, name="ETotal", verbose=False)
            mean_ac = df["ETotal_ac"].iat[0]
            err_ac  = df["ETotal_error_ac"].iat[0]

            ac = df["ac"].iat[0]  # integrated autocorr or suggested block length
            block_size = max(1, int(np.ceil(ac)))  # round up to be safe

            n = len(y)
            n_blocks = n // block_size
            if n_blocks >= 1:
                y_trim = y[: n_blocks * block_size]
                blocks = y_trim.reshape(n_blocks, block_size)

                block_means = blocks.mean(axis=1)
                block_stds  = blocks.std(axis=1, ddof=1) if block_size > 1 else np.zeros(n_blocks)

                overall_mean = block_means.mean()
                overall_sem  = block_means.std(ddof=1) / np.sqrt(n_blocks) if n_blocks > 1 else 0.0

                starts = start + np.arange(n_blocks) * block_size
                ends   = starts + block_size - 1
                block_df = pd.DataFrame({
                    "block_id":   np.arange(n_blocks),
                    "start_idx":  starts,
                    "end_idx":    ends,
                    "block_size": block_size,
                    "mean":       block_means,
                    "std":        block_stds,
                })
                block_df.attrs["overall_mean_from_blocks"] = overall_mean
                block_df.attrs["overall_sem_from_blocks"]  = overall_sem

        return mean_ac, err_ac, block_size, block_df










class AFQMCRunner:
  def __init__(self, 
        params: AFQMCParams,
        mol_problem: BuildMoleculeProblem,
        Trial: TrialWfn,
        global_offset=0,
        comm=MPI.COMM_WORLD,
        group_id = 0): 
    self.params = params
    self.mol_problem = mol_problem
    self.Trial = Trial
    self.global_offset = global_offset
    self.comm  = comm
    self.group_id = group_id

    
  def run(self):
    # ham  = self.mol_problem.build_ipie_ham()
    ham = self.mol_problem.build_ipie_ham_from_fcidump()
    PH_trial = self.Trial.build_PH_trial() #half rotate (ham), calculates energy
    initial_walker = np.hstack([PH_trial.psi0a, PH_trial.psi0b]) #both coefficient matriciies for occupied orbitals of alpha and beta electrons from FIRST DETERMINENT in Trial
    np.random.seed(123456789)

    walkers = UHFWalkersParticleHole(
        initial_walker,
        self.mol_problem.mol_nelec[0], #num of alpha and beta electrons
        self.mol_problem.mol_nelec[1],
        PH_trial.psi0a.shape[0],
        self.params.num_walkers,
        MPIHandler(self.comm))#handles any MPI    )
    walkers.build(PH_trial)
   

    afqmc_msd = AFQMC.build(
        self.mol_problem.mol_nelec,
        ham,
        PH_trial, 
        walkers=walkers, # initial walker populationo object
        num_walkers=self.params.num_walkers, 
        num_steps_per_block=self.params.num_steps_per_block, # number of dt steps per block, each step: sample from AF, apply propogatoor, comptute overlaps, and update walker weightss
        num_blocks=self.params.num_blocks, # each block aaverages local energies (or any observable) over all (num_steps_per_block) steps 
        timestep=self.params.timestep, # dt
        stabilize_freq=self.params.stabilize_freq, # after (stapilize_freq) steps, we re-orthonormalize from repeatedly propogating
        seed=12,
        pop_control_freq=self.params.pop_control_freq, # after (pop_control_freq) steps, we kill low-weight walkers and clone high weight walkers, to keep # of walkers roughly the same
        verbose=self.params.verbose,
        mpi_handler = MPIHandler(self.comm))
    estimator_filename = f"estimators_group{self.group_id}.h5"
    t0 = time.time()
    afqmc_msd.run(estimator_filename = estimator_filename)
    runtime = time.time()-t0
    qmc_data = extract_observable(estimator_filename , "energy")
    ETotals = qmc_data["ETotal"].tolist()
    afqmc_msd.finalise(verbose=True)
    trial_energy = PH_trial.energy
    return ETotals, trial_energy ,runtime





def unpack_vqe_data(full_path,mpi_info=None):
    vqe_run_data = load(full_path, allow_pickle=True)
    fname = os.path.basename(full_path)

    # unpack and format data
    atom = str(vqe_run_data['atom'])
    basis = str(vqe_run_data['basis'])
    spin = int(vqe_run_data['spin'])
    mol_identifier = str(vqe_run_data['mol_identifier'])
    # print("mol_identifier: ", mol_identifier)
    # temporary, remove  later
    # mol_identifier  =  grab_middle(fname) 
    active_orbitals = None
    active_mol_nelec = None

    if vqe_run_data['active_orbitals'] is not None:
        active_orbitals = vqe_run_data['active_orbitals']
        active_mol_nelec = vqe_run_data['active_mol_nelec']
    print("active_orbitals",active_orbitals)
    print("active_mol_nelec",active_mol_nelec)
    mol_problem = BuildMoleculeProblem(atom, basis, spin, active_orbitals, active_mol_nelec,mol_identifier=mol_identifier,mpi_info=mpi_info)

    coeffs = vqe_run_data['coeffs']
    occ_a = vqe_run_data['occ_a']
    occ_b = vqe_run_data['occ_b']
    wf = (np.asarray(coeffs, dtype=np.complex128),
        occ_a,
        occ_b)
    
    return mol_problem, wf, vqe_run_data

