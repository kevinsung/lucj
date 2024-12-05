import logging
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
from molecules_catalog.util import load_molecular_data
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import (
    solve_fermion,
)
from qiskit_addon_sqd.subsampling import postselect_and_subsample

from lucj.params import LUCJParams
from lucj.util import interaction_pairs_spin_balanced

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJSQDInitialParamsTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    shots: int
    samples_per_batch: int
    n_batches: int
    max_davidson: int
    entropy: int | None

    @property
    def dirpath(self) -> Path:
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.2f}"
            )
            / self.lucj_params.dirpath
            / f"shots-{self.shots}"
            / f"samples_per_batch-{self.samples_per_batch}"
            / f"n_batches-{self.n_batches}"
            / f"max_davidson-{self.max_davidson}"
            / f"entropy-{self.entropy}"
        )


def run_lucj_sqd_initial_params_task(
    task: LUCJSQDInitialParamsTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJSQDInitialParamsTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "data.pickle"
    if (not overwrite) and os.path.exists(data_filename):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian
    mol_data = load_molecular_data(
        f"{task.molecule_basename}_d-{task.bond_distance:.2f}",
        molecules_catalog_dir=molecules_catalog_dir,
    )
    norb = mol_data.norb
    nelec = mol_data.nelec

    # Initialize initial state and LUCJ parameters
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )

    # use CCSD to initialize parameters
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        mol_data.ccsd_t2,
        n_reps=task.lucj_params.n_reps,
        t1=mol_data.ccsd_t1 if task.lucj_params.with_final_orbital_rotation else None,
        interaction_pairs=(pairs_aa, pairs_ab),
    )

    # Compute final state
    logging.info(f"{task} Computing state vector...\n")
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    # Run SQD
    logging.info(f"{task} Running SQD...\n")
    rng = np.random.default_rng(task.entropy)
    samples = ffsim.sample_state_vector(
        final_state,
        norb=norb,
        nelec=nelec,
        shots=task.shots,
        seed=rng,
    )
    counts = Counter(samples)
    bitstring_matrix_full, probs_arr_full = counts_to_arrays(counts)
    n_alpha, n_beta = nelec
    batches = postselect_and_subsample(
        bitstring_matrix_full,
        probs_arr_full,
        hamming_right=n_alpha,
        hamming_left=n_beta,
        samples_per_batch=task.samples_per_batch,
        num_batches=task.n_batches,
        rand_seed=rng,
    )
    energies = np.zeros(task.n_batches)
    spin_squareds = np.zeros(task.n_batches)
    sci_states = []
    for i, batch in enumerate(batches):
        energy_sci, sci_state, avg_occs, spin = solve_fermion(
            batch,
            mol_data.one_body_integrals,
            mol_data.two_body_integrals,
            open_shell=False,
            spin_sq=0,
            max_davidson=task.max_davidson,
        )
        energies[i] = energy_sci + mol_data.core_energy
        spin_squareds[i] = spin
        sci_states.append(sci_state)
    index = np.argmin(energies)
    energy = energies[index]
    spin_squared = spin_squareds[index]
    sci_state = sci_states[index]
    error = energy - mol_data.fci_energy

    # Save data
    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
        "sci_vec_shape": sci_state.amplitudes.shape,
        "n_reps": operator.n_reps,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
