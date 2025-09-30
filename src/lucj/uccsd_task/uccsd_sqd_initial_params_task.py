# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
from molecules_catalog.util import load_molecular_data
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)
from functools import partial
import scipy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class UCCSDSQDInitialParamsTask:
    molecule_basename: str
    bond_distance: float | None
    shots: int
    samples_per_batch: int
    n_batches: int
    energy_tol: float
    occupancies_tol: float
    carryover_threshold: float
    max_iterations: int
    symmetrize_spin: bool
    entropy: int | None
    max_dim: int | None

    @property
    def dirpath(self) -> Path:
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.5f}"
            )
            / "uccsd"
            / f"shots-{self.shots}"
            / f"samples_per_batch-{self.samples_per_batch}"
            / f"n_batches-{self.n_batches}"
            / f"energy_tol-{self.energy_tol}"
            / f"occupancies_tol-{self.occupancies_tol}"
            / f"carryover_threshold-{self.carryover_threshold}"
            / f"max_iterations-{self.max_iterations}"
            / f"symmetrize_spin-{self.symmetrize_spin}"
            / f"entropy-{self.entropy}"
            / f"max_dim-{self.max_dim}"
        )

    @property
    def vqepath(self) -> Path:
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.5f}"
            )
            / "uccsd"
        )


def run_uccsd_sqd_initial_params_task(
    task: UCCSDSQDInitialParamsTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> UCCSDSQDInitialParamsTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "sqd_data.pickle"
    vqe_data_filename = data_dir / task.vqepath / "data.pickle"
    if (not overwrite) and os.path.exists(data_filename):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian
    mol_data = load_molecular_data(
        f"{task.molecule_basename}_d-{task.bond_distance:.5f}",
        molecules_catalog_dir=molecules_catalog_dir,
    )
    mol_ham = mol_data.hamiltonian
    norb = mol_data.norb
    nelec = mol_data.nelec

    # Initialize initial state
    hamiltonian = ffsim.linear_operator(mol_ham, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    # use CCSD to initialize parameters
    operator = ffsim.UCCSDOpRestrictedReal(t1=mol_data.ccsd_t1, t2=mol_data.ccsd_t2)

    # Compute final state
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    # record vqe energy
    logging.info(f"{task} Computing VQE data...\n")
    energy = np.vdot(final_state, hamiltonian @ final_state).real
    if mol_data.fci_energy is None:
        error = energy - mol_data.sci_energy
    else:
        error = energy - mol_data.fci_energy
    spin_squared = ffsim.spin_square(
        final_state, norb=mol_data.norb, nelec=mol_data.nelec
    )
    probs = np.abs(final_state) ** 2
    entropy = scipy.stats.entropy(probs)

    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
        "entropy": entropy,
    }

    logging.info(f"{task} Saving VQE data...\n")
    with open(vqe_data_filename, "wb") as f:
        pickle.dump(data, f)

    # Run SQD
    logging.info(f"{task} Running SQD...\n")
    rng = np.random.default_rng(task.entropy)
    samples = ffsim.sample_state_vector(
        final_state,
        norb=norb,
        nelec=nelec,
        shots=task.shots,
        seed=rng,
        bitstring_type=ffsim.BitstringType.INT,
    )

    result_history_energy = []
    result_history_subspace_dim = []
    result_history = []

    def callback(results: list[SCIResult]):
        result_energy = []
        result_subspace_dim = []
        iteration = len(result_history)
        result_history.append(results)
        logging.info(f"Iteration {iteration}")
        for i, result in enumerate(results):
            result_energy.append(result.energy + mol_data.core_energy)
            result_subspace_dim.append(result.sci_state.amplitudes.shape)
            logging.info(f"\tSubsample {i}")
            logging.info(f"\t\tEnergy: {result.energy + mol_data.core_energy}")
            logging.info(
                f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
            )
        result_history_energy.append(result_energy)
        result_history_subspace_dim.append(result_subspace_dim)

    bit_array = BitArray.from_samples(samples, num_bits=2 * norb)
    sci_solver = partial(solve_sci_batch, spin_sq=0.0)
    result = diagonalize_fermionic_hamiltonian(
        mol_ham.one_body_tensor,
        mol_ham.two_body_tensor,
        bit_array,
        samples_per_batch=task.samples_per_batch,
        norb=norb,
        nelec=nelec,
        num_batches=task.n_batches,
        energy_tol=task.energy_tol,
        occupancies_tol=task.occupancies_tol,
        max_iterations=task.max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=task.symmetrize_spin,
        carryover_threshold=task.carryover_threshold,
        seed=rng,
        callback=callback,
        max_dim=task.max_dim,
    )
    energy = result.energy + mol_data.core_energy
    sci_state = result.sci_state
    spin_squared = sci_state.spin_square()
    error = energy - mol_data.fci_energy

    # Save data
    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
        "sci_vec_shape": sci_state.amplitudes.shape,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
