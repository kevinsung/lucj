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

import numpy as np
from molecules_catalog.util import load_molecular_data

from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, SCIResult
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform
# from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left
from qiskit_addon_dice_solver import solve_sci_batch
from qiskit_addon_dice_solver.dice_solver import DiceExecutionError
# from functools import partial


logger = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class RandomSQDEnergyTask:
    molecule_basename: str
    bond_distance: float | None
    shots: int
    valid_string_only: bool = False,
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
            / "random_sample"
            / (
                ""
                if not self.valid_string_only
                else "valid_string_only"
            )
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

def run_random_sqd_energy_task(
    task: RandomSQDEnergyTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> RandomSQDEnergyTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "sqd_data.pickle"
    
    if (not overwrite) and os.path.exists(data_filename):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task
    

    # Get molecular data and molecular Hamiltonian
    if task.molecule_basename == "fe2s2_30e20o":
        mol_data = load_molecular_data(
            task.molecule_basename,
            molecules_catalog_dir=molecules_catalog_dir,
        )
    else:
        mol_data = load_molecular_data(
            f"{task.molecule_basename}_d-{task.bond_distance:.5f}",
            molecules_catalog_dir=molecules_catalog_dir,
        )
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    logging.info(f"{task} Sampling...\n")
    rng = np.random.default_rng(task.entropy)
    random_bit_string = []
    if task.valid_string_only:
        for i in range(task.shots):
            right_bit_str = ['0' for _ in range(norb)]
            left_bit_str = ['0' for _ in range(norb)]
            bit_one_locs_right = np.random.choice(norb, nelec[0], replace=False)
            bit_one_locs_left = np.random.choice(norb, nelec[0], replace=False)
            for i in bit_one_locs_right:
                right_bit_str[i] = '1'
            for i in bit_one_locs_left:
                left_bit_str[i] = '1'
            random_bit_string.append(''.join(x for x in right_bit_str + left_bit_str))
        bit_array = BitArray.from_samples(random_bit_string, num_bits=2 * norb)

    else:
        bit_array = generate_bit_array_uniform(task.shots, 2 * norb, rand_seed=rng)
    
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

    # # Run configuration recovery loop
    # raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    # # If we don't have average orbital occupancy information, simply postselect
    # # bitstrings with the correct numbers of spin-up and spin-down electrons
    # bitstrings, probs = postselect_by_hamming_right_and_left(
    #     raw_bitstrings, raw_probs, hamming_right=nelec[0], hamming_left=nelec[1]
    # )
    # print(f"len valid bitstr: {len(bitstrings)}")
    # assert(0)

    # Run SQD
    logging.info(f"{task} Running SQD...\n")
    # sci_solver = partial(solve_sci_batch, spin_sq=0.0)
    solve = False
    while not solve:
        try:
            result = diagonalize_fermionic_hamiltonian(
                mol_hamiltonian.one_body_tensor,
                mol_hamiltonian.two_body_tensor,
                bit_array,
                samples_per_batch=task.samples_per_batch,
                norb=norb,
                nelec=nelec,
                num_batches=task.n_batches,
                energy_tol=task.energy_tol,
                occupancies_tol=task.occupancies_tol,
                max_iterations=task.max_iterations,
                sci_solver=solve_sci_batch,
                symmetrize_spin=task.symmetrize_spin,
                carryover_threshold=task.carryover_threshold,
                seed=rng,
                callback=callback,
                max_dim=task.max_dim
            )
            solve = True
        except DiceExecutionError:
            logging.info(f"{task} Dice execution error\n")
    logging.info(f"{task} Finish SQD\n")
    energy = result.energy + mol_data.core_energy
    sci_state = result.sci_state
    spin_squared = sci_state.spin_square()
    if mol_data.fci_energy is None:
        error = energy - mol_data.sci_energy
    else:
        error = energy - mol_data.fci_energy

    # Save data
    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
        "sci_vec_shape": sci_state.amplitudes.shape,
        "history_energy": result_history_energy,
        "history_sci_vec_shape": result_history_subspace_dim
    }
    
    logging.info(f"{task} Saving SQD data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)