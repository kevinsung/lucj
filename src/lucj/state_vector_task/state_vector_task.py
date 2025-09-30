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
import pyscf
import ffsim
import numpy as np
from molecules_catalog.util import load_molecular_data
from ffsim.variational.util import interaction_pairs_spin_balanced
from lucj.params import LUCJParams, CompressedT2Params

from qiskit.primitives import BitArray

logger = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class StateVecTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    compressed_t2_params: CompressedT2Params | None
    connectivity_opt: bool = False
    random_op: bool = False
    regularization: bool = False
    regularization_option: int | None = None
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
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        elif self.compressed_t2_params is not None:
            compress_option = self.compressed_t2_params.dirpath
            if self.regularization:
                compress_option = f"{compress_option}/regularization_{self.regularization_option}"
        else:
            compress_option = "truncated"
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.5f}"
            )
            / self.lucj_params.dirpath
            / compress_option
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
    def operatorpath(self) -> Path:
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        elif self.compressed_t2_params is not None:
            compress_option = self.compressed_t2_params.dirpath
            if self.regularization:
                compress_option = f"{compress_option}/regularization_{self.regularization_option}"
        else:
            compress_option = "truncated"
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.5f}"
            )
            / self.lucj_params.dirpath
            / compress_option
        )


def load_operator(task: StateVecTask, data_dir: str, mol_data):
    if task.random_op:
        logging.info(f"Generate random operator for {task}.\n")
        norb = mol_data.norb
        pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
            task.lucj_params.connectivity, norb
        )
        operator = ffsim.random.random_ucj_op_spin_balanced(
            norb,
            n_reps=task.lucj_params.n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=True
        )
    elif task.connectivity_opt or task.compressed_t2_params is not None:
        operator_filename = data_dir / task.operatorpath / "operator.npz"
        if not os.path.exists(operator_filename):
            logging.info(f"Operator for {task} does not exists.\n")
            return None
        logging.info(f"Load operator for {task}.\n")
        operator = np.load(operator_filename)
        diag_coulomb_mats = operator["diag_coulomb_mats"]
        orbital_rotations = operator["orbital_rotations"]

        final_orbital_rotation = None
        if mol_data.ccsd_t1 is not None:
            final_orbital_rotation = (
                ffsim.variational.util.orbital_rotation_from_t1_amplitudes(mol_data.ccsd_t1)
            )
        elif mol_data.ccsd_t2 is None:
            nelec = mol_data.nelec
            norb = mol_data.norb
            c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
                mol_data.cisd_vec, norb, nelec[0]
            )
            assert abs(c0) > 1e-8
            t1 = c1 / c0
            final_orbital_rotation = (
                ffsim.variational.util.orbital_rotation_from_t1_amplitudes(t1)
            )

        operator = ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )
    else:
        logging.info(f"Generate truncated operator for {task}.\n")
        norb = mol_data.norb
        nelec = mol_data.nelec
        pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
            task.lucj_params.connectivity, norb
        )
        if mol_data.ccsd_t2 is None:
            c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
                mol_data.cisd_vec, norb, nelec[0]
            )
            assert abs(c0) > 1e-8
            t1 = c1 / c0
            t2 = c2 / c0 - np.einsum("ia,jb->ijab", t1, t1)
            operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                t2,
                t1=t1,
                n_reps=task.lucj_params.n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=task.lucj_params.connectivity, norb=norb
                ),
            ) 
        else:
            operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                mol_data.ccsd_t2,
                n_reps=task.lucj_params.n_reps,
                t1=mol_data.ccsd_t1 if task.lucj_params.with_final_orbital_rotation else None,
                interaction_pairs=(pairs_aa, pairs_ab),
            )
    return operator

def run_state_vec_task(
    task: StateVecTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> StateVecTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    sample_filename = data_dir / task.operatorpath / "sample.pickle"
    state_vector_filename = data_dir / task.operatorpath / "state_vector.npy"

    if (not overwrite) and os.path.exists(state_vector_filename):
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

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    # use CCSD to initialize parameters
    
    rng = np.random.default_rng(task.entropy)
    
    if not os.path.exists(sample_filename):
        if os.path.exists(state_vector_filename):
            with open(state_vector_filename, "rb") as f:
                final_state = np.load(f)
        else:
            operator = load_operator(task, data_dir, mol_data)
            if operator is None:
                return
            
            # Compute final state
            if not os.path.exists(state_vector_filename):
                logging.info(f"{task} compute state vector...\n")
                final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
                with open(state_vector_filename, "wb") as f:
                    np.save(f, final_state)

        logging.info(f"{task} Sampling...\n")
        samples = ffsim.sample_state_vector(
            final_state,
            norb=norb,
            nelec=nelec,
            shots=1_000_000,
            seed=rng,
            bitstring_type=ffsim.BitstringType.INT,
        )
        bit_array = BitArray.from_samples(samples, num_bits=2 * norb)
        bit_array_count = bit_array.get_int_counts()
        with open(sample_filename, "wb") as f:
            pickle.dump(bit_array_count, f)
    
    else:
        logging.info(f"{task} load sample...\n")
        with open(sample_filename, "rb") as f:
            bit_array_count = pickle.load(f)
            bit_array = BitArray.from_counts(bit_array_count)