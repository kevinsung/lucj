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
import scipy.stats
from molecules_catalog.util import load_molecular_data
from lucj.params import LUCJParams, CompressedT2Params
from opt_einsum import contract

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class UCCSDCompressedTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    compressed_t2_params: CompressedT2Params
    connectivity_opt: bool = False
    random_op: bool = False
    regularization: bool = (False,)
    regularization_option: int = (0,)

    @property
    def dirpath(self) -> Path:
        compress_option = self.compressed_t2_params.dirpath
        if self.regularization:
            compress_option = (
                f"{compress_option}/regularization_{self.regularization_option}"
            )
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


def load_operator(task: UCCSDCompressedTask, data_dir: str, mol_data):
    operator_filename = data_dir / task.dirpath / "operator.npz"
    if not os.path.exists(operator_filename):
        logging.info(f"Operator for {task} does not exists.\n")
        return None
    logging.info(f"Load operator for {task}.\n")
    operator = np.load(operator_filename)
    diag_coulomb_mats = operator["diag_coulomb_mats"]
    orbital_rotations = operator["orbital_rotations"]

    diag_coulomb_mats = np.unstack(diag_coulomb_mats, axis=1)[0]
    nocc, _, nvrt, _ = mol_data.ccsd_t2.shape
    t2_reconstructed = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
            # optimize="greedy"
        )[:nocc, :nocc, nocc:, nocc:]
    )
    operator = ffsim.UCCSDOpRestricted(t1=mol_data.ccsd_t1, t2=t2_reconstructed)
    return operator


def run_vqe_energy_task(
    task: UCCSDCompressedTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> UCCSDCompressedTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "data_uccsd.pickle"
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

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    # use CCSD to initialize parameters
    state_vector_filename = data_dir / task.dirpath / "state_vector_uccsd.npy"

    if os.path.exists(state_vector_filename):
        with open(state_vector_filename, "rb") as f:
            final_state = np.load(f)
    else:
        operator = load_operator(task, data_dir, mol_data)
        if operator is None:
            return

        # Compute final state
        if not os.path.exists(state_vector_filename):
            final_state = ffsim.apply_unitary(
                reference_state, operator, norb=norb, nelec=nelec
            )
            with open(state_vector_filename, "wb") as f:
                np.save(f, final_state)

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
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
