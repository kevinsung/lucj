import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.stats
from ffsim.variational.util import interaction_pairs_spin_balanced
from molecules_catalog.util import load_molecular_data
from lucj.params import LUCJParams
from lucj.tasks.lucj_compressed_t2_task_ffsim.compressed_t2 import double_factorized_t2_compress
from opt_einsum import contract

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class UCCSDCompressedT2ParamsTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams

    @property
    def dirpath(self) -> Path:
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.5f}"
            )
            / self.lucj_params.dirpath
        )


def run_uccsd_compressed_t2_params_task(
    task: UCCSDCompressedT2ParamsTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> UCCSDCompressedT2ParamsTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "data.pickle"
    if (not overwrite) and os.path.exists(data_filename):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian
    mol_data = load_molecular_data(
        f"{task.molecule_basename}_d-{task.bond_distance:.5f}",
        molecules_catalog_dir=molecules_catalog_dir,
    )
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # Initialize Hamiltonian and initial state
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )


    # use CCSD to initialize parameters
    nocc, _, nvrt, _ = mol_data.ccsd_t2.shape
    operator_filename = data_dir / task.dirpath / "operator.npz"
    
    if (not overwrite) and os.path.exists(operator_filename):
        logging.info(f"Operator for {task} already exists. Skipping...\n")
        operator = np.load(operator_filename)
        diag_coulomb_mats = operator["diag_coulomb_mats"]
        orbital_rotations = operator["orbital_rotations"]
    else:
        logging.info(f"{task} Construct compressed t2\n")
        diag_coulomb_mats, orbital_rotations, _, _ = (
                double_factorized_t2_compress(
                    mol_data.ccsd_t2,
                    nocc=nocc,
                    n_reps=task.lucj_params.n_reps,
                    interaction_pairs=(pairs_aa, pairs_ab),
                    multi_stage_optimization=True,
                )
            )
        np.savez(operator_filename, diag_coulomb_mats=diag_coulomb_mats, orbital_rotations=orbital_rotations)
    
    diag_coulomb_mats = np.unstack(diag_coulomb_mats, axis=1)[0]
    
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
    logging.info(f"{task} UCCSDOpRestricted\n")
    operator = ffsim.UCCSDOpRestricted(t1=mol_data.ccsd_t1, t2=t2_reconstructed)

    # Compute energy and other properties of final state vector
    logging.info(f"{task} Compute final state\n")
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    energy = np.vdot(final_state, hamiltonian @ final_state).real
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

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
