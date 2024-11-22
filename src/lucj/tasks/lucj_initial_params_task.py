import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.stats
from molecules_catalog.util import load_molecular_data

from lucj.params import LUCJParams
from lucj.util import interaction_pairs_spin_balanced

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJInitialParamsTask:
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
                else f"bond_distance-{self.bond_distance:.2f}"
            )
            / self.lucj_params.dirpath
        )


def run_lucj_initial_params_task(
    task: LUCJInitialParamsTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJInitialParamsTask:
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
    mol_hamiltonian = mol_data.hamiltonian

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
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
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    # Compute energy and other properties of final state vector
    logging.info(f"{task} Computing energy and other properties...\n")
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
        "n_reps": operator.n_reps,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
