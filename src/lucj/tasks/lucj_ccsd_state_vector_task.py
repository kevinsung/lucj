import logging
import os
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
from ffsim.variational.util import interaction_pairs_spin_balanced
from molecules_catalog.util import load_molecular_data

from lucj.params import LUCJParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJCCSDStateVectorTask:
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


def run_lucj_ccsd_state_vector_task(
    task: LUCJCCSDStateVectorTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJCCSDStateVectorTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    state_vector_filename = data_dir / task.dirpath / "state_vector.npy"
    if (not overwrite) and os.path.exists(state_vector_filename):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian
    logging.info(f"{task} Loading molecular data...\n")
    mol_data = load_molecular_data(
        f"{task.molecule_basename}_d-{task.bond_distance:.5f}",
        molecules_catalog_dir=molecules_catalog_dir,
    )
    norb = mol_data.norb
    nelec = mol_data.nelec

    # Initialize reference state and LUCJ parameters
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )

    # use CCSD to initialize parameters
    logging.info(f"{task} Initializing ansatz from CCSD parameters...\n")
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        mol_data.ccsd_t2,
        n_reps=task.lucj_params.n_reps,
        t1=mol_data.ccsd_t1 if task.lucj_params.with_final_orbital_rotation else None,
        interaction_pairs=(pairs_aa, pairs_ab),
    )

    # Compute final state
    logging.info(f"{task} Computing state vector...\n")
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    logging.info(f"{task} Saving data...\n")
    with open(state_vector_filename, "wb") as f:
        np.save(f, final_state)
