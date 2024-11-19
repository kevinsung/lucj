import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from molecules_catalog.util import load_molecular_data
import ffsim
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class UCCSDInitialParamsTask:
    molecule_basename: str
    bond_distance: float | None

    @property
    def dirpath(self) -> Path:
        return Path(self.molecule_basename) / (
            ""
            if self.bond_distance is None
            else f"bond_distance-{self.bond_distance:.2f}"
        )


def run_uccsd_initial_params_task(
    task: UCCSDInitialParamsTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path,
    overwrite: bool = True,
) -> UCCSDInitialParamsTask:
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

    # use CCSD to initialize parameters
    operator = ffsim.UCCSDOpRestrictedReal(t1=mol_data.ccsd_t1, t2=mol_data.ccsd_t2)

    # Compute energy and other properties of final state vector
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

    energy = np.vdot(final_state, hamiltonian @ final_state).real
    error = energy - mol_data.fci_energy

    spin_squared = ffsim.spin_square(
        final_state, norb=mol_data.norb, nelec=mol_data.nelec
    )

    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
