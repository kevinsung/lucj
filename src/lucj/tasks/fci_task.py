import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.stats
from molecules_catalog.util import load_molecular_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FCITask:
    molecule_basename: str
    bond_distance: float | None

    @property
    def dirpath(self) -> Path:
        return Path(self.molecule_basename) / (
            ""
            if self.bond_distance is None
            else f"bond_distance-{self.bond_distance:.2f}"
        )


def run_fci_task(
    task: FCITask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> FCITask:
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

    # Run FCI
    previous_fci_energy = mol_data.fci_energy
    mol_data.run_fci(store_fci_vec=True)
    np.testing.assert_allclose(mol_data.fci_energy, previous_fci_energy)
    fci_vec = mol_data.fci_vec.reshape(-1)

    # Compute energy and other properties of final state vector
    spin_squared = ffsim.spin_square(fci_vec, norb=mol_data.norb, nelec=mol_data.nelec)
    probs = np.abs(fci_vec) ** 2
    entropy = scipy.stats.entropy(probs)

    data = {
        "energy": mol_data.fci_energy,
        "spin_squared": spin_squared,
        "entropy": entropy,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
