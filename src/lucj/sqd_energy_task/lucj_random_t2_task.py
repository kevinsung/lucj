import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from molecules_catalog.util import load_molecular_data

from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
from functools import partial


logger = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class RandomSQDEnergyTask:
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
            / "random_sample"
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
    random_int_list = np.random.randint(2, size=(task.shots, norb*2))
    random_bit_string = []
    for random_int in random_int_list:
        random_bit_string.append(''.join(str(x) for x in random_int))

    bit_array = BitArray.from_samples(random_bit_string, num_bits=2 * norb)

    # Run SQD
    logging.info(f"{task} Running SQD...\n")
    sci_solver = partial(solve_sci_batch, spin_sq=0.0)
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
        sci_solver=sci_solver,
        symmetrize_spin=task.symmetrize_spin,
        carryover_threshold=task.carryover_threshold,
        seed=rng,
        max_dim=task.max_dim
    )
    logging.info(f"{task} Finish SQD\n")
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
    
    logging.info(f"{task} Saving SQD data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)





