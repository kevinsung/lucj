import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
from ffsim.variational.util import interaction_pairs_spin_balanced
from molecules_catalog.util import load_molecular_data
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
from functools import partial
from lucj.params import LUCJParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJSQDRandomTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    shots: int
    samples_per_batch: int
    n_batches: int
    energy_tol: float
    occupancies_tol: float
    carryover_threshold: float
    max_iterations: int
    symmetrize_spin: bool
    entropy: int | None

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
            / f"shots-{self.shots}"
            / f"samples_per_batch-{self.samples_per_batch}"
            / f"n_batches-{self.n_batches}"
            / f"energy_tol-{self.energy_tol}"
            / f"occupancies_tol-{self.occupancies_tol}"
            / f"carryover_threshold-{self.carryover_threshold}"
            / f"max_iterations-{self.max_iterations}"
            / f"symmetrize_spin-{self.symmetrize_spin}"
            / f"entropy-{self.entropy}"
        )


def run_lucj_sqd_random_task(
    task: LUCJSQDRandomTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJSQDRandomTask:
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
    mol_ham = mol_data.hamiltonian
    norb = mol_data.norb
    nelec = mol_data.nelec

    # Initialize initial state and LUCJ parameters
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )

    # use CCSD to initialize parameters
    operator = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=task.lucj_params.n_reps,
        interaction_pairs=(pairs_aa, pairs_ab),
        with_final_orbital_rotation=True
    )
    
    # Compute final state
    logging.info(f"{task} Computing state vector...\n")
    final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

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
        "n_reps": operator.n_reps,
        "init_loss": init_loss,
        "final_loss": final_loss,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
