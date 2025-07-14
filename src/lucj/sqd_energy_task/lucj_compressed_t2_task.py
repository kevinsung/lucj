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

from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
from functools import partial


logger = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class SQDEnergyTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    compressed_t2_params: CompressedT2Params | None
    connectivity_opt: bool = False
    random_op: bool = False
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


def run_sqd_energy_task(
    task: SQDEnergyTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> SQDEnergyTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "sqd_data.pickle"
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

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)

    # use CCSD to initialize parameters
    operator_filename = data_dir / task.operatorpath / "operator.npz"
    vqe_filename = data_dir / task.operatorpath / "data.pickle"
    sample_filename = data_dir / task.operatorpath / "sample.pickle"
    state_vector_filename = data_dir / task.operatorpath / "state_vector.npy"
    
    if not os.path.exists(sample_filename):
        if os.path.exists(state_vector_filename):
            with open(state_vector_filename, "rb") as f:
                final_state = np.load(f)
        else:
            if not os.path.exists(operator_filename):
                logging.info(f"Operator for {task} does not exists.\n")

            operator = np.load(operator_filename)
            diag_coulomb_mats = operator["diag_coulomb_mats"]
            orbital_rotations = operator["orbital_rotations"]
            
            final_orbital_rotation = None
            if mol_data.ccsd_t1 is not None:
                final_orbital_rotation = (
                    ffsim.variational.util.orbital_rotation_from_t1_amplitudes(mol_data.ccsd_t1)
                )

            operator = ffsim.UCJOpSpinBalanced(
                    diag_coulomb_mats=diag_coulomb_mats,
                    orbital_rotations=orbital_rotations,
                    final_orbital_rotation=final_orbital_rotation,
                )
            
            # Compute final state
            if not os.path.exists(state_vector_filename):
                final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
                with open(state_vector_filename, "wb") as f:
                    np.save(f, final_state)
        
        # record vqe energy
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

        logging.info(f"{task} Saving VQE data...\n")
        with open(vqe_filename, "wb") as f:
            pickle.dump(data, f)


        logging.info(f"{task} Sampling...\n")
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
        bit_array_count = bit_array.get_int_counts()
        with open(sample_filename, "wb") as f:
            pickle.dump(bit_array_count, f)
    
    else:
        with open(sample_filename, "rb") as f:
            bit_array_count = pickle.load(f)
            bit_array = BitArray.from_counts(bit_array_count)


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
    }
    
    logging.info(f"{task} Saving SQD data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)





