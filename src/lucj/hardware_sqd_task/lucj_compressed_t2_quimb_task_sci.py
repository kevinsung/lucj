import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
import ffsim
import numpy as np
from molecules_catalog.util import load_molecular_data
from ffsim.variational.util import interaction_pairs_spin_balanced

from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, SCIResult, solve_sci_batch
from lucj.hardware_sqd_task.hardware_job.hardware_job import (
    constrcut_lucj_circuit,
    run_on_hardware,
)
from lucj.quimb_task.lucj_sqd_quimb_task import LUCJSQDQuimbTask


logger = logging.getLogger(__name__)

hardware_path = "dynamic_decoupling_xy_opt_0"

@dataclass(frozen=True, kw_only=True)
class HardwareSQDQuimbEnergyTask:
    lucj_sqd_quimb_task: LUCJSQDQuimbTask
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
    dynamic_decoupling: bool = True
    n_hardware_run: int

    @property
    def dirpath(self) -> Path:
        return (
            self.lucj_sqd_quimb_task.dirpath
            / ("" if self.dynamic_decoupling is False else hardware_path)
            / f"n_hardware_run-{self.n_hardware_run}"
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


def load_operator(task: HardwareSQDQuimbEnergyTask, data_dir: str, mol_data):
    result_filename = data_dir / task.lucj_sqd_quimb_task.dirpath / "data.pickle"
    if not os.path.exists(result_filename):
        logging.info(f"Operator for {task} does not exists.\n")
        print(result_filename)
        input()
        return None
    logging.info(f"Load operator for {task}.\n")
    with open(result_filename, "rb") as f:
        result = pickle.load(f)

    norb = mol_data.norb
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_sqd_quimb_task.lucj_params.connectivity, norb
    )

    operator = ffsim.UCJOpSpinBalanced.from_parameters(
        np.array(result['x_best']),
        norb=norb,
        n_reps=task.lucj_sqd_quimb_task.lucj_params.n_reps,
        interaction_pairs=(pairs_aa, pairs_ab),
        with_final_orbital_rotation=task.lucj_sqd_quimb_task.lucj_params.with_final_orbital_rotation,
    )
    return operator


def run_hardware_sqd_energy_task(
    task: HardwareSQDQuimbEnergyTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> HardwareSQDQuimbEnergyTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "hardware_sqd_data.pickle"
    if (not overwrite) and os.path.exists(data_filename):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian
    if task.lucj_sqd_quimb_task.molecule_basename == "fe2s2_30e20o":
        mol_data = load_molecular_data(
            task.lucj_sqd_quimb_task.molecule_basename,
            molecules_catalog_dir=molecules_catalog_dir,
        )
    else:
        mol_data = load_molecular_data(
            f"{task.lucj_sqd_quimb_task.molecule_basename}_d-{task.lucj_sqd_quimb_task.bond_distance:.5f}",
            molecules_catalog_dir=molecules_catalog_dir,
        )
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # use CCSD to initialize parameters
    sample_filename = data_dir / task.lucj_sqd_quimb_task.dirpath / f"hardware_sample_{task.n_hardware_run}.pickle"

    rng = np.random.default_rng(task.entropy)

    if not os.path.exists(sample_filename):
        # assert 0
        operator = load_operator(task, data_dir, mol_data)
        if operator is None:
            return
        # construct lucj circuit
        circuit = constrcut_lucj_circuit(norb, nelec, operator)

        # run on hardware and get the sample
        logging.info(f"{task} Sampling from real device...\n")
        samples = run_on_hardware(circuit, norb, 1_000_000, sample_filename, task.dynamic_decoupling)
    else:
        logging.info(f"{task} load sample...\n")
        with open(sample_filename, "rb") as f:
            samples = pickle.load(f)

    logging.info(f"{task} Done sampling\n")
    # print(samples)
    samples = samples[: task.shots]
    # print(samples)

    # Run SQD
    logging.info(f"{task} Running SQD...\n")
    # sci_solver = partial(solve_sci_batch, spin_sq=0.0)
    result_history = []

    def callback(results: list[SCIResult]):
        result_history.append(results)
        iteration = len(result_history)
        logging.info(f"Iteration {iteration}")
        for i, result in enumerate(results):
            logging.info(f"\tSubsample {i}")
            logging.info(f"\t\tEnergy: {result.energy + mol_data.core_energy}")
            logging.info(
                f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
            )

    # # Convert BitArray into bitstring and probability arrays
    # raw_bitstrings, raw_probs = bit_array_to_arrays(samples)

    # # Run configuration recovery loop
    # # If we don't have average orbital occupancy information, simply postselect
    # # bitstrings with the correct numbers of spin-up and spin-down electrons
    # bitstrings, probs = postselect_by_hamming_right_and_left(
    #     raw_bitstrings, raw_probs, hamming_right=mol_data.nelec[0], hamming_left=mol_data.nelec[1]
    # )

    result = diagonalize_fermionic_hamiltonian(
        mol_hamiltonian.one_body_tensor,
        mol_hamiltonian.two_body_tensor,
        samples,
        samples_per_batch=task.samples_per_batch,
        norb=norb,
        nelec=nelec,
        num_batches=task.n_batches,
        energy_tol=task.energy_tol,
        occupancies_tol=task.occupancies_tol,
        max_iterations=task.max_iterations,
        sci_solver=solve_sci_batch,
        symmetrize_spin=task.symmetrize_spin,
        carryover_threshold=task.carryover_threshold,
        seed=rng,
        max_dim=task.max_dim,
        callback=callback,
    )
    energy = result.energy + mol_data.core_energy
    sci_state = result.sci_state
    spin_squared = sci_state.spin_square()
    if mol_data.fci_energy is not None:
        error = energy - mol_data.fci_energy
    elif mol_data.sci_energy is not None:
        error = energy - mol_data.sci_energy
    else:
        error = -1

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
