import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
import pyscf
import ffsim
import numpy as np
from molecules_catalog.util import load_molecular_data
from ffsim.variational.util import interaction_pairs_spin_balanced
from lucj.params import LUCJParams, CompressedT2Params

from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, SCIResult
from qiskit_addon_sqd.counts import bit_array_to_arrays, bitstring_matrix_to_integers
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left
from qiskit_addon_dice_solver import solve_sci_batch
from qiskit_addon_dice_solver.dice_solver import DiceExecutionError
from lucj.hardware_sqd_task.hardware_job.hardware_job import (
    constrcut_lucj_circuit,
    run_on_hardware,
)

hardware_path = "dynamic_decoupling_xy"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class HardwareSQDEnergyTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    compressed_t2_params: CompressedT2Params
    connectivity_opt: bool = False
    random_op: bool = False
    shots: int
    samples_per_batch: int
    n_batches: int
    n_hardware_run: int
    energy_tol: float
    occupancies_tol: float
    carryover_threshold: float
    max_iterations: int
    symmetrize_spin: bool
    entropy: int | None
    max_dim: int | None
    dynamic_decoupling: bool = False

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


def load_operator(task: HardwareSQDEnergyTask, data_dir: str, mol_data):
    if task.random_op:
        logging.info(f"Generate random operator for {task}.\n")
        norb = mol_data.norb
        pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
            task.lucj_params.connectivity, norb
        )
        operator = ffsim.random.random_ucj_op_spin_balanced(
            norb,
            n_reps=task.lucj_params.n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=True,
        )
    elif task.connectivity_opt or task.compressed_t2_params is not None:
        operator_filename = data_dir / task.operatorpath / "operator.npz"
        if not os.path.exists(operator_filename):
            logging.info(f"Operator for {task} does not exists.\n")
            return None
        logging.info(f"Load operator for {task}.\n")
        operator = np.load(operator_filename)
        diag_coulomb_mats = operator["diag_coulomb_mats"]
        orbital_rotations = operator["orbital_rotations"]

        final_orbital_rotation = None
        if mol_data.ccsd_t1 is not None:
            final_orbital_rotation = (
                ffsim.variational.util.orbital_rotation_from_t1_amplitudes(
                    mol_data.ccsd_t1
                )
            )
        elif mol_data.ccsd_t2 is None:
            nelec = mol_data.nelec
            norb = mol_data.norb
            c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
                mol_data.cisd_vec, norb, nelec[0]
            )
            assert abs(c0) > 1e-8
            t1 = c1 / c0
            final_orbital_rotation = (
                ffsim.variational.util.orbital_rotation_from_t1_amplitudes(t1)
            )

        operator = ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )
    else:
        logging.info(f"Generate truncated operator for {task}.\n")
        norb = mol_data.norb
        nelec = mol_data.nelec
        pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
            task.lucj_params.connectivity, norb
        )
        if mol_data.ccsd_t2 is None:
            c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
                mol_data.cisd_vec, norb, nelec[0]
            )
            assert abs(c0) > 1e-8
            t1 = c1 / c0
            t2 = c2 / c0 - np.einsum("ia,jb->ijab", t1, t1)
            operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                t2,
                t1=t1,
                n_reps=task.lucj_params.n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=task.lucj_params.connectivity, norb=norb
                ),
            )
        else:
            operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                mol_data.ccsd_t2,
                n_reps=task.lucj_params.n_reps,
                t1=mol_data.ccsd_t1
                if task.lucj_params.with_final_orbital_rotation
                else None,
                interaction_pairs=(pairs_aa, pairs_ab),
            )

    return operator


def run_hardware_sqd_energy_task(
    task: HardwareSQDEnergyTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> HardwareSQDEnergyTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "hardware_sqd_data.pickle"
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

    # use CCSD to initialize parameters
    if task.dynamic_decoupling:
        sample_filename = (
            data_dir
            / task.operatorpath
            / hardware_path
            / f"n_hardware_run-{task.n_hardware_run}"
            / "hardware_sample.pickle"
        )
    else:
        sample_filename = data_dir / task.operatorpath / f"n_hardware_run-{task.n_hardware_run}/hardware_sample.pickle"

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
        samples = run_on_hardware(
            circuit,
            norb,
            1_000_000,
            sample_filename=sample_filename,
            dynamic_decoupling=task.dynamic_decoupling,
        )
        logging.info(f"{task} Finish sample\n")

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
        energy = [result.energy for result in results]
        result_history.append(min(energy) + mol_data.core_energy)
        iteration = len(result_history)
        logging.info(f"Iteration {iteration}")
        for i, result in enumerate(results):
            logging.info(f"\tSubsample {i}")
            logging.info(f"\t\tEnergy: {result.energy + mol_data.core_energy}")
            logging.info(
                f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
            )

    # Convert BitArray into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(samples)

    # Run configuration recovery loop
    # If we don't have average orbital occupancy information, simply postselect
    # bitstrings with the correct numbers of spin-up and spin-down electrons
    bitstrings, probs = postselect_by_hamming_right_and_left(
        raw_bitstrings, raw_probs, hamming_right=mol_data.nelec[0], hamming_left=mol_data.nelec[1]
    )

    unique_valid_bitstr, _ = np.unique(
        bitstring_matrix_to_integers(bitstrings), return_counts=True
    )
    logging.info(f"{task} #Valid bitstr: {bitstrings.shape}, #unique bitstr: {len(unique_valid_bitstr)}\n")

    # return 

    def solve_sci_batch_wrap(ci_strings, one_body_tensor, two_body_tensor, norb, nelec):
        solve = False
        while not solve:
            try:
                solve_sci_batch(ci_strings, one_body_tensor, two_body_tensor, norb, nelec)
                solve = True
            except DiceExecutionError:
                logging.info(f"{task} Dice execution error\n")
        

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
        sci_solver=solve_sci_batch_wrap,
        symmetrize_spin=task.symmetrize_spin,
        carryover_threshold=task.carryover_threshold,
        seed=rng,
        callback=callback,
        max_dim=task.max_dim
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
        "result_history": result_history,
        "spin_squared": spin_squared,
        "sci_vec_shape": sci_state.amplitudes.shape,
    }

    logging.info(f"{task} Saving SQD data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
