import dataclasses
import logging
import os
import pickle
import timeit
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
from ffsim.variational.util import (
    interaction_pairs_spin_balanced,
    orbital_rotation_to_parameters,
)
from molecules_catalog.util import load_molecular_data
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch

from lucj.params import COBYQAParams, LUCJParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJSQDCOBYQATask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    cobyqa_params: COBYQAParams
    shots: int
    samples_per_batch: int
    n_batches: int
    energy_tol: float
    occupancies_tol: float
    carryover_threshold: float
    max_iterations: int
    symmetrize_spin: bool
    entropy: int | None
    # TODO set limit on subspace dimension

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
            / self.cobyqa_params.dirpath
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


def run_lucj_sqd_cobyqa_task(
    task: LUCJSQDCOBYQATask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    bootstrap_task: LUCJSQDCOBYQATask | None = None,
    bootstrap_data_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJSQDCOBYQATask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filename = data_dir / task.dirpath / "result.pickle"
    info_filename = data_dir / task.dirpath / "info.pickle"
    if (
        (not overwrite)
        and os.path.exists(result_filename)
        and os.path.exists(info_filename)
    ):
        logger.info(f"Data for {task} already exists. Skipping...\n")
        return task
    intermediate_result_filename = (
        data_dir / task.dirpath / "intermediate_result.pickle"
    )

    # Get molecular data and molecular Hamiltonian
    molecule_basename = task.molecule_basename
    if task.bond_distance is not None:
        molecule_basename += f"_d-{task.bond_distance:.5f}"
    mol_data = load_molecular_data(
        molecule_basename,
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

    rng = np.random.default_rng(task.entropy)

    def fun(x: np.ndarray) -> float:
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
            x,
            norb=norb,
            n_reps=task.lucj_params.n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=task.lucj_params.with_final_orbital_rotation,
        )
        final_state = ffsim.apply_unitary(
            reference_state, operator, norb=norb, nelec=nelec
        )
        samples = ffsim.sample_state_vector(
            final_state,
            norb=norb,
            nelec=nelec,
            shots=task.shots,
            seed=rng,
            bitstring_type=ffsim.BitstringType.INT,
        )
        bit_array = BitArray.from_samples(samples, num_bits=2 * norb)
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
            sci_solver=solve_sci_batch,
            symmetrize_spin=task.symmetrize_spin,
            carryover_threshold=task.carryover_threshold,
            seed=rng,
        )
        return result.energy + mol_data.core_energy

    # Generate initial parameters
    if bootstrap_task is None:
        # use CCSD to initialize parameters
        op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            mol_data.ccsd_t2,
            n_reps=task.lucj_params.n_reps,
            t1=mol_data.ccsd_t1
            if task.lucj_params.with_final_orbital_rotation
            else None,
            interaction_pairs=(pairs_aa, pairs_ab),
        )
        params = op.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
    else:
        bootstrap_result_filename = os.path.join(
            bootstrap_data_dir or data_dir, bootstrap_task.dirpath, "result.pickle"
        )
        with open(bootstrap_result_filename, "rb") as f:
            result = pickle.load(f)
            params = result.x
        if bootstrap_task.lucj_params.n_reps < task.lucj_params.n_reps:
            n_params = ffsim.UCJOpSpinBalanced.n_params(
                norb=norb,
                n_reps=task.lucj_params.n_reps,
                interaction_pairs=(pairs_aa, pairs_ab),
                with_final_orbital_rotation=bootstrap_task.lucj_params.with_final_orbital_rotation,
            )
            params = np.concatenate([params, np.zeros(n_params - len(params))])
        if (
            task.lucj_params.with_final_orbital_rotation
            and not bootstrap_task.lucj_params.with_final_orbital_rotation
        ):
            params = np.concatenate([params, np.zeros(norb**2)])
            params[-(norb**2) :] = orbital_rotation_to_parameters(
                np.eye(norb, dtype=complex)
            )

    # Optimize ansatz
    logger.info(f"{task} Optimizing ansatz...\n")
    info = defaultdict(list)
    info["nit"] = 0

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        logger.info(f"Task {task} is on iteration {info['nit']}.\n")
        logger.info(f"\tObjective function value: {intermediate_result.fun}.\n")
        info["x"].append(intermediate_result.x)
        info["fun"].append(intermediate_result.fun)
        info["nit"] += 1
        with open(intermediate_result_filename, "wb") as f:
            pickle.dump(intermediate_result, f)

    t0 = timeit.default_timer()
    result = scipy.optimize.minimize(
        fun,
        x0=params,
        method="COBYQA",
        options=dataclasses.asdict(task.cobyqa_params),
        callback=callback,
    )
    t1 = timeit.default_timer()
    logger.info(f"{task} Done optimizing ansatz in {t1 - t0} seconds.\n")

    logger.info(f"{task} Saving data...\n")
    with open(result_filename, "wb") as f:
        pickle.dump(result, f)
    with open(info_filename, "wb") as f:
        pickle.dump(info, f)
