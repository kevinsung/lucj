# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import dataclasses
import logging
import os
import pickle
import timeit
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from functools import partial

import ffsim
import numpy as np
import scipy.optimize
from ffsim.variational.util import interaction_pairs_spin_balanced
from ffsim.linalg.util import unitaries_to_parameters
from molecules_catalog.util import load_molecular_data
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
)
from qiskit_addon_dice_solver import solve_sci_batch
from lucj.operator_task.lucj_compressed_t2_task_ffsim.compressed_t2 import (
    from_t_amplitudes_compressed,
)
from lucj.params import COBYQAParams, LUCJParams, CompressedT2Params
from qiskit.circuit import QuantumCircuit, QuantumRegister
import quimb.tensor
from qiskit_quimb import quimb_circuit

import PyNomad

import pyscf
import jax

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJSQDQuimbTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    cobyqa_params: COBYQAParams
    compressed_t2_params: CompressedT2Params | None
    connectivity_opt: bool = False
    random_op: bool = False
    regularization: bool = (False,)
    regularization_option: int = (0,)
    shots: int
    samples_per_batch: int
    n_batches: int
    energy_tol: float
    occupancies_tol: float
    carryover_threshold: float
    max_iterations: int
    symmetrize_spin: bool
    entropy: int | None
    max_bond: int
    perm_mps: bool
    cutoff: int
    seed: int
    max_dim: int

    @property
    def dirpath(self) -> Path:
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        elif self.compressed_t2_params is not None:
            compress_option = self.compressed_t2_params.dirpath
            if self.regularization:
                compress_option = (
                    f"{compress_option}/regularization_{self.regularization_option}"
                )
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
            / "quimb"
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
            / f"max_dim-{self.max_dim}"
            / f"max_bond-{self.max_bond}"
            / f"cutoff-{self.cutoff}"
            / f"perm_mps-{self.perm_mps}"
            / f"seed-{self.seed}"
        )

    @property
    def operatorpath(self) -> Path:
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        elif self.compressed_t2_params is not None:
            compress_option = self.compressed_t2_params.dirpath
            if self.regularization:
                compress_option = (
                    f"{compress_option}/regularization_{self.regularization_option}"
                )
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


def load_operator(task: LUCJSQDQuimbTask, data_dir: str, mol_data):
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


def to_backend(x):
    # return jnp(x, dtype=torch.complex64, device="cuda")
    return jax.device_put(x)


def run_lucj_sqd_quimb_task(
    task: LUCJSQDQuimbTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    bootstrap_task: LUCJSQDQuimbTask | None = None,
    bootstrap_data_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJSQDQuimbTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filename = data_dir / task.dirpath / "data.pickle"
    info_filename = data_dir / task.dirpath / "info.pickle"
    state_vector_filename = data_dir / task.dirpath / "state_vector.npy"
    sample_filename = data_dir / task.dirpath / "sample.pickle"
    sqd_result_filename = data_dir / task.dirpath / "sqd_data.pickle"
    cache_file = data_dir / task.dirpath / "cache.txt"

    if (
        (not overwrite)
        and os.path.exists(result_filename)
        and os.path.exists(info_filename)
        and os.path.exists(sqd_result_filename)
    ):
        logger.info(f"Data for {task} already exists. Skipping...\n")
        return task
    # intermediate_result_filename = data_dir / task.dirpath / "intermediate_data.pickle"

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

    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )

    rng = np.random.default_rng(task.entropy)

    result_history_energy = []
    result_history_subspace_dim = []
    result_history = []

    def callback(results: list[SCIResult]):
        result_energy = []
        result_subspace_dim = []
        iteration = len(result_history)
        result_history.append(results)
        logging.info(f"Iteration {iteration}")
        for i, result in enumerate(results):
            result_energy.append(result.energy + mol_data.core_energy)
            result_subspace_dim.append(result.sci_state.amplitudes.shape)
            logging.info(f"\tSubsample {i}")
            logging.info(f"\t\tEnergy: {result.energy + mol_data.core_energy}")
            logging.info(
                f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
            )
        result_history_energy.append(result_energy)
        result_history_subspace_dim.append(result_subspace_dim)

    def fun(bbo_x) -> bool:
        dim = bbo_x.size()
        x = np.array([bbo_x.get_coord(i) for i in range(dim)])
        # x = bbo_x
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
            x,
            norb=norb,
            n_reps=task.lucj_params.n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=task.lucj_params.with_final_orbital_rotation,
        )
        # Construct Qiskit circuit
        qubits = QuantumRegister(2 * norb)
        circuit = QuantumCircuit(qubits)
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
        circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator), qubits)
        # change to quimb
        # Sample using quimb
        decomposed = circuit.decompose(reps=2)
        quimb_circ = quimb_circuit(
            decomposed,
            quimb_circuit_class=quimb.tensor.CircuitPermMPS
            if task.perm_mps
            else quimb.tensor.CircuitMPS,
            max_bond=task.max_bond,
            cutoff=task.cutoff,
            progbar=True,
            # to_backend=to_backend,
        )
        # assert(0)
        logger.info(f"{task}\n\tSampling circuit...")
        t0 = timeit.default_timer()
        # quimb_circ.apply_to_arrays(lambda x: x.cpu().numpy())
        # samples = list(quimb_circ.sample(task.shots, seed=task.seed, backend='jax'))
        samples = list(quimb_circ.sample(task.shots, seed=task.seed))
        t1 = timeit.default_timer()
        time = t1 - t0
        logger.info(f"{task}\n\tDone sampling circuit in {time} seconds.")
        samples = [sample[::-1] for sample in samples]
        # assert(0)
        bit_array = BitArray.from_samples(samples, num_bits=2 * norb)
        # sci_solver = partial(solve_sci_batch, spin_sq=0.0)
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
            max_dim=task.max_dim,
            callback=callback,
        )
        logger.info(f"{task}\n\tEnergy: {(result.energy + mol_data.core_energy)}")
        f = result.energy + mol_data.core_energy
        bbo_x.setBBO(str(f).encode("UTF-8"))
        return 1

    if overwrite or (
        not os.path.exists(result_filename) and not os.path.exists(info_filename)
    ):
        # Generate initial parameters
        if bootstrap_task is None:
            # use CCSD to initialize parameters
            operator = load_operator(task, data_dir, mol_data)
            if operator is None:
                operator, _, _ = from_t_amplitudes_compressed(
                    mol_data.ccsd_t2,
                    n_reps=task.lucj_params.n_reps,
                    t1=mol_data.ccsd_t1
                    if task.lucj_params.with_final_orbital_rotation
                    else None,
                    interaction_pairs=(pairs_aa, pairs_ab),
                    optimize=True,
                )
            params = operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            bootstrap_result_filename = os.path.join(
                bootstrap_data_dir or data_dir, bootstrap_task.dirpath, "data.pickle"
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
                params[-(norb**2) :] = unitaries_to_parameters(
                    np.eye(norb, dtype=complex)
                )

        # Optimize ansatz
        logger.info(f"{task} Optimizing ansatz...\n")

        t0 = timeit.default_timer()

        # fun(params)
        # data = {
        #     "history_energy": result_history_energy,
        #     "history_sci_vec_shape": result_history_subspace_dim,
        # }
        # sqd_init_result_filename = data_dir / task.dirpath / "init_mps_sqd_result_filename.pickle"
        # logger.info(f"{task} Saving SQD data..\n")
        # with open(sqd_init_result_filename, "wb") as f:
        #     pickle.dump(data, f)
        # return
        lb = []
        ub = []

        nomad_params = [
            "BB_OUTPUT_TYPE OBJ",
            "MAX_BB_EVAL 500",
            # "NM_OPTIMIZATION yes",
            # "UPPER_BOUND * 2",
            # "LOWER_BOUND * -2",
            "DISPLAY_DEGREE 2",
            "DISPLAY_ALL_EVAL false",
            # "DISPLAY_STATS BBE OBJNB_THREADS_PARALLEL_EVAL 4",
            "PSD_MADS_OPTIMIZATION True",
            "PSD_MADS_NB_VAR_IN_SUBPROBLEM 20",
            "PSD_MADS_SUBPROBLEM_MAX_BB_EVAL 20",
            "PSD_MADS_NB_SUBPROBLEM 6",
            f"CACHE_FILE {cache_file}",
        ]

        result = PyNomad.optimize(fun, params, lb, ub, nomad_params)

        fmt = ["{} = {}".format(n, v) for (n, v) in result.items() if n != "x_best"]
        output = "\n".join(fmt)
        print("\nNOMAD results \n" + output + " \n")

        t1 = timeit.default_timer()
        logger.info(f"{task} Done optimizing ansatz in {t1 - t0} seconds.\n")

        logger.info(f"{task} Saving data...\n")
        with open(result_filename, "wb") as f:
            pickle.dump(result, f)
    else:
        with open(result_filename, "rb") as f:
            result = pickle.load(f)

    # continue to run sqd
    if not os.path.exists(state_vector_filename):
        logging.info(f"{task} Computing state vector\n")
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
            np.array(result["x_best"]),
            norb=norb,
            n_reps=task.lucj_params.n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=task.lucj_params.with_final_orbital_rotation,
        )
        reference_state = ffsim.hartree_fock_state(norb, nelec)
        final_state = ffsim.apply_unitary(
            reference_state, operator, norb=norb, nelec=nelec
        )
        with open(state_vector_filename, "wb") as f:
            np.save(f, final_state)
    else:
        with open(state_vector_filename, "rb") as f:
            final_state = np.load(f)

    if not os.path.exists(sample_filename):
        logging.info(f"{task} Sampling...\n")
        samples = ffsim.sample_state_vector(
            final_state,
            norb=norb,
            nelec=nelec,
            shots=100_000,
            seed=rng,
            bitstring_type=ffsim.BitstringType.INT,
        )
        bit_array = BitArray.from_samples(samples, num_bits=2 * norb)
        bit_array_count = bit_array.get_int_counts()
        with open(sample_filename, "wb") as f:
            pickle.dump(bit_array_count, f)
    else:
        logging.info(f"{task} load sample...\n")
        with open(sample_filename, "rb") as f:
            bit_array_count = pickle.load(f)
            bit_array = BitArray.from_counts(bit_array_count)

    # Run SQD
    logging.info(f"{task} Running SQD...\n")
    # sci_solver = partial(solve_sci_batch, spin_sq=0.0)

    result_history_energy = []
    result_history_subspace_dim = []
    result_history = []

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
        callback=callback,
        max_dim=task.max_dim,
    )
    logging.info(f"{task} Finish SQD\n")
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
        "history_energy": result_history_energy,
        "history_sci_vec_shape": result_history_subspace_dim,
    }

    logger.info(f"{task} Saving SQD data..\n")
    with open(sqd_result_filename, "wb") as f:
        pickle.dump(data, f)
