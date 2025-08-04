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
from ffsim.variational.util import (
    interaction_pairs_spin_balanced,
    orbital_rotation_to_parameters,
)
from molecules_catalog.util import load_molecular_data
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
from lucj.tasks.lucj_compressed_t2_task_ffsim.compressed_t2 import (
    from_t_amplitudes_compressed,
)
from lucj.params import COBYQAParams, LUCJParams, CompressedT2Params

from qiskit.circuit import QuantumCircuit, QuantumRegister
import quimb.tensor
import quimb.tensor.belief_propagation
from qiskit_quimb import quimb_gates

import pyscf
import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJSQDQuimbPEPSTask:
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
            / "peps"
            / f"max_dim-{self.max_dim}"
            / f"max_bond-{self.max_bond}"
            / f"cutoff-{self.cutoff}"
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


def load_operator(task: LUCJSQDQuimbPEPSTask, data_dir: str, mol_data):
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


def run_lucj_sqd_quimb_task(
    task: LUCJSQDQuimbPEPSTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    bootstrap_task: LUCJSQDQuimbPEPSTask | None = None,
    bootstrap_data_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJSQDQuimbPEPSTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filename = data_dir / task.dirpath / "data.pickle"
    info_filename = data_dir / task.dirpath / "info.pickle"
    if (
        (not overwrite)
        and os.path.exists(result_filename)
        and os.path.exists(info_filename)
    ):
        logger.info(f"Data for {task} already exists. Skipping...\n")
        return task
    intermediate_result_filename = data_dir / task.dirpath / "intermediate_data.pickle"

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

    def fun(x: np.ndarray) -> float:
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

        # vacuum state
        peps = quimb.tensor.TN_from_sites_product_state(
            {q: [1.0, 0.0] for q in range(circuit.num_qubits)}
        )  # type: TensorNetworkGenVector

        # create size 1 bonds
        for i, j in pairs_aa:
            peps[i].new_bond(peps[j])
            peps[i + norb].new_bond(peps[j + norb])
        for i, j in pairs_ab:
            peps[i].new_bond(peps[j + norb])

        gauges = {}

        for gate in quimb_gates(decomposed):
            peps.gate_simple_(
                gate.array,
                gate.qubits,
                gauges=gauges,
                # this is a crucial paramter - the maximum bond dimension
                max_bond=task.max_bond,
                cutoff=task.cutoff,
                cutoff_mode="rel",
                renorm=False,
            )

        # the state normalization gives us an idea of fidelity
        logger.info(f"{task}\n\tPeps fidelity {peps.normalize_simple(gauges)}")

        # equilibrate gauges
        peps.gauge_all_simple_(100, tol=1e-9, gauges=gauges, progbar=True)

        logger.info(f"{task}\n\tSampling circuit...")
        t0 = timeit.default_timer()
        samples = []
        for _ in range(task.shots):
            # config, tn_config, omega = quimb.tensor.belief_propagation.sample_d2bp(peps, seed=task.seed, progbar=False)
            config, tn_config, omega = quimb.tensor.belief_propagation.sample_hd1bp(peps, seed=task.seed, progbar=False)
            # config, omega = peps.sample_configuration_cluster(
            #     gauges=gauges,
            #     seed=task.seed,
            #     # single site clusters
            #     max_distance=0,
            # )
            x = "".join(map(str, (config[f"k{i}"] for i in range(circuit.num_qubits))))
            samples.append(x)
        t1 = timeit.default_timer()
        time = t1 - t0
        logger.info(f"{task}\n\tDone sampling circuit in {time} seconds.")
        samples = [sample[::-1] for sample in samples]

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
            max_dim=task.max_dim,
        )
        return result.energy + mol_data.core_energy

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
        if info["nit"] > 3:
            if (abs(info["fun"][-1] - info["fun"][-2]) < 1e-5) and (
                abs(info["fun"][-2] - info["fun"][-3]) < 1e-5
            ):
                raise StopIteration(
                    "Objective function value does not decrease for two iterations."
                )

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
