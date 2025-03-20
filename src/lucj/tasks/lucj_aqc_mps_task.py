import logging
import os
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import ffsim
import numpy as np
import quimb.tensor
import scipy.optimize
import scipy.stats
from molecules_catalog.util import load_lucj_circuit, load_molecular_data
from qiskit_addon_aqc_tensor.ansatz_generation import parametrize_circuit
from qiskit_addon_aqc_tensor.objective import OneMinusFidelity
from qiskit_addon_aqc_tensor.simulation import (
    compute_overlap,
    tensornetwork_from_circuit,
)
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator

from lucj.params import LUCJParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJAQCMPSTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    init_params: str = "ccsd"
    max_bond: int | None = None
    cutoff: float = 1e-10

    @property
    def dirpath(self) -> Path:
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.2f}"
            )
            / self.lucj_params.dirpath
            / f"init_params-{self.init_params}"
            / f"max_bond-{self.max_bond}"
            / f"cutoff-{self.cutoff}"
        )


def run_lucj_aqc_mps_task(
    task: LUCJAQCMPSTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJAQCMPSTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    data_filename = data_dir / task.dirpath / "data.pickle"
    if (not overwrite) and os.path.exists(data_filename):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian
    molecule_basename = task.molecule_basename
    if task.bond_distance is not None:
        molecule_basename += f"_d-{task.bond_distance:.2f}"
    mol_data = load_molecular_data(
        molecule_basename=molecule_basename,
        molecules_catalog_dir=molecules_catalog_dir,
    )
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian
    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    # Target circuit for AQC
    target_circuit = load_lucj_circuit(
        molecule_basename=molecule_basename,
        connectivity="all-to-all",
        n_reps=None,
        params=task.init_params,
        molecules_catalog_dir=molecules_catalog_dir,
    )
    target_circuit = ffsim.qiskit.PRE_INIT.run(target_circuit).decompose()

    # Ansatz circuit
    ansatz_circuit = load_lucj_circuit(
        molecule_basename=molecule_basename,
        connectivity=task.lucj_params.connectivity,
        n_reps=task.lucj_params.n_reps,
        params=task.init_params,
        molecules_catalog_dir=molecules_catalog_dir,
    )
    ansatz_circuit = ffsim.qiskit.PRE_INIT.run(ansatz_circuit).decompose()

    # AQC ansatz
    aqc_ansatz, aqc_initial_parameters = parametrize_circuit(ansatz_circuit)

    # Simulator settings
    simulator_settings = QuimbSimulator(
        partial(quimb.tensor.CircuitMPS, max_bond=task.max_bond, cutoff=task.cutoff),
        autodiff_backend="jax",
    )

    # Target MPS
    logging.info(f"{task} Constructing target MPS...\n")
    aqc_target_mps = tensornetwork_from_circuit(target_circuit, simulator_settings)
    logger.info(
        f"{task} Target MPS maximum bond dimension: {aqc_target_mps.psi.max_bond()}"
    )

    # Optimize
    logging.info(f"{task} Optimizing ansatz MPS...\n")
    objective = OneMinusFidelity(aqc_target_mps, aqc_ansatz, simulator_settings)
    result = scipy.optimize.minimize(
        objective,
        aqc_initial_parameters,
        method="L-BFGS-B",
        jac=True,
        # options=dict(maxiter=100),
    )

    # Compute optimized state vector
    logging.info(f"{task} Computing optimized state vector...\n")
    final_circuit = aqc_ansatz.assign_parameters(result.x)
    final_vec = ffsim.qiskit.final_state_vector(
        final_circuit, norb=norb, nelec=nelec
    ).vec

    # Log initial and final fidelities
    initial_mps = tensornetwork_from_circuit(ansatz_circuit, simulator_settings)
    initial_fidelity = abs(compute_overlap(initial_mps, aqc_target_mps)) ** 2
    final_mps = tensornetwork_from_circuit(final_circuit, simulator_settings)
    final_fidelity = abs(compute_overlap(final_mps, aqc_target_mps)) ** 2
    logger.info(
        f"{task} Initial fidelity: {initial_fidelity}; Final fidelity: {final_fidelity}"
    )

    # Compute energy and other properties of final state vector
    logging.info(f"{task} Computing energy and other properties...\n")
    energy = np.vdot(final_vec, linop @ final_vec).real
    error = energy - mol_data.fci_energy
    spin_squared = ffsim.spin_square(
        final_vec, norb=mol_data.norb, nelec=mol_data.nelec
    )
    probs = np.abs(final_vec) ** 2
    entropy = scipy.stats.entropy(probs)
    # TODO don't need to compute norm anymore since the circuit preserves symmetries
    norm = np.linalg.norm(final_vec)
    logging.info(f"{task} State vector norm: {norm}\n")

    data = {
        "energy": energy,
        "error": error,
        "spin_squared": spin_squared,
        "entropy": entropy,
        "norm": norm,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
