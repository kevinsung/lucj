import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import ffsim
from molecules_catalog.util import load_molecular_data, load_ucj_op_spin_balanced
from tenpy.algorithms.tebd import TEBDEngine

from lucj.params import LUCJParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJMPSTenpyTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    chi_max: float
    svd_min: float
    params: str
    seed: int | None = None

    @property
    def dirpath(self) -> Path:
        path = (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.2f}"
            )
            / self.lucj_params.dirpath
            / f"chi_max-{self.chi_max}"
            / f"svd_min-{self.svd_min}"
            / f"params-{self.params}"
        )
        if self.params == "random":
            path /= f"seed-{self.seed}"
        return path


def run_lucj_mps_tenpy_task(
    task: LUCJMPSTenpyTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJMPSTenpyTask:
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
        molecule_basename,
        molecules_catalog_dir=molecules_catalog_dir,
    )
    norb = mol_data.norb
    nelec = mol_data.nelec
    n_alpha, n_beta = nelec
    mpo_model = ffsim.tenpy.MolecularHamiltonianMPOModel.from_molecular_hamiltonian(
        mol_data.hamiltonian
    )
    mpo = mpo_model.H_MPO

    # Get LUCJ operator
    ucj_op = load_ucj_op_spin_balanced(
        molecule_basename=molecule_basename,
        connectivity=task.lucj_params.connectivity,
        n_reps=task.lucj_params.n_reps,
        params=task.params,
        seed=task.seed,
    )

    # Construct Hartree-Fock state
    psi_mps = ffsim.tenpy.bitstring_to_mps(
        ((1 << n_alpha) - 1, (1 << n_beta) - 1), norb
    )

    # Construct the TEBD engine
    options = {"trunc_params": {"chi_max": task.chi_max, "svd_min": task.svd_min}}
    eng = TEBDEngine(psi_mps, None, options)

    # Apply the LUCJ operator
    ffsim.tenpy.apply_ucj_op_spin_balanced(eng, ucj_op)

    # Compute the expectation value
    energy = mpo.expectation_value_finite(psi_mps)
    error = energy - mol_data.fci_energy

    # Save data
    data = {
        "energy": energy,
        "error": error,
        "n_reps": ucj_op.n_reps,
    }

    logging.info(f"{task} Saving data...\n")
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
