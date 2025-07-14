import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ffsim
from ffsim.variational.util import interaction_pairs_spin_balanced
from molecules_catalog.util import load_molecular_data

from lucj.params import LUCJParams, CompressedT2Params

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJCompressedT2Task:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    compressed_t2_params: CompressedT2Params | None
    connectivity_opt: bool | None = False
    random_op: bool = False

    @property
    def dirpath(self) -> Path:
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        else:
            compress_option = self.compressed_t2_params.dirpath
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


def run_lucj_compressed_t2_task(
    task: LUCJCompressedT2Task,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    overwrite: bool = True,
) -> LUCJCompressedT2Task:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath , exist_ok=True)

    operator_filename = data_dir / task.dirpath / "operator.npz"
    if (
        (not overwrite)
        and os.path.exists(operator_filename)
    ):
        logging.info(f"Data for {task} already exists. Skipping...\n")
        return task

    # Get molecular data and molecular Hamiltonian
    mol_data = load_molecular_data(
        f"{task.molecule_basename}_d-{task.bond_distance:.5f}",
        molecules_catalog_dir=molecules_catalog_dir,
    )
    norb = mol_data.norb

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )

    # use CCSD to initialize parameters
    nocc, _, nvrt, _ = mol_data.ccsd_t2.shape
    
    if task.random_op:
        operator = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=task.lucj_params.n_reps,
        interaction_pairs=(pairs_aa, pairs_ab),
        with_final_orbital_rotation=True
    )
    else:
        if task.connectivity_opt:
            from lucj.operator_task.lucj_compressed_t2_task_ffsim.compressed_t2_connectivity import (
                from_t_amplitudes_compressed,
            )
            operator, init_loss, final_loss = from_t_amplitudes_compressed(
                mol_data.ccsd_t2,
                n_reps=task.lucj_params.n_reps,
                t1=mol_data.ccsd_t1 if task.lucj_params.with_final_orbital_rotation else None,
                interaction_pairs=(pairs_aa, pairs_ab),
                optimize=True,
            )
        else:
            from lucj.operator_task.lucj_compressed_t2_task_ffsim.compressed_t2 import (
                from_t_amplitudes_compressed,
            )
            operator, init_loss, final_loss = from_t_amplitudes_compressed(
                mol_data.ccsd_t2,
                n_reps=task.lucj_params.n_reps,
                t1=mol_data.ccsd_t1 if task.lucj_params.with_final_orbital_rotation else None,
                interaction_pairs=(pairs_aa, pairs_ab),
                optimize=True,
                multi_stage_optimization=task.compressed_t2_params.multi_stage_optimization,
                step=task.compressed_t2_params.step,
                begin_reps=task.compressed_t2_params.begin_reps,
            )
        data_filename = data_dir / task.dirpath / "opt_data.pickle"
        data = {"init_loss": init_loss, "final_loss": final_loss}

        with open(data_filename, "wb") as f:
            pickle.dump(data, f)

    logging.info(f"{task} Saving data...\n")

    np.savez(
        operator_filename,
        diag_coulomb_mats=operator.diag_coulomb_mats,
        orbital_rotations=operator.orbital_rotations,
    )

    
