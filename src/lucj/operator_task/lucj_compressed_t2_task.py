import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ffsim
from ffsim.variational.util import interaction_pairs_spin_balanced
from molecules_catalog.util import load_molecular_data
import pyscf
from lucj.params import LUCJParams, CompressedT2Params

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJCompressedT2Task:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    compressed_t2_params: CompressedT2Params | None
    connectivity_opt: bool = False
    random_op: bool = False
    regularization: bool = False
    regularization_option: int | None = None
    regularization_factor: float | None = None

    @property
    def dirpath(self) -> Path:
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        elif self.compressed_t2_params is not None:
            compress_option = self.compressed_t2_params.dirpath
            if self.regularization:
                if self.regularization_factor is None:
                    compress_option = f"{compress_option}/regularization_{self.regularization_option}"
                else:
                    compress_option = f"{compress_option}/regularization_{self.regularization_option}_{self.regularization_factor:.6f}"
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
    if task.molecule_basename == "fe2s2_30e20o":
        mol_data = load_molecular_data(
            task.molecule_basename,
            molecules_catalog_dir=molecules_catalog_dir,
        )
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
            mol_data.cisd_vec, mol_data.norb, mol_data.nelec[0]
        )
        assert abs(c0) > 1e-8
        t1 = c1 / c0
        t2 = c2 / c0 - np.einsum("ia,jb->ijab", t1, t1)
    else:
        mol_data = load_molecular_data(
            f"{task.molecule_basename}_d-{task.bond_distance:.5f}",
            molecules_catalog_dir=molecules_catalog_dir,
        )
        t2 = mol_data.ccsd_t2
        t1 = mol_data.ccsd_t1
    norb = mol_data.norb

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )

    # use CCSD to initialize parameters
    logging.info(f"{task} Run optimization...\n")
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
                t2,
                n_reps=task.lucj_params.n_reps,
                t1=t1 if task.lucj_params.with_final_orbital_rotation else None,
                interaction_pairs=(pairs_aa, pairs_ab),
                optimize=True,
            )
        else:
            from lucj.operator_task.lucj_compressed_t2_task_ffsim.compressed_t2 import (
                from_t_amplitudes_compressed,
            )
            if task.compressed_t2_params is not None:
                operator, init_loss, final_loss = from_t_amplitudes_compressed(
                    t2,
                    n_reps=task.lucj_params.n_reps,
                    t1=t1 if task.lucj_params.with_final_orbital_rotation else None,
                    interaction_pairs=(pairs_aa, pairs_ab),
                    optimize=True,
                    multi_stage_optimization=task.compressed_t2_params.multi_stage_optimization,
                    regularization=task.regularization,
                    regularization_option=task.regularization_option,
                    regularization_factor=task.regularization_factor,
                    step=task.compressed_t2_params.step,
                    begin_reps=task.compressed_t2_params.begin_reps,
                )
            else:
                operator, init_loss, final_loss = from_t_amplitudes_compressed(
                    t2,
                    n_reps=task.lucj_params.n_reps,
                    t1=t1 if task.lucj_params.with_final_orbital_rotation else None,
                    interaction_pairs=(pairs_aa, pairs_ab),
                    optimize=False,
                )
        data_filename = data_dir / task.dirpath / "opt_data.pickle"
        data = {"init_loss": init_loss, "final_loss": final_loss}

        with open(data_filename, "wb") as f:
            pickle.dump(data, f)

    logging.info(f"{task} Saving data...\n")

    np.savez_compressed(
        operator_filename,
        diag_coulomb_mats=operator.diag_coulomb_mats,
        orbital_rotations=operator.orbital_rotations,
    )

    
