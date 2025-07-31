from __future__ import annotations

import logging
import os
from pathlib import Path

from lucj.params import LUCJParams, CompressedT2Params, COBYQAParams
from lucj.quimb_sqd_task.lucj_sqd_quimb_task import (
    LUCJSQDQuimbTask,
    run_lucj_sqd_quimb_task,
)

filename = f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}.log"
os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=filename,
)

DATA_ROOT = "/media/storage/WanHsuan.Lin/"
# DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = DATA_ROOT 
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
MAX_PROCESSES = 1
OVERWRITE = False

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"

connectivity = "heavy-hex"
n_reps=1
shots = 100_000
n_batches = 10
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 1
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0
max_dim = 4000
samples_per_batch = max_dim
seed = 0

cobyqa_maxiter = 25
max_bond = 50
cutoff = 1e-10

task = LUCJSQDQuimbTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True,
            begin_reps=20,
            step=2
        ),
        regularization=False,
        cobyqa_params=COBYQAParams(maxiter=cobyqa_maxiter),
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_bond = max_bond,
        perm_mps = False,
        cutoff = cutoff,
        seed = seed,
        max_dim = max_dim,
    )

run_lucj_sqd_quimb_task(
    task,
    data_dir=DATA_DIR,
    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    overwrite=OVERWRITE,
)