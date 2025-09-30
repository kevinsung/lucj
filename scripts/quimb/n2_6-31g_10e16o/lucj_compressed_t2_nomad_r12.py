# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from lucj.params import LUCJParams, CompressedT2Params, COBYQAParams
from lucj.quimb_task.lucj_sqd_quimb_task_sci_nomad import (
    LUCJSQDQuimbTask,
    run_lucj_sqd_quimb_task,
)

filename = (
    f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}_init_sqd_max_bound_50.log"
)
os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=filename,
)

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
# DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = DATA_ROOT
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
MAX_PROCESSES = 2
OVERWRITE = True

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance_range = [1.2, 2.4]
# bond_distance_range = [1.2]

connectivities = [
    # "square",
    # "all-to-all",
    "heavy-hex"
]

# n_reps_range = list(range(1, 11))
n_reps_range = [1]
shots = 10_000
n_batches = 10
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 1
symmetrize_spin = True
cobyqa_maxiter = 0  # 25
# TODO set entropy and generate seeds properly
entropy = 0
max_bond: int
max_bonds = [
    # 5,
    # 10,
    # 25,
    50,
    # 100,
    # 200,
    # None,
]
cutoffs = [
    # 1e-3,
    # 1e-6,
    1e-10,
]
seed = 0
perm_mps = False
max_dim = 4000
samples_per_batch = max_dim

tasks = [
    LUCJSQDQuimbTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True, begin_reps=20, step=2
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
        max_bond=max_bond,
        perm_mps=perm_mps,
        cutoff=cutoff,
        seed=seed,
        max_dim=max_dim,
    )
    for (connectivity, n_reps, max_bond, cutoff) in itertools.product(
        connectivities, n_reps_range, max_bonds, cutoffs
    )
    for d in bond_distance_range
]
if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_sqd_quimb_task(
            task,
            data_dir=DATA_DIR,
            molecules_catalog_dir=MOLECULES_CATALOG_DIR,
            overwrite=OVERWRITE,
        )
else:
    with tqdm(total=len(tasks)) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for task in tasks:
                future = executor.submit(
                    run_lucj_sqd_quimb_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
