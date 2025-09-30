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
from lucj.quimb_task.lucj_sqd_quimb_task_sci import LUCJSQDQuimbTask
from lucj.hardware_sqd_task.lucj_compressed_t2_quimb_task_sci import (
    HardwareSQDQuimbEnergyTask,
    run_hardware_sqd_energy_task,
)

filename = f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}.log"
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
MAX_PROCESSES = 10
OVERWRITE = False

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

# bond_distance_range = [1.2, 2.4]
bond_distance_range = [2.4]

n_reps_range = [1]

shots = 1_000_000
n_batches = 10
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 1
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropies = [1]
n_hardware_run_ranges = list(range(10))


max_dim = 1000
samples_per_batch = max_dim

tasks = [
    HardwareSQDQuimbEnergyTask(
        lucj_sqd_quimb_task=LUCJSQDQuimbTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=CompressedT2Params(
                multi_stage_optimization=True, begin_reps=20, step=2
            ),
            regularization=False,
            cobyqa_params=COBYQAParams(maxiter=0),
            shots=10_000,
            samples_per_batch=4000,
            n_batches=n_batches,
            energy_tol=1e-5,
            occupancies_tol=1e-3,
            carryover_threshold=1e-3,
            max_iterations=1,
            symmetrize_spin=symmetrize_spin,
            entropy=0,
            max_bond=50,
            perm_mps=False,
            cutoff=1e-10,
            seed=0,
            max_dim=4000,
        ),
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_dim=max_dim,
        n_hardware_run=n_hardware_run,
        dynamic_decoupling=True,
    )
    for n_reps in n_reps_range
    for d in bond_distance_range
    for entropy in entropies
    for n_hardware_run in n_hardware_run_ranges
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_hardware_sqd_energy_task(
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
                    run_hardware_sqd_energy_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())

# /media/storage/WanHsuan.Lin/n2_6-31g_10e16o/bond_distance-2.40000/connectivity-heavy-hex/n_reps-1/with_final_orbital_rotation-True/multi_stage_optimization-True/begin_reps-20/step-2/quimb/maxiter-0/shots-10000/samples_per_batch-4000/n_batches-10/energy_tol-1e-05/occupancies_tol-0.001/carryover_threshold-0.001/max_iterations-1/symmetrize_spin-True/entropy-0/max_dim-4000/max_bond-200/cutoff-1e-10/perm_mps-False/seed-0/data.pickle
