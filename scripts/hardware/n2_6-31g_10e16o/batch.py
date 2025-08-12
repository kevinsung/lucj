from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from lucj.params import LUCJParams, CompressedT2Params
from lucj.hardware_sqd_task.lucj_t2_seperate_sqd_task_sci import (
    HardwareSQDEnergyTask,
    run_hardware_sqd_energy_batch_task,
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
MAX_PROCESSES = 1
OVERWRITE = False

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance_range = [1.2, 2.4]
n_hardware_run = 10
n_reps_range = [1]

shots = 1_000_000
n_batches = 10
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 1
symmetrize_spin = True
entropies = [1]

max_dim = 1000
samples_per_batch = 4000

compressed_tasks = [
    HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True,
            begin_reps=20,
            step=2
        ),
        n_hardware_run=n_hardware_run,
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
        dynamic_decoupling=True,
    )
    for n_reps in n_reps_range
    for d in bond_distance_range
    for entropy in entropies
]

random_tasks = [
    HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=None,
        random_op =True,
        n_hardware_run=n_hardware_run,
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
        dynamic_decoupling=True,
    )
    for n_reps in n_reps_range
    for d in bond_distance_range
    for entropy in entropies
]

truncated_tasks = [
    HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=None,
        random_op =False,
        n_hardware_run=n_hardware_run,
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
        dynamic_decoupling=True,
    )
    for n_reps in n_reps_range
    for d in bond_distance_range
    for entropy in entropies
]

if MAX_PROCESSES == 1:
    for random_task, truncated_task, compressed_task in tqdm(zip(random_tasks, truncated_tasks, compressed_tasks)):
        run_hardware_sqd_energy_batch_task(
            random_task,
            truncated_task,
            compressed_task,
            data_dir=DATA_DIR,
            molecules_catalog_dir=MOLECULES_CATALOG_DIR,
            overwrite=OVERWRITE,
        )
else:
    with tqdm(total=len(random_tasks)) as progress:
        with ProcessPoolExecutor(MAX_PROCESSES) as executor:
            for random_task, truncated_task, compressed_task in zip(random_tasks, truncated_tasks, compressed_tasks):
                future = executor.submit(
                    run_hardware_sqd_energy_batch_task(
                        random_task,
                        truncated_task,
                        compressed_task,
                        data_dir=DATA_DIR,
                        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                        overwrite=OVERWRITE,
                    )
                )
                future.add_done_callback(lambda _: progress.update())

