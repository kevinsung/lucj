from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 3749045068534c721371f8a0d2d8536d6888413a
from tqdm import tqdm

from lucj.sqd_energy_task.lucj_random_t2_task import (
    RandomSQDEnergyTask,
    run_random_sqd_energy_task,
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

shots = 100_000
<<<<<<< HEAD
samples_per_batch_range = [1000]
=======
samples_per_batch = 1000
>>>>>>> 3749045068534c721371f8a0d2d8536d6888413a
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0
max_dim_range = [500, 1000]


tasks = [
    RandomSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
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
    )
<<<<<<< HEAD
    for samples_per_batch in samples_per_batch_range
=======
>>>>>>> 3749045068534c721371f8a0d2d8536d6888413a
    for max_dim in max_dim_range
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_random_sqd_energy_task(
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
                    run_random_sqd_energy_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
