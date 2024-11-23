from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lucj.params import COBYQAParams, LUCJParams
from lucj.tasks.lucj_sqd_cobyqa_task import (
    LUCJSQDCOBYQATask,
    run_lucj_sqd_cobyqa_task,
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
DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
MAX_PROCESSES = 96
OVERWRITE = True

molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

start = 0.9
stop = 2.7
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)

connectivity = "heavy-hex"
n_reps = 1
shots = 100_000
samples_per_batch = 5000
n_batches = 3
max_davidson = 200
# TODO set entropy and generate seeds properly
entropy = None

tasks = [
    LUCJSQDCOBYQATask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        cobyqa_params=COBYQAParams(maxiter=10_000),
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        max_davidson=max_davidson,
        entropy=entropy,
    )
    for d in bond_distance_range
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_sqd_cobyqa_task(
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
                    run_lucj_sqd_cobyqa_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
