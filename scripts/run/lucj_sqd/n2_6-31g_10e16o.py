from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lucj.params import LUCJParams
from lucj.tasks.lucj_sqd_task import LUCJSQDTask, run_lucj_sqd_task

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
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

start = 0.9
stop = 2.7
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)

connectivities = [
    "heavy-hex",
]
n_reps_range = [1]
params_and_seed = [
    ("ccsd", None),
    ("optimized-for-qsci", None),
    ("random", 149331520031158709917174116125181872889),
]
shots = 100_000
samples_per_batch_range = [1000, 2000, 5000]
n_batches = 3
max_davidson = 200
# TODO set entropy and generate seeds properly
entropy = None

tasks = [
    LUCJSQDTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        params=params,
        params_seed=params_seed,
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        max_davidson=max_davidson,
        entropy=entropy,
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for d in bond_distance_range
    for params, params_seed in params_and_seed
    for samples_per_batch in samples_per_batch_range
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_sqd_task(
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
                    run_lucj_sqd_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
