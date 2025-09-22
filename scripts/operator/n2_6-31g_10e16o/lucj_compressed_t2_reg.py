from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lucj.operator_task.lucj_compressed_t2_task import (
    LUCJCompressedT2Task,
    run_lucj_compressed_t2_task,
)
from lucj.params import CompressedT2Params, LUCJParams

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
MAX_PROCESSES = 60
OVERWRITE = True

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance_range = [1.2, 2.4]

connectivities = [
    "heavy-hex",
    # "square",
    "all-to-all",
]
n_reps_range = list(range(1, 11, 1))
regularization_options = [1]
regularization_factors = [1e-4, 1e-3, 1e-2, 1e-1]

tasks = [
    LUCJCompressedT2Task(
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
        regularization=True,
        regularization_option=regularization_option,
        regularization_factor=regularization_factor,
    )
    for regularization_option in regularization_options
    for regularization_factor in regularization_factors
    for d in bond_distance_range
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_compressed_t2_task(
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
                    run_lucj_compressed_t2_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
