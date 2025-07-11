from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lucj.params import LUCJParams
from lucj.tasks.lucj_compressed_t2_multi_stage_task import (
    LUCJCompressedT2MultiStageTask,
    run_lucj_compressed_t2_multi_stage_task,
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
MAX_PROCESSES = 180
OVERWRITE = True

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

start = 0.9
stop = 2.7
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
bond_distance_range = [0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.7]

connectivities = [
    "heavy-hex",
    "hex",
    "square",
    "all-to-all",
]
n_reps_range = list(range(2, 25, 2)) + [None, 1, 3, 5, 7]

tasks = [
    LUCJCompressedT2MultiStageTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for d in bond_distance_range
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_compressed_t2_multi_stage_task(
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
                    run_lucj_compressed_t2_multi_stage_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
