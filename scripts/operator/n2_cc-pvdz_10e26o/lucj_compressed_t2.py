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

import numpy as np
from tqdm import tqdm

from lucj.params import LUCJParams, CompressedT2Params
from lucj.operator_task.lucj_compressed_t2_task import (
    LUCJCompressedT2Task,
    run_lucj_compressed_t2_task,
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
MAX_PROCESSES = 40
OVERWRITE = True

molecule_name = "n2"
basis = "cc-pvdz"
nelectron, norb = 10, 26
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance_range = [1.2, 2.4]

connectivities = [
    "heavy-hex",
    # "square",
    "all-to-all",
]
n_reps_range = list(range(1, 11, 1))

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
            multi_stage_optimization=True,
            begin_reps=50, #40,
            step=2, #4
        ),
        regularization=False,
        regularization_option=None,
        regularization_factor=None,
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for d in bond_distance_range
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
