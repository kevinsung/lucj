from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from lucj.params import LUCJParams
from lucj.tasks.lucj_mps_tenpy_task import LUCJMPSTenpyTask, run_lucj_mps_tenpy_task

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

molecule_name = "h20"
basis = "sto-6g"
nelectron, norb = 20, 20
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bohr = 0.529177
bond_distance_range = [1 * bohr, 2 * bohr, 3 * bohr]

connectivities = [
    "square",
]
n_reps_range = [None]
chi_max_range = [100, 250, 500, 1000, 2500, 5000]
svd_min_range = [1e-8]

tasks = [
    LUCJMPSTenpyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        chi_max=chi_max,
        svd_min=svd_min,
        params="ccsd",
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for chi_max in chi_max_range
    for svd_min in svd_min_range
    for d in bond_distance_range
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_mps_tenpy_task(
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
                    run_lucj_mps_tenpy_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
