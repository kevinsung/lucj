import os
import pickle
from pathlib import Path

import numpy as np

from lucj.params import COBYQAParams, LUCJParams
from lucj.tasks.lucj_sqd_cobyqa_task import LUCJSQDCOBYQATask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
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
entropy = 0

tasks = [
    LUCJSQDCOBYQATask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        cobyqa_params=COBYQAParams(maxiter=1000),
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        max_cycle=max_davidson,
        entropy=entropy,
    )
    for d in bond_distance_range
]


for task in tasks:
    destpath = (
        Path(MOLECULES_CATALOG_DIR)
        / "data"
        / "lucj_params"
        / f"{molecule_name}_{basis}_{nelectron}e{norb}o_d-{task.bond_distance:.5f}"
        / task.lucj_params.connectivity
        / f"n_reps-{task.lucj_params.n_reps}"
        / "optimized-for-qsci.npy"
    )
    destpath.parent.mkdir(parents=True, exist_ok=True)
    filepath = DATA_ROOT / "lucj_sqd_cobyqa" / task.dirpath / "result.pickle"
    with open(filepath, "rb") as f:
        result = pickle.load(f)
    with open(destpath, "wb") as f:
        np.save(f, result.x)
