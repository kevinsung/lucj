from __future__ import annotations
import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from lucj.params import COBYQAParams, LUCJParams
from lucj.tasks.lucj_sqd_quimb_task import (
    LUCJSQDQuimbTask,
    run_lucj_sqd_quimb_task,
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
MAX_PROCESSES = 16
OVERWRITE = False

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance_range = [1.2, 2.4]
n_reps_range = [1, 2, 3, 4, 6]
connectivities = [
    # "square",
    # "all-to-all",
    "heavy-hex"
]
n_reps = 1
shots = 100_000
samples_per_batch = 100
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
cobyqa_maxiter = 50
# TODO set entropy and generate seeds properly
entropy = 0
max_bond: int
max_bonds = [
    10,
    # 25,
    # 50,
    # 100,
    # 200,
    # None,
]
cutoffs = [
    1e-3,
    # 1e-6,
    # 1e-10,
]
seed = 0
perm_mps = False
max_dim_range = [250, 500]
max_dim_range = [500]
# TODO set limit on subspace dimension

tasks = [
    LUCJSQDQuimbTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        cobyqa_params=COBYQAParams(maxiter=cobyqa_maxiter),
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_bond = max_bond,
        perm_mps = perm_mps,
        cutoff = cutoff,
        seed = seed,
        max_dim = max_dim,
    )
    for (connectivity, n_reps, max_bond, cutoff) in itertools.product(
        connectivities, n_reps_range, max_bonds, cutoffs
    )
    for max_dim in max_dim_range
    for d in bond_distance_range
]

if MAX_PROCESSES == 1:
    for task in tqdm(tasks):
        run_lucj_sqd_quimb_task(
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
                    run_lucj_sqd_quimb_task,
                    task,
                    data_dir=DATA_DIR,
                    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
                    overwrite=OVERWRITE,
                )
                future.add_done_callback(lambda _: progress.update())
