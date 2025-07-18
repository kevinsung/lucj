import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.tasks.lucj_initial_params_task import LUCJInitialParamsTask
from lucj.operator_task.lucj_compressed_t2_task import (
    LUCJCompressedT2Task,
)

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

bond_distance_range = [1.0] #, 2.4]

connectivity = "all-to-all"
n_reps_range = list(range(4, 14, 2))

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, axes = plt.subplots(1, len(bond_distance_range), figsize=(6, 3), layout="constrained")

for i, bond_distance in enumerate(bond_distance_range):

    tasks_lucj = [
        LUCJInitialParamsTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
        )
        for n_reps in n_reps_range
    ]

    data_lucj = {}
    for task in tasks_lucj:
        filepath = DATA_ROOT / "lucj_initial_params" / task.dirpath / "data.pickle"
        with open(filepath, "rb") as f:
            data_lucj[task] = pickle.load(f)

    errors = [data_lucj[task]["error"] for task in tasks_lucj]
    axes.plot(
        n_reps_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ truncated",
        color=colors[2],
    )

    tasks_compressed_t2 = [
        LUCJCompressedT2Task(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=CompressedT2Params(
                multi_stage_optimization=True,
                begin_reps=20,
                step=2
            )
        )
        for n_reps in n_reps_range
    ]
    
    results_compressed_t2 = {}
    for task in tasks_compressed_t2:
        filepath = DATA_ROOT / task.dirpath / "data.pickle"
        with open(filepath, "rb") as f:
            result = pickle.load(f)
            results_compressed_t2[task] = result
            
    errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]

    axes.plot(
        n_reps_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ Compressed-t2",
        color=colors[5],
    )

    results_compressed_t2 = {}
    for task in tasks_compressed_t2:
        filepath = DATA_ROOT / task.dirpath / "asc_data.pickle"
        with open(filepath, "rb") as f:
            result = pickle.load(f)
            results_compressed_t2[task] = result
            
    errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]

    axes.plot(
        n_reps_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ Compressed-t2-asc",
        color=colors[3],
    )

    results_compressed_t2 = {}
    for task in tasks_compressed_t2:
        filepath = DATA_ROOT  / task.dirpath / "des_data.pickle"
        with open(filepath, "rb") as f:
            result = pickle.load(f)
            results_compressed_t2[task] = result
            
    errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]

    axes.plot(
        n_reps_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ Compressed-t2-des",
        color=colors[4],
    )


    results_compressed_t2 = {}
    for task in tasks_compressed_t2:
        filepath = DATA_ROOT  / task.dirpath / "rand_data.pickle"
        with open(filepath, "rb") as f:
            result = pickle.load(f)
            results_compressed_t2[task] = result
            
    errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]

    axes.plot(
        n_reps_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ Compressed-t2-rand",
        color=colors[6],
    )
    axes.set_title(f"{bond_distance} Å")
    axes.set_yscale("log")
    axes.axhline(1.6e-3, linestyle="--", color="gray")
    axes.set_ylabel("Energy error (Hartree)")
    axes.set_xlabel("Repetitions")
    axes.set_xticks(n_reps_range)
    axes.legend()

    fig.suptitle(
        f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o)"
    )


filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf"
)
plt.savefig(filepath)
plt.close()
