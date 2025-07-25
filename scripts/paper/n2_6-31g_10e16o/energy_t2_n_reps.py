import itertools
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.operator_task.lucj_compressed_t2_task import LUCJCompressedT2Task
from lucj.tasks.lucj_initial_params_task import LUCJInitialParamsTask
from lucj.tasks.uccsd_initial_params_task import UCCSDInitialParamsTask

import json

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

bond_distance_range = [1.2, 2.4]

connectivities = [
    "all-to-all",
    "heavy-hex",
]
n_reps_range = list(range(2, 12, 2)) + [None]

shots = 100_000
samples_per_batch_range = [1000]
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0

def load_data(filepath):
    if not os.path.exists(filepath):
        result = {
            "energy": 0,
            "error": 0,
            "spin_squared": 0,
            "sci_vec_shape": (0, 0),
            "n_reps": 0,
        }
    else:
        with open(filepath, "rb") as f:
            result = pickle.load(f)
    return result


print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
linestyles = ["--", ":"]

with open('scripts/paper/color.json', 'r') as file:
    colors = json.load(file)


fig, axes = plt.subplots(
    1,
    len(bond_distance_range) * len(connectivities),
    figsize=(10, 6),  # , layout="constrained"
)
for i, (bond_distance, connectivity) in enumerate(itertools.product(bond_distance_range, connectivities)):
    # UCCSD data
    task_uccsd = UCCSDInitialParamsTask(
        molecule_basename=molecule_basename, bond_distance=bond_distance
    )

    filepath = (
        "lucj/uccsd_initial_params" / task_uccsd.dirpath / "data.pickle"
    )
    data_uccsd = load_data(filepath)

    axes[i].axhline(
        data_uccsd["error"],
        linestyle="--",
        label="UCCSD init",
        color=colors[0],
    )

    # LUCJ data
    tasks_lucj = [
        LUCJInitialParamsTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ))
        for n_reps in n_reps_range
    ]
    data_lucj = {}
    for task in tasks_lucj:
        filepath = (
            "lucj/lucj_initial_params" / task.dirpath / "data.pickle"
        )
        data_lucj[task] = load_data(filepath)

    task_lucj = tasks_lucj[-1]
    assert task_lucj.lucj_params.n_reps is None
    full_n_reps = data_lucj[task_lucj]["n_reps"]
    axes[i].axhline(
        data_lucj[task_lucj]["error"],
        linestyle="--",
        label=f"LUCJ full ({full_n_reps} reps)",
        color=colors[1],
    )

    these_n_reps = [n_reps for n_reps in n_reps_range if n_reps is not None]

    energies = [data_lucj[task]["energy"] for task in tasks_lucj[:-1]]
    errors = [data_lucj[task]["error"] for task in tasks_lucj[:-1]]
    axes[i].plot(
        these_n_reps,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ truncated",
        color=colors[2],
    )

    # compressed_t2
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
                multi_stage_optimization=True, begin_reps=20, step=2
            ),
            regularization=False,
        )
        for n_reps in these_n_reps
    ]

    results_compressed_t2 = {}
    for task in tasks_compressed_t2:
        filepath = DATA_ROOT / task.dirpath / "data.pickle"
        results_compressed_t2[task] = load_data(filepath)

    energies = [
        results_compressed_t2[task]["energy"] for task in tasks_compressed_t2
    ]
    errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]
    axes[i].plot(
        these_n_reps,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ compressed",
        color=colors[5],
    )

    axes[i].set_title(f"R={bond_distance} Å / {connectivity}")
    axes[i].set_yscale("log")
    axes[i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[i].set_ylabel("Energy error (Hartree)")
    axes[i].set_xlabel("Repetitions")
    axes[i].set_xticks(these_n_reps)

    leg = axes[1].legend(
        bbox_to_anchor=(-0.3, -0.25), loc="upper center", ncol=3
    )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    fig.suptitle(
        f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o) / {connectivity}"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_{connectivity}.pdf",
)
plt.savefig(filepath)
plt.close()
