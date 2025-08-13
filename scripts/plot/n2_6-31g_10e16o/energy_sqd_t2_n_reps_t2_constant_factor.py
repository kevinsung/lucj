import itertools
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.sqd_energy_task.lucj_compressed_t2_task_sci import SQDEnergyTask
from molecules_catalog.util import load_molecular_data
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

n_reps_range = list(range(1, 11))

shots = 100_000
n_batches = 10
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 1
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0

max_dim = 4000
samples_per_batch = max_dim
constant_factors = [None, 0.5, 1.5, 2, 2.5]

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
    2,
    len(bond_distance_range) * len(connectivities),
    figsize=(10, 5),  # , layout="constrained"
)

for i, (bond_distance, connectivity) in enumerate(itertools.product(bond_distance_range, connectivities)):

    task_lucj_full = SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=None,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=None,
        regularization=False,
        regularization_option=None,
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_dim=max_dim,
    )

    filepath = DATA_ROOT / task_lucj_full.dirpath / "sqd_data.pickle"
    results = load_data(filepath)

    axes[0, i].axhline(
        results['error'],
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )

    axes[1, i].axhline(
        results['sci_vec_shape'][0],
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )
    
    for j, c in enumerate(constant_factors):
        tasks_compressed_t2 = [
            SQDEnergyTask(
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
                ),
                regularization=False,
                regularization_option=None,
                shots=shots,
                samples_per_batch=samples_per_batch,
                n_batches=n_batches,
                energy_tol=energy_tol,
                occupancies_tol=occupancies_tol,
                carryover_threshold=carryover_threshold,
                max_iterations=max_iterations,
                symmetrize_spin=symmetrize_spin,
                entropy=entropy,
                max_dim=max_dim,
                t2_constant_factor=c
            )
            for n_reps in n_reps_range
        ]


        results = {}
        for task in tasks_compressed_t2:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results[task] = load_data(filepath)

        errors = [results[task]["error"] for task in tasks_compressed_t2]
        sci_vec_shape = [results[task]["sci_vec_shape"][0] for task in tasks_compressed_t2]
        
        axes[0, i].plot(
            n_reps_range,
            errors,
            f"{markers[j]}{linestyles[0]}",
            label="factor_{c}",
            color=colors["lucj_compressed"],
            alpha=c / 3 + 0.1 if c is not None else 1
        )

        axes[1, i].plot(
            n_reps_range,
            sci_vec_shape,
            f"{markers[j]}{linestyles[0]}",
            label=f"factor_{c}",
            color=colors["lucj_compressed"],
            alpha=c / 3 + 0.1 if c is not None else 1
        )


    axes[0, i].set_title(f"R={bond_distance} Å / {connectivity}")
    axes[0, i].set_yscale("log")
    axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(n_reps_range)

    axes[1, i].set_ylabel("SCI subspace")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(n_reps_range)

    leg = axes[1, 2].legend(
        bbox_to_anchor=(-0.4, -0.28), loc="upper center", ncol=6
    )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16, top=0.88)

    fig.suptitle(
        f"$N_2$ (6-31g, {nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()