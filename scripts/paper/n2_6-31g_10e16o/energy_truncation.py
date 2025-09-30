# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
import json

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

bond_distance_range = [1.2, 2.4]

n_reps_range = list(range(1, 10, 1)) + list(range(10, 110, 10))

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

bond_distance = 1.2


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

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)

tasks_ucj = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity="all-to-all",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        random_op=False,
        connectivity_opt=False,
        compressed_t2_params=None,
        regularization=False,
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
    for n_reps in n_reps_range + [None]
]

tasks_lucj_square = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity="square",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        random_op=False,
        connectivity_opt=False,
        compressed_t2_params=None,
        regularization=False,
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
    for n_reps in n_reps_range + [None]
]

tasks_lucj_heavy_hex = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        random_op=False,
        connectivity_opt=False,
        compressed_t2_params=None,
        regularization=False,
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
    for n_reps in n_reps_range + [None]
]

list_tasks = [tasks_ucj, tasks_lucj_square, tasks_lucj_heavy_hex]
color_keys = ["ucj", "lucj_square", "lucj_full"]
labels = ["UCJ", "LUCJ:square", "LUCJ:heavy-hex"]

for plot_type in ["vqe", "sqd"]:
    fig = plt.figure(figsize=(5, 4), layout="constrained")
    for tasks, color_key, label in zip(list_tasks, color_keys, labels):
        results = {}
        for task in tasks:
            if plot_type == "vqe":
                filepath = DATA_ROOT / task.operatorpath / "data.pickle"
            else:
                filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results[task] = load_data(filepath)

        energies = [results[task]["energy"] for task in tasks]
        errors = [results[task]["error"] for task in tasks]
        # if label == "UCJ":
        #     print(errors)
        plt.plot(
            n_reps_range + [110],
            errors,
            f"{markers[0]}{linestyles[0]}",
            # linestyles[0],
            label=label,
            color=colors[color_key],
        )

    plt.legend()
    plt.ylabel("Energy error (Hartree)")
    plt.xlabel("Repetitions")
    plt.xticks([1, 5] + list(range(10, 120, 10)))
    plt.tight_layout()
    plt.yscale("log")
    plt.axhline(1.6e-3, linestyle="--", color="black")

    # if plot_type == "vqe":
    #     plt.subplots_adjust(top=0.93, left=0.15)
    #     plt.title(f"N$_2$/6-31G/R=1.2Å ({nelectron}e, {norb}o)")
    # else:
    plt.subplots_adjust(top=0.93)
    plt.title(f"N$_2$/6-31G/R=1.2Å ({nelectron}e, {norb}o)")

    filepath = os.path.join(
        plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_{plot_type}.pdf"
    )
    plt.savefig(filepath)
    print(f"Saved figure to {filepath}.")
    plt.close()
