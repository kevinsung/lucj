# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import CompressedT2Params, LUCJParams
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

bond_distance = 1.2

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


def load_data(filepath):
    with open(filepath, "rb") as f:
        result = pickle.load(f)
    return result


print("Done loading data.")

linestyles = ["-.", ":", "--"]
markers = ["o", "s", "v", "D", "p", "*", "P", "X"]

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)


fig, axes = plt.subplots(3, 2, figsize=(12, 12))


for i, connectivity in enumerate(connectivities):
    title_map = {"all-to-all": "UCJ", "heavy-hex": "LUCJ heavy-hex"}
    axes[0, i].set_title(title_map[connectivity], fontsize=22)

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
    sci_dim_lucj_full = results["sci_vec_shape"][0]
    sqd_error_lucj_full = results["error"]
    filepath = DATA_ROOT / task_lucj_full.operatorpath / "data.pickle"
    results = load_data(filepath)
    entropy_lucj_full = results["entropy"]

    axes[0, i].axhline(
        entropy_lucj_full,
        linestyle="--",
        label="full",
        color=colors["lucj_full"],
    )
    axes[1, i].axhline(
        sci_dim_lucj_full,
        linestyle="--",
        label="full",
        color=colors["lucj_full"],
    )
    axes[2, i].axhline(
        sqd_error_lucj_full,
        linestyle="--",
        label="full",
        color=colors["lucj_full"],
    )

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
                multi_stage_optimization=True, begin_reps=20, step=2
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
        )
        for n_reps in n_reps_range
    ]

    tasks_compressed_t2_reg = [
        SQDEnergyTask(
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
            regularization=True,
            regularization_option=1,
            regularization_factor=5e-3,
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
        for n_reps in n_reps_range
    ]

    tasks_truncated = [
        SQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            random_op=False,
            compressed_t2_params=None,
            connectivity_opt=False,
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
        for n_reps in n_reps_range
    ]

    list_tasks = [
        tasks_truncated,
        tasks_compressed_t2,
        tasks_compressed_t2_reg,
    ]
    color_keys = ["lucj_truncated", "lucj_compressed", "lucj_compressed_reg"]
    labels = ["truncated", "compressed", "compressed with regularization 5e-3"]

    for tasks, color_key, label, marker in zip(list_tasks, color_keys, labels, markers):
        results = {}
        results_op = {}
        for task in tasks:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results[task] = load_data(filepath)
            filepath = DATA_ROOT / task.operatorpath / "data.pickle"
            results_op[task] = load_data(filepath)
        sci_vec_shape = [results[task]["sci_vec_shape"][0] for task in tasks]
        errors = [results[task]["error"] for task in tasks]
        entropies = [results_op[task]["entropy"] for task in tasks]
        axes[0, i].plot(
            n_reps_range,
            entropies,
            f"{marker}--",
            markersize=5,
            label=label,
            color=colors[color_key],
        )

        axes[1, i].plot(
            n_reps_range,
            sci_vec_shape,
            f"{marker}--",
            markersize=5,
            label=label,
            color=colors[color_key],
        )

        axes[2, i].plot(
            n_reps_range,
            errors,
            f"{marker}--",
            markersize=5,
            label=label,
            color=colors[color_key],
        )

    axes[0, i].set_ylabel("Entropy", fontsize=16)
    axes[0, i].set_xlabel("Repetitions", fontsize=16)
    axes[0, i].set_yscale("log")
    # axes[0, i].set_ylim(1e-2, 10)
    # axes[0, i].set_ylim(0, 8)

    axes[1, i].set_ylabel("SCI dim sqrt", fontsize=16)
    axes[1, i].set_xlabel("Repetitions", fontsize=16)
    axes[1, i].set_ylim(0, 1000)

    axes[2, i].set_ylabel("Energy error (Hartree)", fontsize=16)
    axes[2, i].set_xlabel("Repetitions", fontsize=16)
    axes[2, i].set_yscale("log")
    axes[2, i].set_ylim(1e-3, 1)


fig.suptitle(
    f"N$_2$ / 6-31G ({nelectron}e, {norb}o) bond length {bond_distance} Å", fontsize=24
)

# axes[2, 0].legend()

leg = axes[2, 0].legend(
    bbox_to_anchor=(1, -0.25),
    loc="upper center",
    ncol=4,
    # columnspacing=1,
    handletextpad=0.8,
    fontsize=12,
)
leg.set_in_layout(False)

for row in axes:
    for ax in row:
        ax.tick_params(axis="both", labelsize=13)

plt.tight_layout()
plt.subplots_adjust(
    top=0.91,
    bottom=0.1,
    hspace=0.3,
)

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()
