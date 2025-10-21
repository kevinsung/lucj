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

import ffsim
import matplotlib.pyplot as plt
import numpy as np
from ffsim.variational.util import interaction_pairs_spin_balanced
from molecules_catalog.util import load_molecular_data
from opt_einsum import contract

from lucj.params import CompressedT2Params, LUCJParams
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "cc-pvdz"
nelectron, norb = 10, 26
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
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
entropy = 0

max_dim = 4000
samples_per_batch = max_dim


def load_data(filepath):
    with open(filepath, "rb") as f:
        result = pickle.load(f)
    return result


print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
linestyles = ["--", ":"]

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)

for d in bond_distance_range:
    mol_data = load_molecular_data(
        f"{molecule_basename}_d-{d:.5f}",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )

fig, axes = plt.subplots(
    len(bond_distance_range) * 2,
    2,
    figsize=(12, 12),
    gridspec_kw={"height_ratios": [2, 1, 2, 1], "hspace": 0.12},
)

for bond_idx, bond_distance in enumerate(bond_distance_range):
    for col, connectivity in enumerate(connectivities):
        energy_row = bond_idx * 2
        sci_row = bond_idx * 2 + 1

        mol_data = load_molecular_data(
            f"{molecule_basename}_d-{bond_distance:.5f}",
            molecules_catalog_dir=MOLECULES_CATALOG_DIR,
        )
        energy_ground_truth = mol_data.sci_energy

        # full
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
        error_avg = np.average(results["history_energy"]) - energy_ground_truth
        error_min = np.min(results["history_energy"]) - energy_ground_truth
        error_max = np.max(results["history_energy"]) - energy_ground_truth
        sci_vec_shape_avg = np.average(results["history_sci_vec_shape"][0])
        sci_vec_shape_min = np.min(results["history_sci_vec_shape"][0])
        sci_vec_shape_max = np.max(results["history_sci_vec_shape"][0])
        axes[energy_row, col].axhline(
            error_avg,
            linestyle="--",
            label="full",
            color=colors["lucj_full"],
        )
        axes[energy_row, col].axhline(
            error_min, linestyle="--", color=colors["lucj_full"], alpha=0.7
        )
        axes[energy_row, col].axhline(
            error_max, linestyle="--", color=colors["lucj_full"], alpha=0.7
        )
        axes[energy_row, col].axhspan(
            error_min,
            error_max,
            color=colors["lucj_full"],
            alpha=0.5,
        )
        axes[sci_row, col].axhline(
            sci_vec_shape_avg,
            linestyle="--",
            label="full",
            color=colors["lucj_full"],
        )
        axes[sci_row, col].axhline(
            sci_vec_shape_min, linestyle="--", color=colors["lucj_full"], alpha=0.7
        )
        axes[sci_row, col].axhline(
            sci_vec_shape_max, linestyle="--", color=colors["lucj_full"], alpha=0.7
        )
        axes[sci_row, col].axhspan(
            sci_vec_shape_min,
            sci_vec_shape_max,
            color=colors["lucj_full"],
            alpha=0.5,
        )

        # truncated and compressed
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
                    multi_stage_optimization=True, begin_reps=50, step=2
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
        list_tasks = [tasks_truncated, tasks_compressed_t2]
        color_keys = ["lucj_truncated", "lucj_compressed"]
        labels = ["truncated", "compressed"]
        for tasks, color_key, label, marker in zip(
            list_tasks, color_keys, labels, markers
        ):
            error_avg = []
            error_min = []
            error_max = []
            sci_vec_shape_avg = []
            sci_vec_shape_min = []
            sci_vec_shape_max = []
            for task in tasks:
                filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
                results = load_data(filepath)

                svs_avg = np.average(results["history_sci_vec_shape"][0])
                sci_vec_shape_avg.append(svs_avg)
                sci_vec_shape_min.append(
                    svs_avg - np.min(results["history_sci_vec_shape"][0])
                )
                sci_vec_shape_max.append(
                    np.max(results["history_sci_vec_shape"][0]) - svs_avg
                )
                if svs_avg > 0:
                    energy_avg = np.average(results["history_energy"])
                    error_avg.append(energy_avg - mol_data.sci_energy)
                    error_min.append(energy_avg - np.min(results["history_energy"]))
                    error_max.append(np.max(results["history_energy"]) - energy_avg)
                else:
                    error_avg.append(0)
                    error_min.append(0)
                    error_max.append(0)
            axes[energy_row, col].errorbar(
                n_reps_range,
                error_avg,
                [error_min, error_max],
                fmt=f"{marker}{linestyles[0]}",
                markersize=5,
                capsize=3,
                color=colors[color_key],
                label=label,
            )
            axes[sci_row, col].errorbar(
                n_reps_range,
                sci_vec_shape_avg,
                [sci_vec_shape_min, sci_vec_shape_max],
                fmt=f"{marker}{linestyles[0]}",
                markersize=5,
                capsize=3,
                color=colors[color_key],
                label=label,
            )

        # axis properties
        axes[energy_row, col].set_yscale("log")
        axes[energy_row, col].set_ylabel("Energy error (Hartree)", fontsize=16)
        axes[energy_row, col].set_ylim(1e-3, 1)
        axes[energy_row, col].set_xticks([])

        axes[sci_row, col].set_ylim(0, 2000)
        axes[sci_row, col].set_ylabel("SCI dim sqrt", fontsize=16)
        axes[sci_row, col].set_xlabel("Repetitions", fontsize=16)

        title_map = {"all-to-all": "UCJ", "heavy-hex": "LUCJ heavy-hex"}
        axes[energy_row, col].set_title(
            f"bond length {bond_distance} Å {title_map[connectivity]}", fontsize=20
        )


leg = axes[1, 0].legend(
    bbox_to_anchor=(1.05, -4.75),
    loc="upper center",
    ncol=4,
    handletextpad=0.8,
    fontsize=12,
)
leg.set_in_layout(False)

for row in axes:
    for ax in row:
        ax.tick_params(axis="both", labelsize=13)

# plt.tight_layout()
plt.subplots_adjust(
    bottom=0.2,
    top=0.9,
    left=0.07,
    right=0.97,
    # hspace=0.3,
)

for col in range(2):
    extra_space = 0.09
    pos2 = axes[2, col].get_position()
    pos3 = axes[3, col].get_position()

    axes[2, col].set_position([pos2.x0, pos2.y0 - extra_space, pos2.width, pos2.height])
    axes[3, col].set_position([pos3.x0, pos3.y0 - extra_space, pos3.width, pos3.height])

fig.suptitle(f"N$_2$ / cc-pVDZ ({nelectron}e, {norb}o)", fontsize=24)

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()
