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

from lucj.params import LUCJParams, CompressedT2Params
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
import json
from molecules_catalog.util import load_molecular_data
import numpy as np


DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

connectivities = [
    "all-to-all",
    "heavy-hex",
]

n_reps_range = list(range(1, 11))

dmrg_energy = -116.6056091  # ref: https://github.com/jrm874/sqd_data_repository/blob/main/classical_reference_energies/2Fe-2S/classical_methods_energies.txt

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
    if not os.path.exists(filepath):
        result = {
            "energy": 0,
            "history_energy": [0],
            "error": 0,
            "spin_squared": 0,
            "sci_vec_shape": (0, 0),
            "n_reps": 0,
            "history_sci_vec_shape": [0],
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

mol_data = load_molecular_data(
    f"{molecule_basename}",
    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
)
norb = mol_data.norb
nelec = mol_data.nelec

fig, axes = plt.subplots(
    2,
    len(connectivities),
    figsize=(12, 5),
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.07},
)

for col, connectivity in enumerate(connectivities):
    energy_row = 0
    sci_row = 1

    # full
    task_lucj_full = SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
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
    error_avg = np.average(results["history_energy"]) - dmrg_energy
    error_min = np.min(results["history_energy"]) - dmrg_energy
    error_max = np.max(results["history_energy"]) - dmrg_energy

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
        error_min,
        linestyle="--",
        color=colors["lucj_full"],
        alpha=0.7,
    )

    axes[energy_row, col].axhline(
        error_max,
        linestyle="--",
        color=colors["lucj_full"],
        alpha=0.7,
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
        sci_vec_shape_min,
        linestyle="--",
        color=colors["lucj_full"],
        alpha=0.7,
    )

    axes[sci_row, col].axhline(
        sci_vec_shape_max,
        linestyle="--",
        color=colors["lucj_full"],
        alpha=0.7,
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
            bond_distance=None,
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

    tasks_truncated = [
        SQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=None,
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
                error_avg.append(energy_avg - dmrg_energy)
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
    axes[energy_row, col].set_ylabel("Energy error (Hartree)", fontsize=12)
    axes[energy_row, col].set_ylim(1e-1, 10)
    axes[energy_row, col].set_xticks([])
    axes[energy_row, col].set_title(connectivity, fontsize=16)

    axes[sci_row, col].set_ylim(0, 4000)
    axes[sci_row, col].set_ylabel("Subspace dim", fontsize=12)
    axes[sci_row, col].set_xlabel("Repetitions", fontsize=12)

leg = axes[1, 0].legend(
    bbox_to_anchor=(1.05, -0.4),
    loc="upper center",
    ncol=4,
    handletextpad=0.8,
)
leg.set_in_layout(False)

plt.subplots_adjust(
    bottom=0.18,
    top=0.85,
    left=0.07,
    right=0.97,
)

fig.suptitle(f"Fe$_2$S$_2$ ({nelectron}e, {norb}o)", fontsize=18)

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()