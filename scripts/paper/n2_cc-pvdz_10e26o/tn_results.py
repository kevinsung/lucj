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

from lucj.params import COBYQAParams, CompressedT2Params, LUCJParams
from lucj.quimb_task.lucj_sqd_quimb_task_nomad import LUCJSQDQuimbTask
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_basename = "n2_cc-pvdz_10e26o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

connectivity = "heavy-hex"
n_reps = 1

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

mol_data = load_molecular_data(
    "n2_cc-pvdz_10e26o_d-1.20000", molecules_catalog_dir=MOLECULES_CATALOG_DIR
)
energy_n2_eq = mol_data.sci_energy

mol_data = load_molecular_data(
    "n2_cc-pvdz_10e26o_d-2.40000", molecules_catalog_dir=MOLECULES_CATALOG_DIR
)
energy_n2_stretched = mol_data.sci_energy

mol_data = load_molecular_data(
    "fe2s2_30e20o", molecules_catalog_dir=MOLECULES_CATALOG_DIR
)
energy_fe2s2 = mol_data.sci_energy

exact_energies = [energy_n2_eq, energy_n2_stretched, energy_fe2s2]

fig, ax = plt.subplots(
    1,
    1,
    # figsize=(12, 12),
)

tasks_n2_eq = [
    SQDEnergyTask(
        molecule_basename="n2_cc-pvdz_10e26o",
        bond_distance=1.2,
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
    ),
    SQDEnergyTask(
        molecule_basename="n2_cc-pvdz_10e26o",
        bond_distance=1.2,
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
    ),
    LUCJSQDQuimbTask(
        molecule_basename="n2_cc-pvdz_10e26o",
        bond_distance=1.2,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=1,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True, begin_reps=50, step=2
        ),
        regularization=False,
        cobyqa_params=COBYQAParams(maxiter=0),
        shots=10_000,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_bond=50,
        perm_mps=False,
        cutoff=1e-10,
        seed=0,
        max_dim=max_dim,
    ),
]
tasks_n2_stretched = [
    SQDEnergyTask(
        molecule_basename="n2_cc-pvdz_10e26o",
        bond_distance=2.4,
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
    ),
    SQDEnergyTask(
        molecule_basename="n2_cc-pvdz_10e26o",
        bond_distance=2.4,
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
    ),
    LUCJSQDQuimbTask(
        molecule_basename="n2_cc-pvdz_10e26o",
        bond_distance=2.4,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=1,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True, begin_reps=50, step=2
        ),
        regularization=False,
        cobyqa_params=COBYQAParams(maxiter=0),
        shots=10_000,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_bond=50,
        perm_mps=False,
        cutoff=1e-10,
        seed=0,
        max_dim=max_dim,
    ),
]
tasks_fe2s2 = [
    SQDEnergyTask(
        molecule_basename="fe2s2_30e20o",
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
    ),
    SQDEnergyTask(
        molecule_basename="fe2s2_30e20o",
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
    ),
    LUCJSQDQuimbTask(
        molecule_basename="fe2s2_30e20o",
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=1,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True, begin_reps=20, step=2
        ),
        regularization=False,
        cobyqa_params=COBYQAParams(maxiter=0),
        shots=10_000,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_bond=50,
        perm_mps=False,
        cutoff=1e-10,
        seed=0,
        max_dim=max_dim,
    ),
]

task_lists = [tasks_n2_eq, tasks_n2_stretched, tasks_fe2s2]
labels = ["truncated", "compressed", "tn-optimized"]
color_keys = ["lucj_truncated", "lucj_compressed", "lucj_compressed_quimb"]
hatches = ["//", r"xx", ""]
bar_width = 0.25  # Width of each bar
x_vals = range(len(task_lists))  # Base positions for each molecule

for i, (tasks, exact_energy) in enumerate(zip(task_lists, exact_energies)):
    for j, (task, color_key, label, marker, hatch) in enumerate(
        zip(tasks, color_keys, labels, markers, hatches)
    ):
        filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
        results = load_data(filepath)

        svs_avg = np.average(results["history_sci_vec_shape"][0])
        sci_vec_shape_avg = svs_avg
        sci_vec_shape_min = svs_avg - np.min(results["history_sci_vec_shape"][0])
        sci_vec_shape_max = np.max(results["history_sci_vec_shape"][0]) - svs_avg

        energy_avg = np.average(results["history_energy"])
        error_avg = energy_avg - exact_energy
        error_min = energy_avg - np.min(results["history_energy"])
        error_max = np.max(results["history_energy"]) - energy_avg

        x_pos = i + (j - 1) * bar_width
        ax.bar(
            x_pos,
            error_avg,
            width=bar_width,
            hatch=hatch,
            edgecolor="#343a3f",
            # edgecolor="black",
            label=label if i == 0 else None,
            color=colors[color_key],
            # linewidth=0.5,
        )
        ax.errorbar(
            x_pos,
            error_avg,
            [[error_min], [error_max]],
            color="black",
            capsize=3,
            fmt="none",
        )

ax.set_yscale("log")
ax.set_xticks(x_vals)
ax.set_xticklabels([r"N$_2$/cc-pVDZ 1.2Å", r"N$_2$/cc-pVDZ 2.4Å", "[2Fe-2S]"])
ax.set_ylabel("Energy error (Hartree)", fontsize=16)
ax.tick_params(axis="both", labelsize=13)
ax.legend(fontsize=12)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
plt.tight_layout()

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_maxdim-{max_dim}_shot-{shots}.pdf",
)
plt.savefig(filepath)
plt.close()
