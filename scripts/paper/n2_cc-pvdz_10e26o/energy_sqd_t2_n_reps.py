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
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask
from molecules_catalog.util import load_molecular_data
import json
from ffsim.variational.util import interaction_pairs_spin_balanced
import ffsim
import numpy as np
from opt_einsum import contract

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
# TODO set entropy and generate seeds properly
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


def init_loss(n_reps: int, bond_distance: float, connectivity):
    mol_data = load_molecular_data(
        f"{molecule_basename}_d-{bond_distance:.5f}",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )

    # print(mol_data)
    # input()

    norb = mol_data.norb
    nelec = mol_data.nelec
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        connectivity, norb
    )
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        mol_data.ccsd_t2,
        n_reps=n_reps,
        t1=mol_data.ccsd_t1,
        interaction_pairs=(pairs_aa, pairs_ab),
    )
    diag_coulomb_mats = operator.diag_coulomb_mats
    orbital_rotations = operator.orbital_rotations
    t2 = mol_data.ccsd_t2
    nocc, _, _, _ = t2.shape
    diag_coulomb_mats = np.unstack(diag_coulomb_mats, axis=1)[0]
    reconstructed = (
            1j
            * contract(
                "mpq,map,mip,mbq,mjq->ijab",
                diag_coulomb_mats,
                orbital_rotations,
                orbital_rotations.conj(),
                orbital_rotations,
                orbital_rotations.conj(),
                # optimize="greedy"
            )[:nocc, :nocc, nocc:, nocc:]
        )
    diff = reconstructed - t2
    return 0.5 * np.sum(np.abs(diff) ** 2)

print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
linestyles = ["--", ":"]

with open('scripts/paper/color.json', 'r') as file:
    colors = json.load(file)

results_random = {}
for d in bond_distance_range:
    mol_data = load_molecular_data(
        f"{molecule_basename}_d-{d:.5f}",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )

    task = RandomSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        valid_string_only=True,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_dim=max_dim,
    )
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    results_random[d] = load_data(filepath)

fig, axes = plt.subplots(
    3,
    len(bond_distance_range) * len(connectivities),
    figsize=(10, 5),  # , layout="constrained"
)

for i, (bond_distance, connectivity) in enumerate(itertools.product(bond_distance_range, connectivities)):
    mol_data = load_molecular_data(
        f"{molecule_basename}_d-{bond_distance:.5f}",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )
    energy_ground_truth = mol_data.sci_energy
    
    error_avg = np.average(results_random[bond_distance]['history_energy']) - energy_ground_truth
    error_min = np.min(results_random[bond_distance]['history_energy']) - energy_ground_truth
    error_max = np.max(results_random[bond_distance]['history_energy']) - energy_ground_truth

    sci_vec_shape_avg = np.average(results_random[bond_distance]['history_sci_vec_shape'][0]) 
    sci_vec_shape_min = np.min(results_random[bond_distance]['history_sci_vec_shape'][0]) 
    sci_vec_shape_max = np.max(results_random[bond_distance]['history_sci_vec_shape'][0]) 

    axes[0, i].axhline(
        error_avg,
        linestyle="--",
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )

    axes[0, i].axhline(
        error_min,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7
    )

    axes[0, i].axhline(
        error_max,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7
    )

    axes[0, i].axhspan(
        error_min,
        error_max,
        color=colors["random_bit_string"],
        alpha=0.5,
    )

    axes[1, i].axhline(
        sci_vec_shape_avg,
        linestyle="--",
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )

    axes[1, i].axhline(
        sci_vec_shape_min,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7
    )

    axes[1, i].axhline(
        sci_vec_shape_max,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7
    )
    
    axes[1, i].axhspan(
        sci_vec_shape_min,
        sci_vec_shape_max,
        color=colors["random_bit_string"],
        alpha=0.5,
    )

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

    error_avg = np.average(results['history_energy']) - energy_ground_truth
    error_min = np.min(results['history_energy']) - energy_ground_truth
    error_max = np.max(results['history_energy']) - energy_ground_truth

    sci_vec_shape_avg = np.average(results['history_sci_vec_shape'][0]) 
    sci_vec_shape_min = np.min(results['history_sci_vec_shape'][0]) 
    sci_vec_shape_max = np.max(results['history_sci_vec_shape'][0]) 

    # print("lucj full")
    # print(bond_distance)
    # print(connectivity)
    # print(np.average(results['history_energy']))
    # print(error_avg)
    # print(error_min)
    # print(error_max)
    # if connectivity == "all-to-all":
    if True:
        axes[0, i].axhline(
            error_avg,
            linestyle="--",
            label="LUCJ-full",
            color=colors["lucj_full"],
        )

        axes[0, i].axhline(
            error_min,
            linestyle="--",
            # label="LUCJ-full",
            color=colors["lucj_full"],
            alpha=0.7
        )

        axes[0, i].axhline(
            error_max,
            linestyle="--",
            # label="LUCJ-full",
            color=colors["lucj_full"],
            alpha=0.7
        )

        axes[0, i].axhspan(
            error_min,
            error_max,
            color=colors["lucj_full"],
            alpha=0.5,
        )

        axes[1, i].axhline(
            sci_vec_shape_avg,
            linestyle="--",
            label="LUCJ-full",
            color=colors["lucj_full"],
        )

        axes[1, i].axhline(
            sci_vec_shape_min,
            linestyle="--",
            # label="LUCJ-full",
            color=colors["lucj_full"],
            alpha=0.7
        )

        axes[1, i].axhline(
            sci_vec_shape_max,
            linestyle="--",
            # label="LUCJ-full",
            color=colors["lucj_full"],
            alpha=0.7
        )
        
        axes[1, i].axhspan(
            sci_vec_shape_min,
            sci_vec_shape_max,
            color=colors["lucj_full"],
            alpha=0.5,
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
                multi_stage_optimization=True,
                begin_reps=50,
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
    labels = ["LUCJ-truncated", "LUCJ-compressed"]

    for tasks, color_key, label in zip(list_tasks, color_keys, labels):
        error_avg = []
        error_min = []
        error_max = []

        sci_vec_shape_avg = []
        sci_vec_shape_min = []
        sci_vec_shape_max = []
        
        for task in tasks:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results = load_data(filepath)
            
            # energy_avg = np.average(results['history_energy'])
            # error_avg.append(energy_avg - energy_ground_truth)
            # error_min.append(energy_avg - np.min(results['history_energy']))
            # error_max.append(np.max(results['history_energy']) - energy_avg)

            svs_avg = np.average(results['history_sci_vec_shape'][0])
            sci_vec_shape_avg.append(svs_avg)
            sci_vec_shape_min.append(svs_avg - np.min(results['history_sci_vec_shape'][0]))
            sci_vec_shape_max.append(np.max(results['history_sci_vec_shape'][0]) - svs_avg)

            if svs_avg > 0:
                energy_avg = np.average(results['history_energy'])
                error_avg.append(energy_avg - mol_data.sci_energy)
                error_min.append(energy_avg - np.min(results['history_energy']))
                error_max.append(np.max(results['history_energy']) - energy_avg)
                # print(energy_avg)
            else:
                error_avg.append(0)
                error_min.append(0)
                error_max.append(0)
            
        
        axes[0, i].plot(
            n_reps_range,
            error_avg,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )

        axes[0, i].errorbar(
            n_reps_range,
            error_avg,
            [error_min, error_max],
            color=colors[color_key],
        )

        axes[1, i].plot(
            n_reps_range,
            sci_vec_shape_avg,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )

        axes[1, i].errorbar(
            n_reps_range,
            sci_vec_shape_avg,
            [sci_vec_shape_min, sci_vec_shape_max],
            color=colors[color_key],
        )

        list_loss = [[], []]


        for n_reps in n_reps_range:
            list_loss[0].append(init_loss(n_reps, bond_distance, connectivity))

        for task in tasks_compressed_t2:
            filepath = DATA_ROOT / task.operatorpath / "opt_data.pickle"
            results = load_data(filepath)
            list_loss[1].append(results["final_loss"])
        
        color_keys = ["lucj_truncated", "lucj_compressed"]
        labels = ["LUCJ-truncated", "LUCJ-compressed"]
        for loss, color_key, label in zip(list_loss, color_keys, labels):
            axes[2, i].plot(
                n_reps_range,
                loss,
                f"{markers[0]}{linestyles[0]}",
                label=label,
                color=colors[color_key],
            )


    axes[0, i].set_title(f"R={bond_distance} Å / {connectivity}")
    axes[0, i].set_yscale("log")
    axes[0, i].axhline(1.6e-3, linestyle="--", color="black")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(n_reps_range)

    axes[1, i].set_ylabel("SCI subspace")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(n_reps_range)

    axes[2, i].set_ylabel("Operator loss")
    axes[2, i].set_xlabel("Repetitions")
    axes[2, i].set_xticks(n_reps_range)
    axes[2, i].set_yscale("log")

    leg = axes[1, 2].legend(bbox_to_anchor=(-0.24, -2.35), loc="upper center", ncol=5)

    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)

    fig.suptitle(
        f"N$_2$/cc-PVDZ ({nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()