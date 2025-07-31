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
import numpy as np 

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
    2,
    len(bond_distance_range) * len(connectivities),
    figsize=(10, 5),  # , layout="constrained"
)

for i, (bond_distance, connectivity) in enumerate(itertools.product(bond_distance_range, connectivities)):

    error_avg = np.average(results_random[bond_distance]['history_energy']) - mol_data.sci_energy
    error_min = np.min(results_random[bond_distance]['history_energy']) - mol_data.sci_energy
    error_max = np.max(results_random[bond_distance]['history_energy']) - mol_data.sci_energy

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

    error_avg = np.average(results['history_energy']) - mol_data.sci_energy
    error_min = np.min(results['history_energy']) - mol_data.sci_energy
    error_max = np.max(results['history_energy']) - mol_data.sci_energy

    sci_vec_shape_avg = np.average(results['history_sci_vec_shape'][0]) 
    sci_vec_shape_min = np.min(results['history_sci_vec_shape'][0]) 
    sci_vec_shape_max = np.max(results['history_sci_vec_shape'][0]) 

    # print(error_avg)
    # print(error_min)
    # print(error_max)

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
        bbox_to_anchor=(-0.3, -0.28), loc="upper center", ncol=5
    )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)

    fig.suptitle(
        f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()