import itertools
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params, COBYQAParams
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask
from lucj.quimb_task.lucj_sqd_quimb_task import LUCJSQDQuimbTask
from molecules_catalog.util import load_molecular_data
import json


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
            "error": 0,
            "spin_squared": 0,
            "sci_vec_shape": (0, 0),
            "n_reps": 0,
            "nit": 0
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


task = RandomSQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=None,
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
results_random = load_data(filepath)


fig, axes = plt.subplots(
    3,
    len(connectivities),
    figsize=(8, 5),  # , layout="constrained"
)

for i, connectivity in enumerate(connectivities):

    axes[0, i].axhline(
        results_random['error'],
        linestyle="--",
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )

    axes[2, i].axhline(
        results_random['sci_vec_shape'][0],
        linestyle="--",
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )


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

    axes[0, i].axhline(
        results['error'],
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )

    axes[2, i].axhline(
        results['sci_vec_shape'][0],
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )
    

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

    if connectivity == "heavy-hex":
        tasks_compressed_t2_quimb = [
            LUCJSQDQuimbTask(
                molecule_basename=molecule_basename,
                bond_distance=None,
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
                cobyqa_params=COBYQAParams(maxiter=25),
                shots=10_000,
                samples_per_batch=samples_per_batch,
                n_batches=n_batches,
                energy_tol=energy_tol,
                occupancies_tol=occupancies_tol,
                carryover_threshold=carryover_threshold,
                max_iterations=max_iterations,
                symmetrize_spin=symmetrize_spin,
                entropy=entropy,
                max_bond = 100,
                perm_mps = False,
                cutoff = 1e-10,
                seed = 0,
                max_dim = max_dim,
            )
            for n_reps in n_reps_range]


        # list_tasks = [tasks_truncated, tasks_compressed_t2, tasks_compressed_t2_naive, tasks_compressed_t2_quimb, tasks_compressed_t2_quimb_do]
        list_tasks = [tasks_truncated, tasks_compressed_t2, tasks_compressed_t2_quimb]
        color_keys = ["lucj_truncated", "lucj_compressed", "lucj_compressed_quimb"]
        labels = ["LUCJ-truncated", "LUCJ-compressed", "lucj-compressed-tn"]
        nit = []
        for task in tasks_compressed_t2_quimb:
            filepath = DATA_ROOT / task.dirpath / "info.pickle"
            results = load_data(filepath)
            nit.append(results['nit'])
        
        axes[1, i].plot(
            n_reps_range,
            nit,
            f"{markers[0]}{linestyles[0]}",
            label="lucj-compressed-tn",
            color=colors["lucj_compressed_quimb"],
        )

        # nit = []
        # for task in tasks_compressed_t2_quimb_do:
        #     filepath = DATA_ROOT / task.dirpath / "info.pickle"
        #     results = load_data(filepath)
        #     nit.append(results['nit'])
        
        # axes[1, i].plot(
        #     n_reps_range,
        #     nit,
        #     f"{markers[0]}{linestyles[0]}",
        #     label="lucj-compressed-tn-do",
        #     color=colors["lucj_compressed_quimb2"],
        # )
        
    else:
        list_tasks = [tasks_truncated, tasks_compressed_t2]
        color_keys = ["lucj_truncated", "lucj_compressed"]
        labels = ["LUCJ-truncated", "LUCJ-compressed"]

    for tasks, color_key, label in zip(list_tasks, color_keys, labels):
        results = {}
        for task in tasks:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results[task] = load_data(filepath)

        errors = [results[task]["error"] for task in tasks]
        sci_vec_shape = [results[task]["sci_vec_shape"][0] for task in tasks]
        
        axes[0, i].plot(
            n_reps_range,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )
        if color_key == "lucj_compressed":
            print("lucj_compressed")
            print([results[task]["energy"] for task in tasks])
        if color_key == "lucj_compressed_quimb":
            print("lucj_compressed_quimb")
            print([results[task]["energy"] for task in tasks])

        axes[2, i].plot(
            n_reps_range,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )


    axes[0, i].set_title(f"{connectivity}")
    axes[0, i].set_yscale("log")
    axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(n_reps_range)

    axes[1, i].set_ylabel("COBYQA Iter")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(n_reps_range)
    axes[1, i].set_yticks(range(1,12,2))
    
    axes[2, i].set_ylabel("SCI subspace")
    axes[2, i].set_xlabel("Repetitions")
    axes[2, i].set_xticks(n_reps_range)

    leg = axes[2, 1].legend(
        bbox_to_anchor=(-0.25, -0.48), loc="upper center", ncol=6,
        columnspacing=1, handletextpad=0.8
    )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16, top=0.86)

    fig.suptitle(
        f"Fe2S2 ({nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()