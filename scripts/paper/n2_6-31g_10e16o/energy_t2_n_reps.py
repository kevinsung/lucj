import itertools
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.uccsd_task.lucj_compressed_t2_task import UCCSDCompressedTask
from lucj.uccsd_task.uccsd_sqd_initial_params_task import UCCSDSQDInitialParamsTask
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
    figsize=(10, 3),  # , layout="constrained"
)

results_uccsd = {}
for d in bond_distance_range:
    task = UCCSDSQDInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
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
    filepath = DATA_ROOT / task.vqepath / "data.pickle"
    results_uccsd[d] = load_data(filepath)

for i, (bond_distance, connectivity) in enumerate(itertools.product(bond_distance_range, connectivities)):
    axes[i].axhline(
        results_uccsd[bond_distance]['error'],
        linestyle="--",
        label="UCCSD",
        color=colors["uccsd"],
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

    filepath = DATA_ROOT / task_lucj_full.operatorpath / "data.pickle"
    results = load_data(filepath)

    axes[i].axhline(
        results['error'],
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )

    if connectivity == "all-to-all":
        # UCCSD
        tasks_uccsd_compressed_t2 = [UCCSDCompressedTask(
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
                regularization=False
            )
            for n_reps in n_reps_range
        ]
    
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

    if connectivity == "all-to-all":
        list_tasks = [tasks_uccsd_compressed_t2, tasks_truncated, tasks_compressed_t2]
        color_keys = ["uccsd-compressed", "lucj_truncated", "lucj_compressed"]
        labels = ["UCCSD-compressed t2", "LUCJ-truncated", "LUCJ-compressed"]
    else:
        list_tasks = [tasks_truncated, tasks_compressed_t2]
        color_keys = ["lucj_truncated", "lucj_compressed"]
        labels = ["LUCJ-truncated", "LUCJ-compressed"]

    for tasks, color_key, label in zip(list_tasks, color_keys, labels):
        results = {}
        for task in tasks:
            if color_key == "uccsd-compressed":
                filepath = DATA_ROOT / task.dirpath / "data_uccsd.pickle"
            else:
                filepath = DATA_ROOT / task.operatorpath / "data.pickle"
            results[task] = load_data(filepath)

        errors = [results[task]["error"] for task in tasks]
        
        axes[i].plot(
            n_reps_range,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )

    axes[i].set_title(f"R={bond_distance} Å / {connectivity}")
    axes[i].set_yscale("log")
    axes[i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[i].set_ylabel("Energy error (Hartree)")
    axes[i].set_xlabel("Repetitions")
    axes[i].set_xticks(n_reps_range)

    leg = axes[2].legend(
        bbox_to_anchor=(-0.3, -0.25), loc="upper center", ncol=5
    )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, top=0.82)

    fig.suptitle(
        f"$N_2$ (6-31g, {nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()