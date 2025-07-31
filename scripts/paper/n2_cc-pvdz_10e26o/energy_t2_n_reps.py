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
from molecules_catalog.util import load_molecular_data
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


def init_loss(n_reps: int, bond_distance):
    mol_data = load_molecular_data(
        f"{molecule_basename}_d-{bond_distance:.5f}",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )
    norb = mol_data.norb
    nelec = mol_data.nelec
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        mol_data.ccsd_t2,
        n_reps=task.lucj_params.n_reps,
        t1=mol_data.ccsd_t1 if task.lucj_params.with_final_orbital_rotation else None,
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


fig, axes = plt.subplots(
    2,
    len(bond_distance_range) * len(connectivities),
    figsize=(10, 5),  # , layout="constrained"
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
    if bond_distance == 1.2:
        energy_ground_truth = -109.20854905
    else:
        energy_ground_truth = -108.94168735

    axes[0, i].axhline(
        results_uccsd[bond_distance]['energy'] - energy_ground_truth,
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

    axes[0, i].axhline(
        results['energy'] - energy_ground_truth,
        linestyle="--",
        label="LUCJ-full",
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

    tasks_compressed_t2_40 = [
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
                begin_reps=40,
                step=4
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
    
    list_tasks = [tasks_truncated, tasks_compressed_t2, tasks_compressed_t2_40]
    color_keys = ["lucj_truncated", "lucj_compressed", "lucj_compressed_1stg"]
    labels = ["LUCJ-truncated", "LUCJ-compressed", "lucj_compressed-40/4"]

    for tasks, color_key, label in zip(list_tasks, color_keys, labels):
        results = {}
        for task in tasks:
            if color_key == "uccsd-compressed":
                filepath = DATA_ROOT / task.dirpath / "data_uccsd.pickle"
            else:
                filepath = DATA_ROOT / task.operatorpath / "data.pickle"
            results[task] = load_data(filepath)

        errors = [results[task]["energy"] - energy_ground_truth for task in tasks]
        
        axes[0, i].plot(
            n_reps_range,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )

    list_tasks = [tasks_compressed_t2, tasks_compressed_t2_40]
    list_loss = [[], [], []]

    for n_reps in n_reps_range:
        list_loss[0].append(init_loss(n_reps, bond_distance))

    for task in tasks_compressed_t2:
        filepath = DATA_ROOT / task.operatorpath / "opt_data.pickle"
        results = load_data(filepath)
        list_loss[1].append(results["final_loss"])
    for task in tasks_compressed_t2_40:
        filepath = DATA_ROOT / task.operatorpath / "opt_data.pickle"
        results = load_data(filepath)
        list_loss[2].append(results["final_loss"])

    color_keys = ["lucj_truncated", "lucj_compressed", "lucj_compressed_1stg"]
    labels = ["LUCJ-truncated", "LUCJ-compressed", "lucj_compressed-40/4"]
    # for loss, color_key, label in zip(list_loss[1:], color_keys[1:], labels[1:]):
    for loss, color_key, label in zip(list_loss, color_keys, labels):
        axes[1, i].plot(
            n_reps_range,
            loss,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )

    axes[0, i].set_title(f"R={bond_distance} Å / {connectivity}")
    axes[0, i].set_yscale("log")
    axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(n_reps_range)

    axes[1, i].set_ylabel("Operator loss")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(n_reps_range)
    axes[1, i].set_yscale("log")

    leg = axes[1, 2].legend(
        bbox_to_anchor=(-0.3, -0.25), loc="upper center", ncol=5
    )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.88)

    fig.suptitle(
        f"$N_2$ (6-31g, {nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()